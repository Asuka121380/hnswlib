# IVF 粗量化与 PQ / OPQ / PQ+QJL 接入 HNSW

日期：2026-09-27。本文区分设计、实现验收与正式实验；本地 correctness 不能证明 GIST1M 的性能提升。

## 1. 冻结问题与架构

9/24 的 compatibility reassessment §4.6、基础设施手册 §12 要求保留 HNSW 图，使用 IVF 的粗中心＋残差编码结构。9/27 的 HORIZONTAL_COMPARISON_AND_IVF_PLAN §5–6 则转成了 base residual、倒排表和 nprobe。两者不是同一实验。本轮依照“HNSW 量化剪枝”的目标采用前者，准确名称为 **HNSW + coarse-residual edge quantization**，短名 ivf_pq / ivf_opq / ivf_pq_qjl。它借用 IVFADC 的两级表示，没有倒排候选路由，不应宣称实现了标准 IndexIVFPQ 搜索。

搜索保留原索引 M=16、efConstruction=200、seed=42，upper layers、visited 语义、候选堆和最终精确距离沿用现有实现。仅在 layer-0 eligible neighbor 的精确距离之前调用 ActiveEdgePruner。未满堆、零长边、非有限估计均回退；严格 estimate > beta * threshold 才剪枝。被保留候选仍计算 exact L2，不设 ADC top-R rerank，不另引入 nprobe。误剪会改变后续搜索路径，因此冻结 trace 与主动搜索必须分别评估。

## 2. 论文与开源实现调查

- Jégou et al., Product Quantization for Nearest Neighbor Search, TPAMI 2011：[作者论文入口](https://inria.hal.science/inria-00514462)。IVFADC 将粗量化和残差 PQ 组合；论文入口在本次访问触发防爬，具体代码行为以下列官方源码为依据。
- [Faiss v1.15.1 IndexIVF.cpp](https://github.com/facebookresearch/faiss/blob/v1.15.1/faiss/IndexIVF.cpp)：粗聚类、assignment 和选桶/列表扫描属于不同步骤。这里只复用前两个步骤的思想。
- [Faiss v1.15.1 IndexIVFPQ.cpp](https://github.com/facebookresearch/faiss/blob/v1.15.1/faiss/IndexIVFPQ.cpp)：by_residual 默认打开；encode_vectors 先 compute_residuals 再 pq.compute_codes；decode_multiple 解码 PQ 后加回粗中心。train_encoder 在传入的残差上训练 PQ。add_core_o 的 residuals_2 是主编码后剩余误差，与 QJL 的修正目标对应，但本项目用原始边尺度定义该误差。
- [Faiss Implementation notes](https://github.com/facebookresearch/faiss/wiki/Implementation-notes)：标准 IVF L2 的预计算表是空间换建表时间。不能把该扫描表直接套到随机访问图边；本项目使用共享内积表和静态 source offset。
- [Ge et al., OPQ, CVPR 2013](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf)：学习正交变换优化分块量化失真。这里训练的是粗分配后的残差，使用单个全局 R；不是先对全部 base 做 OPQ 再聚类的另一种管线。
- [QJL 论文](https://arxiv.org/abs/2406.03482)、[作者代码](https://github.com/amirzandieh/QJL)：采用高斯投影＋符号的非对称内积估计；作者任务是 KV cache，本项目只采用数学估计器，不复用其搜索系统，也不将固定向量对的无偏性说成自适应 HNSW 的 recall 保证。

本地依据：tools/edge_estimation/live_opq_pruner.h、backends/rotated_pq.h、hnswlib/edge_estimation/active_policy.h、scripts/edge_estimation/train_encode.py 以及 trainers/faiss_{pq,opq}.py。原 backend 量化单位边方向并预存 source anchor，必须保持这一几何契约。

## 3. 量化目标选择与完整公式

可选目标有 base vector、原始边 e=v-c 和单位方向 u=e/ell。base residual 可将 O(E) 记录降至 O(N)，但改变当前距离估计器/内存预算；原始边会使长边支配 MSE。本轮选单位边方向以延续横向对比，之后可将节点编码作为独立实验。

令 a_j 为 u 最近的粗中心，r=u-a_j（不归一化），PQ 重构 rhat；OPQ 使用 rhat=R^T decode(PQ(Rr))。定义 h=a_j+rhat：

    Dhat = D(q,c) + ell^2 - 2 ell (q-c)^T h
         = D(q,c) + offset - 2 ell [q^T a_j + (Rq)^T decode]
    offset = ell^2 + 2 ell c^T h

PQ 的 R 为恒等映射。静态 source 项写入 float64 offset；query 只需一次 rotation、一套 M×K 内积表和粗中心点积，不需要 query-source 或 query×nprobe 次旋转。所有中心查询表采用 batch GEMM；成本为 O(B Kc D)，须单独计入 query setup；Kc 大未必更快。第一版不做不可靠的粗层提前剪枝。

QJL 必须编码 **z=(v-c)-ell*h**，而不是 r 或 rhat。令 G 为独立 N(0,1) 的 b×D 矩阵，s=sign(Gz)，scale=||z|| sqrt(pi/2)/b：

    Dhat_QJL = Dhat - 2 scale sum_i (Gq)_i s_i + 2 c^T z

这样恢复原始边尺度，并补偿浮点 unit/rotation/reconstruction 误差。2c^Tz 合入 offset；查询只投影 q，不在线投影 source。b 个符号按小端位序打包，每 4 位构造 16 项 signed-sum LUT，将每边 b 次运算降为 b/4 次查表。零 z 的 scale=0。QJL 不应与旧 companion 混用：模型、粗中心、投影、记录一并封装和绑定。

理想高斯模型下修正误差方差随 ||q||²||z||²/b 缩放；用 q 而不是 q-c 的预计算形式节省 source projection，却可能增加方差。需由误剪曲线检验这一实现取舍，不能只引用 q-c 形式的方差。offset 使用真实 c^Tz 保证期望正确。

## 4. 训练、身份与文件布局

coarse stage 只训练一次；三方法读取同一 coarse bundle、sample IDs 和全图 assignments。残差不归一化。PQ 与 PQ+QJL 共享同一 PQ 模型/码，QJL 只扩展 companion；OPQ 从同一 residual pool 单独训练。矩阵原样保存，seed 不代替 artifact identity。

边训练分布有两种明确口径：默认从冻结 base 图均匀无放回采样边，不使用 evaluation queries；这是无监督 index-dependent 训练，不冒称 official learn split。若要求严格 learn-only，应另建 official learn HNSW 图并将其 catalog/base/mapping 作为 training 输入；编码目标仍为冻结 base 图。不能直接用 learn 原始向量训练单位边 residual。脚本支持单独 training assets/catalog，manifest 记录来源。

内部 ID→base row 必须显式映射文件；不能仅靠 node count 相等假设 identity。catalog 的 offsets/targets 和 identity 重算校验；index/base/mapping/coarse/assignment/model/record 均 SHA-256 绑定。bundle 原子目录发布，complete 标记绑定 native.cfg，native.cfg 绑定 manifest，manifest 绑定数据文件。旧格式不可误加载。

粗 bundle：centers.f32le、assignments.u32le、sample_ids.u64le、sample.f32le、manifest.json。编码分块，不构造 E×D 全量浮点矩阵。新 artifact uq-ivf-edge/1：记录 length f64、offset f64、center ID u32、packed PQ codes；QJL 再带 scale f64 与 ceil(b/8) 符号码。D=960,M32b8 时 PQ/OPQ 为 52 B/edge；b=128 的 QJL 为 76 B/edge。中心表 Kc×D×4，OPQ rotation D²×4，PQ codebook D×256×4；加上 catalog、query scratch 和进程 RSS 后才是总内存。

## 5. 实验配置来自既有横向数据

9/27 有 91 个有效点、repeat=1。95% threshold 的 OPQ 410.93 QPS 对 HNSW 287.25，98% 的 PQ+QJL 172.08 对 OPQ 165.58，只作为选候选/选网格的先验，不是新方法成绩。

1. 先固定 Kc=256（限制 query center setup），另外准备 Kc=1024；Kc=4096 只在 quality 收益足以抵偿 setup 后扩大。Kc=1 是去掉粗层自由度的消融；不要把标准 IVF 的 sqrt(N) 经验直接当图边场景最优。
2. M32b8，coarse/train seed=42，QJL b=128，独立 seed=43；batch={1,32,128,256}，主 pilot=128。coarse sample=100000 个有效边，编码 batch=4096；超大训练开销在计算节点进行。
3. PQ beta={1.35,1.45,1.55,1.65}；OPQ={1.20,1.30,1.40,1.50}；PQ+QJL={1.00,1.04,1.08,1.12,1.16}。这是重新标定的扫描起点。
4. efSearch={250,400,600,800}；高 recall 增补={950,1100,1300}，选中 beta 后增补。基线 ef={200,425,600,950,1100}。所有方法 query 前1000、warmup100、repeat1，与旧 pilot 对齐，但旧结果不能代替同机重测。
5. 质量校准用与最终1000个 query 不重叠的开发集，冻结 trace 共用；balanced/global FP<=0.005,p95 query FP<=0.01；strict <=0.001,<=0.0025。若仅有旧已调参 query，则报告 exploratory，不能称 held-out。
6. 最终前沿邻域重复>=3，固定线程/CPU，同节点随机交错；记录 raw points、Pareto、90/95/97/98/98.5 threshold、p50/p95/p99、prep占比、估计/精算/剪枝/fallback、FP及训练时间、模型大小。

## 6. 验收与计时边界

必须验证 Python→C++ 独立 float64 oracle（非单位残差、非恒等 rotation、QJL 位序/scale/offset）、scalar/batch parity、零边 fallback、截断/错hash/错图/错模型拒绝、非identity ID mapping、no-prune 与原始 HNSW top-k parity。PQ/QJL 的 PQ payload 必须相同。静态 quality 使用既有 UQEV evaluator 与 exact labels；诊断不混入性能计时。

在线 runner 计入中心表、rotation、PQ LUT、QJL projection/LUT和HNSW search；不包含加载、哈希、训练和写结果。batch 的 amortized latency 是吞吐分摊服务时间，不是包含排队等待的端到端单请求延迟；同时输出 batch completion latency，batch=1 测单请求。禁止把批次摊销值直接作为交互系统 p95 SLA。

## 7. 范围与后续

本轮交付三种粗残差边 backend、训练/编码/验证/quality/网格/汇总/Slurm入口。标准倒排 IVF-Flat、IVF-PQ 或 HNSW coarse router 是另一个候选生成实验，应另立名字和 nprobe 网格；本轮不以其替换 HNSW。正式 GIST1M 全图训练、集群 QPS 及质量结论以实际运行记录为准。
