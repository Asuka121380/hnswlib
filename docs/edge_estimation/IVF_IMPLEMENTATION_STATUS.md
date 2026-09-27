# IVF coarse-residual edge integration — 验收状态

2026-09-27。工作区 `hnswlib-unified-estimation`，分支 `ivf-hnsw-integration`，在原 `abb7251` 代码基础上增量实现。本记录固定本地验收状态；尚未启动正式集群作业。用户原有未跟踪文件 `scripts/edge_estimation/plot_recall_qps_results.py` 未修改，也不纳入本次 IVF 提交。

## 已完成

- 调查 Faiss v1.15.1 的 IVF/PQ 源码、OPQ/QJL 原论文及现有 HNSW 剪枝接口；澄清 9/24 与 9/27 文档的架构分歧。
- 实现共享粗中心/全图 assignment/训练边池；PQ、OPQ 从粗残差重新训练；PQ+QJL 复用同一 PQ 模型，重新生成原始边尺度残差 sketch。
- 实现 `uq-ivf-edge/1` artifact、分块编码、full-graph coverage、显式内部ID映射、base/index 向量逐块一致性检查、hash与发布标记校验。
- 接入通用 native replay、质量评价、scalar/batch parity、调试逐事件分数和真实 HNSW ActiveEdgePruner。
- 实现独立性能 runner：同二进制 HNSW baseline 不加载量化 artifact；计入 query prep；限制当前批次 scratch；输出模型/目录内存、peak RSS与分开的延迟口径。
- 实现 train/quality/grid/summarize/refine、Slurm worker、57点 pilot与72点可选旧方法控制组；从实际新前沿派生 high-recall/batch/3-repeat 配置。

## 实际验证

| 验证 | 结果 |
|---|---|
| GCC 16.1/UCRT64 C++17 + OpenBLAS 原生构建 | 通过 |
| 12项原生回归（policy、artifact、query、rotation、capture、active hook等） | 12/12通过 |
| Python统一基础设施/图表回归（MSYS Python） | 26/26通过 |
| 新Faiss训练→编码→native→HNSW→实验编排集成测试 | 通过；亦已注册可选CTest并执行通过 |
| 小图三方法独立float64几何oracle | 每方法259个候选；最大绝对误差 PQ=7.75e-7、OPQ=1.03e-6、PQ+QJL=7.77e-7 |
| D960/M32b8/QJL128 handcrafted oracle（BLAS） | 最大误差 PQ=1.98e-8、OPQ=4.20e-8、PQ+QJL=2.03e-7 |
| D960同一oracle（无BLAS） | 最大误差 PQ=3.15e-8、OPQ=5.16e-8、PQ+QJL=1.10e-6 |
| scalar/batch | 绝对1e-4、相对1e-5容差内；无非有限状态不一致 |
| 零长边、非恒等mapping、非恒等rotation、QJL bit order/scale/source offset | 通过 |
| 无剪枝top-k与精确基线一致、激进beta实际触发剪枝 | 通过 |
| PQ与PQ+QJL主码/center ID逐字节一致 | 通过 |
| 截断记录、修改config、错catalog、改mapping | 拒绝 |
| 严格模式校准/evaluation query重叠 | 拒绝 |
| 小规模 train→quality→7点grid→summary→refine | 通过 |
| 可选PNG/PDF绘图 | 已在有Matplotlib的独立环境执行 |

训练环境为 Python 3.12.14 / NumPy 2.5.3 / Faiss 1.15.1。该环境没有Matplotlib；图表回归和出图改用已有 MSYS Python / NumPy 2.4.6 / Matplotlib 3.11.0，无新增安装。最初完整 unittest 在训练环境因缺Matplotlib报错，随后在图表环境完整通过。

详细本地报告：`build-ivf/ivf-verified/test_report.json`；无BLAS报告：`build-ivf-noblas/dimension960/report.json`。摘要另存 `ivf_validation_summary.json`。测试产生的synthetic QPS仅用于管线验证，不能与真实GIST成绩比较。

## 尚未执行及口径限制

- 未在 GIST1M 全图重训/编码或测量 Recall–QPS；不存在可报告的新方法性能提升。
- 本地已确认有GIST base/query/mapping；正式冻结索引和ground truth的历史记录指向集群，未核实远程当前状态。未自动连接集群或提交Slurm。
- 72点配置中的旧PQ/OPQ/PQ+QJL控制组通过既有CLI组装；本轮未用真实legacy sidecar/companion运行该附加矩阵。
- Linux原生构建/运行尚需在目标节点验收；本地无BLAS与BLAS构建均通过，不代表所有目标ISA已测。
- 实现采用共享粗中心＋残差**边**表示，保持HNSW候选生成；没有实现标准倒排IVF-Flat/IVF-PQ搜索，没有nprobe。
- 全局质量/逐query误剪来自冻结baseline trace，不是主动HNSW全轨迹误剪保证；主动性能输出Recall、估计/剪枝/回退等计数。
- `eligible_exact_distance_count`不含upper layers/入口，baseline未采样为null；latency摊销与批内完成服务时间不含排队等待。
- 只有1000条GIST query；默认全1000 pilot是exploratory，不能同时声称独立held-out test。

命令、参数、输入布局及后续实验步骤见 `scripts/edge_estimation/IVF_RUNBOOK.md`。
