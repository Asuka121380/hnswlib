# 边向量量化剪枝：最终实验实现与运行说明

本分支实现 2026-10-08 实验实施计划的 S0–S8 流程。主方法名为 OPQ 和 CR-OPQ；代码/文件中的 `ivf_opq` 是 CR-OPQ，含粗中心与残差量化，但搜索仍是同一个 HNSW 图，不做 IVF list 路由。最终采用哪种方法由冻结后的真实 QPS/Recall 结果决定。

## 已实现的公平性约束

- 同一套 Release、O3、native OFF、严格浮点（`-fno-fast-math -ffp-contract=off`）、BLAS ON 构建。两个 runner 使用同一份 `performance_loop.h`。HNSW 使用 OPQ runner 的 `--no-prune`，不加载量化资产。
- 正式构建关闭 capture 和旧 V0 诊断/剪枝 hook；capture/build 工具另建目录。编译命令、CMake cache、源文件内容、二进制及 Linux 动态依赖均记录完整 SHA256。Windows smoke 构建记录显式 BLAS 链接文件，正式实验仅支持 Linux/Slurm。
- 搜索和 BLAS 均单线程。正式 worker 要求独占单节点，显式绑定到一个逻辑 CPU；三次独立 Slurm allocation，每次三个串行配对 block。每个 block 的方法顺序随机化，全部方法使用相同的查询顺序。分析拒绝不同 CPU 型号/ISA 混合。
- 主指标是 `QPS = N_unique_queries × inner_repeats / service_wall`。service wall 包含每轮窗口准备、搜索、结果写入预分配内存；不包含磁盘输出、GT 评估、加载和哈希。记录各项时间及剩余时间。指标关闭时使用编译期不累加剪枝计数的模板。
- 实际计时搜索直接保存全部 repeat/query/rank/label/distance，随后验证完整覆盖、合法标签、距离顺序和跨重复一致性；不重新搜索生成 recall。结果缓冲区在计时外分配，默认上限 2 GiB；超过上限即失败。
- 独立诊断模式通过包装原 L2 距离函数统计真实调用数，分类记录阈值不可用、零边长、非有限估计、catalog 不符和 backend 异常。后两者导致失败；hop 暂无可靠 hook，明确输出 null。
- 资产校验逐块比较 index 中向量和 base/mapping，并分别验证外部标签与 GT 的标签空间。OPQ 和 CR-OPQ 共用 100,000 条有效非零边、seed 42、M=32、8 bit、相同 Faiss 训练预算；CR 粗中心 K=256。OPQ 内部采样上限显式设置，禁止悄悄减少训练样本。
- 所有文件（包括大图/模型/结果）都完整哈希。cases、batch choice、freeze 不覆盖。恢复必须核对输入、源码、构建、查询顺序、每个输出哈希；未完成目录保留，不能假装成功续跑。

## 入口与配置

命令均在仓库根目录执行，使用安装了 NumPy、Faiss 的 Python；绘图另需 matplotlib。Linux 还需要 GCC/Clang、CMake、Ninja、OpenBLAS、Slurm。Windows 本地测试使用 MSYS2 UCRT64 编译器和 OpenBLAS。

- `configs/edge_estimation/final_study/protocol.json`：实际实验默认预算与搜索网格。
- `registry.example.json`：GIST1M、SIFT1M、可选 DEEP1M 数据源模板。路径必须改为运行机器上的绝对路径，模板不是已核实数据清单。
- `splits.example.json`：GIST dev/select 各 500，test 5,000；SIFT/DEEP dev 2,000、select 2,000、test 6,000。每个 dev 预留 100 条 warmup，开发测量会排除这 100 条，select/test 与 warmup 不重叠。
- `scripts/datasets/final_study_data.py`：inventory、audit-queries、materialize、exact-gt、validate。
- `scripts/edge_estimation/final_study/`：build、prepare_assets、sample_edges、asset_contract、make_cases、run、select、recommend、freeze、validate_run、analyze、cost、plot。

正式实验默认 GIST + SIFT 必做，DEEP 可选。若官方查询历史已全部使用，SIFT 官方 6,000 条不能直接当作新 test；必须换成有完整审计证据的未用来源并修改 split 配置。GIST learn 的已用区间（例如此前记录的 [10000,13000)）必须在 ledger 标记 used。脚本不会因“没有搜索到历史记录”推断 fresh，也不会自动下载或虚构数据来源。

## S0–S1：编译与环境冻结

```bash
export PYTHON=/absolute/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
bash scripts/edge_estimation/final_study/build.sh --out build-final-tools --tools --blas /absolute/libopenblas.so
bash scripts/edge_estimation/final_study/build.sh --out build-final-perf --blas /absolute/libopenblas.so
```

每个目录产生 `build_manifest.json`。如源码改动，必须重新 build 并重新生成绑定该 build 的 cases/freeze；不能拿旧 freeze 运行新二进制。显式 build flags 防止旧 CMake cache 遗留其他剪枝 hook。

Slurm 会复制 batch 脚本，不能用执行时的 `BASH_SOURCE` 推算仓库位置。worker 优先使用导出的 `REPO`；未设置时使用当前目录所属的 Git 工作树，并检查必要入口文件。`submit_final_study.sh` 会显式导出自身所属仓库并设置 `sbatch --chdir`；自定义提交器也应先导出绝对路径 `REPO`，使用 `--export=ALL --chdir="$REPO"`。worker 另通过 `srun --chdir` 保证 Python 从同一仓库启动。

无需集群资源的回归检查：`python tests/python/final_study_slurm_wrapper_test.py -v`。该检查模拟脚本被复制到 Slurm spool 目录，覆盖明确 REPO、当前 Git 工作树、错误路径拒绝及从其他目录提交；实际独占分配和 CPU 绑定仍需在集群验收。

### 集群 CPU 绑定验收

2026-10-09 的 weirdo 诊断作业 150339 显示，本站 TaskPlugin 为 task/cgroup，srun 请求 --cpu-bind=cores 后 Python 初始 affinity 仍有 192 个逻辑 CPU。NumPy/Faiss 导入和小计算没有改变它；os.sched_setaffinity 显式缩小到一个 CPU 后，子进程及其库初始化保持同一绑定。

worker 保留 Slurm 资源申请与绑定参数，并在 **srun 内部**通过标准库入口 `affinity.py` 读取允许的 CPU 集合，选择其中编号最小的 CPU，设置并回读校验。随后 exec 启动测量 driver，先绑定再加载数值库；所有方法的子进程继承同一绑定。不得在登录节点执行绑核或写死 CPU 0。OpenBLAS 的主线程自行绑核通过 OPENBLAS_MAIN_FREE=1 关闭。

正式 runtime 必须实际只有一个逻辑 CPU；每次启动 runner 前以及发布完整结果前再次检查 affinity 与初始记录一致。结果读取器同样拒绝非 local_smoke 的宽掩码或缺失 affinity 记录，包括旧的 150335 合成记录；旧目录保留作诊断证据。线程数为 1 与 CPU affinity 为单元素是两项独立条件，不能只检查 affinity 非空。

无需 Slurm 的测试：
```bash
"$PYTHON" tests/python/final_study_affinity_test.py -v
"$PYTHON" tests/python/final_study_slurm_wrapper_test.py -v
```

Windows 使用模拟 OS 接口测试失败拒绝和调用顺序，实际 Linux 子进程继承测试明确跳过；Linux 会运行此测试，且 wrapper fixture 使用真实绑核入口。发布前仍需真实独占作业验证，严格要求 run_manifest 的 runtime.affinity 只有一个元素，且所有 18 个 synthetic case/block 结果通过完整性校验。150339 仅证明绑核方案可行，不能代替修复后 worker 的完整验收。源码变更后刷新双构建清单，在新目录生成合成资产和 cases，不能修改旧 manifest 的 hash。

## S2：数据清理、历史审计与精确 GT

复制并修改 registry/splits 模板，保留副本作本次研究输入。history.json 示例：

```json
{"roots":["/absolute/old-runs"],"review_ledger":"/absolute/review-ledger.json"}
```

review-ledger.json 是列表。每项要给 dataset_id、source、源文件 SHA256、status（used / uncertain / unused_confirmed）、零起始半开 range 或 ids、evidence 文件路径。unused_confirmed 还必须有 `review_complete:true`。evidence 是实际历史核查记录；不能为了让脚本通过而直接宣称未用。已用/不确定记录、不同源中的重复向量、与 base 相同的向量均优先排除 test。

```bash
$PYTHON -m scripts.datasets.final_study_data inventory --registry /data/registry.json --out /study/inventory
$PYTHON -m scripts.datasets.final_study_data audit-queries --registry /data/registry.json --history-roots /study/history.json --out /study/audit
$PYTHON -m scripts.datasets.final_study_data materialize --dataset gist1m --audit /study/audit/audit.json --split-spec /study/splits.json --out /study/gist/data
for split in dev select test; do
  $PYTHON -m scripts.datasets.final_study_data exact-gt --dataset-manifest /study/gist/data/dataset_manifest.json --split "$split" --k 100 --threads 8
done
$PYTHON -m scripts.datasets.final_study_data validate --dataset-manifest /study/gist/data/dataset_manifest.json --float64-check-queries 100
```

输出 compact query pool、原 source/row 映射、disjoint split ID、每个 split 的 compact GT 与 query-to-GT-row 映射，以及 `dataset_contract.json`。GT 使用分块 Faiss IndexFlatL2 全扫描 float32，额外对抽样查询做 float64 全扫描 top-10 校验；发现差异就失败，需在新数据版本用 `--backend float64` 重建。float64 tie-breaking 按距离、base row 排序；float32 边界 ties/舍入差异需要核查，不能忽略校验失败。

DEEP 的 fbin 按 little-endian uint32(N,D)+float32 数据读取，转换并截取前 1M base；所有 GT 重新计算，不复用 10M GT。

## S3：共享图、边样本与模型

```bash
$PYTHON -m scripts.edge_estimation.final_study.prepare_assets \
  --protocol configs/edge_estimation/final_study/protocol.json \
  --dataset /study/gist/data/dataset_contract.json \
  --tools-build build-final-tools/build_manifest.json --out /study/gist/assets
```

新图使用 M=16、efConstruction=200、seed=42、单线程、base 行号外部标签。也可给 `--reuse-assets /.../assets.json --reuse-catalog /.../catalog` 复用已验证的现有图；必须提供明确 internal-to-base 的 .npy 映射，即使它是恒等映射。复用不会冒称新图构建时间或单线程历史；原始资产配置作为证据保留。

训练顺序：有效边统一抽样 → OPQ → 粗中心训练/全边 assignment → CR-OPQ。量化维度、seed、线程、迭代数、重启数与采样上限逐项核对。默认训练预算：outer 25、initial PQ 25、inner PQ 25、redos 1、max_train_points 100000、max_points_per_centroid 391。CR 残差不二次归一化。输出 `asset_contract.json`、各阶段日志及 `offline_costs.json`。

## S4–S5：开发扫描、批处理选择与冻结

以下 Bash 数组可直接复用，所有路径需替换。

```bash
common=(--protocol "$PWD/configs/edge_estimation/final_study/protocol.json"
        --dataset /study/gist/data/dataset_contract.json
        --assets /study/gist/assets/asset_contract.json
        --build "$PWD/build-final-perf/build_manifest.json")
$PYTHON -m scripts.edge_estimation.final_study.make_cases "${common[@]}" --phase dev --out /study/gist/dev-cases.json
```

在申请好的 exclusive 单节点 allocation 内运行；worker 本身不申请资源：

```bash
bash scripts/edge_estimation/final_study/worker.sh "${common[@]}" --cases /study/gist/dev-cases.json --out /study/gist/dev --blocks 3
$PYTHON -m scripts.edge_estimation.final_study.make_cases "${common[@]}" --phase batch --dev-runs /study/gist/dev --out /study/gist/batch-cases.json
bash scripts/edge_estimation/final_study/worker.sh "${common[@]}" --cases /study/gist/batch-cases.json --out /study/gist/batch --blocks 3
$PYTHON -m scripts.edge_estimation.final_study.select --runs /study/gist/batch --out /study/gist/batch-choice.json
$PYTHON -m scripts.edge_estimation.final_study.make_cases "${common[@]}" --phase select --dev-runs /study/gist/dev --batch-choice /study/gist/batch-choice.json --out /study/gist/select-cases.json
bash scripts/edge_estimation/final_study/worker.sh "${common[@]}" --cases /study/gist/select-cases.json --out /study/gist/select --blocks 3
```

GIST/SIFT 的 select 都完成后，从全部预注册数据集的选择集记录做全局方法选型，再为各集冻结：

```bash
$PYTHON -m scripts.edge_estimation.final_study.recommend --runs /study/gist/select /study/sift/select --out /study/method-selection.json
$PYTHON -m scripts.edge_estimation.final_study.freeze --runs /study/gist/select \
  --method-selection /study/method-selection.json --dev-runs /study/gist/dev \
  --out /study/gist/freeze.json --diagnostic-out /study/gist/diagnostic-freeze.json \
  --mechanism-out /study/gist/mechanism-cases.json
```

选型对数据集和 recall target 等权，使用配对 block 的 CR/OPQ 加速比；超过 2%、CI 排除 1 且没有单点超过 2% 的退化才推荐全局方法，否则明确记录“没有稳定全局赢家”。若 DEEP 纳入本轮，在 protocol 的 required_datasets 预先登记，并加入该命令的 select 输入。

- 开发阶段：两种剪枝方法相同 ef/beta 网格；HNSW 无 beta。可在未看 test 的前提下修改协议中的 ef 网格并新建版本，补扫边界，再重做 select。
- 批处理阶段：每种剪枝方法固定相同参数，比较 prepare/compute = 128/128、all/128、all/all，另加一个 HNSW 对照，共 7 配置。select 依据两种方法相对 128/128 的 QPS 几何均值选择共同策略；2% 内优先有限窗口，避免没有明显收益却扩大 LUT 内存。
- 准备窗口决定常驻 LUT 峰值；计算块决定 SGEMM 粒度。all/128 只分块计算，但全体查询的 LUT 仍常驻，所以不是有限内存方案。all/all 可能提高矩阵效率，也可能增加峰值内存，不能先验认定最快。
- select 每个 method/target 从 dev 提名 anchor 和同 beta 的上邻 ef，去重后执行；默认 target 95%、97%、98%。变异系数超过 5% 时选择工具拒绝三块数据，需在新目录跑满九块，再以九块 run 作依据。不要把失败的三块重复计入九块。
- protocol 预先登记 `batch_scope=per_dataset_shared`：允许数据集之间策略不同，但同一数据集的三种方法使用共同策略；这属于原计划允许的事前工程选择。scratch 预算默认 1 GiB；select 还拒绝某方法稳定退化超过 2% 的替代策略。
- freeze 用最快 select case 的性能估计本数据集共同的 inner repeats，使最快正式 case 约 30 秒，其余 case 运行相同次数；冻结前检查结果缓冲区以及 test 查询数量下的 scratch 容量。冻结后不依据 test 改 beta/ef/batch/R。正式 recall 未达目标时仅允许使用预先冻结的上邻，否则标记 NOT_REACHED。

## S6–S7：独占正式 QPS 与诊断

```bash
bash scripts/edge_estimation/submit_final_study.sh PARTITION ACCOUNT 08:00:00 /study/gist/test \
  "${common[@]}" --cases /study/gist/freeze.json
```

此命令确实提交三个 Slurm job；本次实现工作没有执行该命令。每个 job 执行三块，输出 allocation-0/1/2。正式测量强制 metrics off，诊断单独跑：

```bash
bash scripts/edge_estimation/final_study/worker.sh "${common[@]}" --cases /study/gist/diagnostic-freeze.json --out /study/gist/diagnostic
bash scripts/edge_estimation/final_study/worker.sh "${common[@]}" --cases /study/gist/mechanism-cases.json --out /study/gist/mechanism
```

M1 diagnostic 使用冻结的 97%/98% 主锚点、test 的同一查询集合与正式第一个 block 的顺序，只运行一遍，校验其每个 query 的 top-k 签名与关闭计数时完全相同。M2 mechanism 在 select 中固定至多 1,000 条查询，beta 默认共同 1.30，ef 来自 dev HNSW 最接近 97%/98% 的点，两个量化方法完全相同，只作单遍诊断。两种诊断 QPS 都不能混进主速度表。机器状态、亲和性、governor、进程快照和 Slurm 作业信息写入 run manifest。一次 run 内任何失败都不发布顶层 complete。

本地调试可直接使用 `python -m ...final_study.run ... --local-smoke`。这种结果及其派生 freeze 永久标记 local_smoke，正式 run 不接收它；本地 QPS 不能用于论文结论。

## S8：汇总、置信区间、成本与图

```bash
$PYTHON -m scripts.edge_estimation.final_study.analyze --freeze /study/gist/freeze.json \
  --runs /study/gist/test/allocation-0 /study/gist/test/allocation-1 /study/gist/test/allocation-2 \
  --diagnostics /study/gist/diagnostic --mechanisms /study/gist/mechanism --out /study/gist/analysis.json
$PYTHON -m scripts.edge_estimation.final_study.cost --assets /study/gist/assets/asset_contract.json \
  --analysis /study/gist/analysis.json --ledger /study/gist/assets/offline_costs.json --out /study/gist/cost.json
$PYTHON -m scripts.edge_estimation.final_study.plot --analysis /study/gist/analysis.json --out /study/gist/figures
```

每个 recall target 按预注册规则选 anchor/上邻。速度比使用 allocation → paired block 的分层 bootstrap；Recall CI 以同一组查询索引配对重采样所有方法，默认 10,000 次，不把 inner repeats 当作新样本。同时报告 recall 均值、配对 recall 差异 CI、下界是否达到目标以及 NOT_REACHED。输出完整原始 case、冻结锚点、注册曲线、诊断/机制的 CSV 表与 reproduction manifest，成本也输出 online/offline CSV。

成本拆分粗中心训练、assignment、OPQ/残差 OPQ 训练、编码；额外提供实际子进程 wall ledger。48/52 bytes per edge 的默认记录布局由实际文件大小核验；模型/目录/查询/结果缓冲区/scratch/进程峰值 RSS 分列。成本摊销只在同目标双方达标、且查询时间确有节省时计算，并明确哪些共享成本未计入。跨数据集按各自真实数据与独立协议运行全部上述环节。

## 验证与兼容性

```bash
$PYTHON tests/python/final_study_integration_test.py --build build-final-tools
$PYTHON tests/python/uq_ivf_integration_test.py --build build-final-tools
```

新增集成测试使用合成数据、真实 Faiss 和原生 C++ runner，覆盖数据审计、float64 GT 校验、单线程建图、同样本训练、三种 batch、开发/选择/冻结、三组配对结果、真实距离计数、恢复与损坏输出拒绝、非连续 ID 和 inner repeats。测试产物默认写系统临时目录。

旧 `--batch-size` 映射到 prepare_window=compute_chunk，保留 legacy 标记；不能与新参数混用。新 CSV 增加 repeat_id，旧 recall 读取器已兼容并加强完整性检查。旧 grid 对大文件改用全量 SHA256，旧 resume 必须有新的完整封印，旧汇总对成本字段一起聚合，baseline-only 不再要求 CR 资产。正式论文实验只使用 final_study 新入口。

尚需在真实运行环境完成：历史查询审计、数据路径核实、GIST/SIFT（以及可选 DEEP）正式训练和 Slurm 独占实验。这些是实验执行工作，不是本地 smoke 已验证的性能结论。

### 本次本地验证（2026-10-08）

- Windows / GCC 16.1.0 / Python 3.12.14 / NumPy 2.5.3 / Faiss 1.15.1；工具与 capture-OFF 性能构建均通过。
- 新 final_study 完整集成测试通过，包含 recommend、单遍 M1/M2、共同 R、配对统计、CSV 和成本输出；旧真实 Faiss 集成测试通过。
- 5 个核心 C++ 测试、3 个 Python 训练测试和旧 Recall-QPS 读取测试通过；Slurm 脚本通过 Bash 语法检查。
- capture-OFF 二进制的本地测量通过；未加 local-smoke 的 Windows 正式运行被正确拒绝。
- 已在项目虚拟环境补装 matplotlib 3.11.2，PNG/PDF 图表导出通过。未执行 Slurm 提交，也未产生真实数据集的正式性能结论。
