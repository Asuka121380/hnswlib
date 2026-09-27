# HNSW 粗中心＋残差量化运行手册

方案见 `docs/edge_estimation/IVF_HNSW_INTEGRATION_PLAN_ZH.md`。这里的 ivf_* 是 **HNSW 边编码**；不使用倒排桶过滤候选，调参是 efSearch 而非 nprobe。

## 1. 实现入口与依赖

| 文件 | 功能 |
|---|---|
| `ivf_train.py` | coarse、encode、validate；两阶段训练，全图分块编码 |
| `ivf_experiments.py` | train、quality、grid、summarize、refine |
| `tools/edge_estimation/backends/ivf_edge.h` | PQ / OPQ / PQ+QJL 原生估计与 batch |
| `uq_ivf_performance_runner` | 真正接入 HNSW ActiveEdgePruner；同二进制 no-prune baseline |
| `uq_native_runner` | 通用 quality、scores-artifact、validate-batch-artifact、bench-batch-artifact |
| `submit_ivf_slurm.sh` | 显式 Slurm worker，固定 BLAS/OpenMP 为单线程 |

训练依赖：Python、NumPy、Faiss CPU；本地测试版本 Python 3.12、NumPy 2.5.3、Faiss 1.15.1（具体 Python 小版本以训练 manifest/dependency 输出为准）。画图可在另一个有 NumPy/Matplotlib 的环境运行，不要求 Faiss。原生 C++17 不链接 Faiss；BLAS 可选，关闭时使用 scalar fallback。Linux 和 Windows 均有代码路径；本次实际构建验收为 Windows UCRT64/GCC，Linux 集群还须重跑测试。

```bash
cmake -S . -B build-ivf -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_BUILD_EDGE_ESTIMATION_TOOLS=ON \
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON \
  -DHNSWLIB_ENABLE_EDGE_ESTIMATION_ACTIVE=ON \
  -DHNSWLIB_ENABLE_EDGE_ESTIMATION_CAPTURE=ON \
  -DHNSWLIB_PERFORMANCE_COMPARABLE_FLAGS=ON -DUQ_WITH_BLAS=ON
cmake --build build-ivf --target uq_native_runner uq_ivf_performance_runner uq_capture uq_capture_fixture -j4
python tests/python/uq_ivf_integration_test.py --build build-ivf --out results/ivf-correctness-001
```

也可配置 `-DUQ_IVF_TEST_PYTHON=/path/to/faiss-enabled/python`，随后运行 `ctest --test-dir build-ivf -R uq_ivf_integration_test --output-on-failure`；省略该变量时不强制基础库测试环境安装Faiss。

构建环境需能找到 BLAS；若指定 `-DUQ_BLAS_LIBRARY=/absolute/path/libopenblas.so`，必须使用实际存在的库。Windows executable 后缀 `.exe`。tests 目录下的 harness 是明确调用的集成测试，不会因为普通 unittest discovery 成功就代表它已执行。

## 2. 冻结输入

先按既有 GIST M16/efConstruction200/seed42 索引导出 catalog：

```bash
build-ivf/uq_capture --catalog --index "$INDEX" --dimension 960 "$CATALOG"
```

创建 assets.json（以下值必须替换为实际路径）：

```json
{
  "index": "/data/gist1m_M16_efc200_seed42.bin",
  "base": "/data/gist_base.fvecs",
  "internal_to_label": "/data/gist1m_internal_to_label.npy"
}
```

`internal_to_label` 沿用旧字段名，但此处实际契约是 internal ID→base 文件行号；external label 不等于 base row 的数据集应提供显式 row mapping。即使恒等映射也必须保存 `.npy`，不按 count 猜测。trainer 会分块逐项核对映射后的 base 向量与索引内向量。当前该审计支持本项目的 hnswlib **64-bit little-endian** 持久化格式；不认识的格式明确拒绝。

catalog complete 中的 index SHA、catalog identity、模型/数据文件 SHA 全部检查。新文件夹运行，不覆盖已有 artifact；失败 partial 留作审计。动态更新 HNSW 后必须重新导出 catalog 和编码。

## 3. 重训全部表示

```bash
python scripts/edge_estimation/ivf_experiments.py train \
  --config configs/edge_estimation/ivf_pilot.json \
  --assets assets.json --catalog "$CATALOG" --out "$ARTIFACT_ROOT"
```

输出 `k256/` 与 `k1024/`，各含共享 `coarse/`、`ivf_pq/`、`ivf_opq/`、`ivf_pq_qjl/`。PQ+QJL 自动读取同目录 PQ 模型，重建基于两级重构误差的 QJL，不重用旧 companion。每个方法完整保存 config、seed、训练 sample 身份、源文件 hash、模型/编码 hash、训练与编码时间、粗分配统计。

默认训练源是冻结 base 图的均匀边样本，无 evaluation query 输入；不是 official learn-only 训练。若需要 learn-only，先在 official learn 上建独立 HNSW，导出它的 catalog 和 mapping，再给 train 增加 `--training-assets learn-assets.json --training-catalog learn-catalog`。不可把 learn 原始向量直接当单位边方向。

单步调试：

```bash
python scripts/edge_estimation/ivf_train.py validate "$ARTIFACT_ROOT/k256/ivf_pq"
```

## 4. 冻结 trace、beta 标定与数值准入

`development_ids.txt` 每行一个 query ID；以同一个 baseline trace 对比三方法：

```bash
build-ivf/uq_capture --index "$INDEX" --queries "$QUERIES" \
  --query-ids development_ids.txt --dimension 960 --k 10 --ef 600 --out "$TRACE"
python scripts/edge_estimation/ivf_experiments.py quality \
  --config configs/edge_estimation/ivf_pilot.json --artifacts "$ARTIFACT_ROOT/k256" \
  --trace "$TRACE" --queries "$QUERIES" --native build-ivf/uq_native_runner --out "$QUALITY"
build-ivf/uq_native_runner validate-batch-artifact \
  "$ARTIFACT_ROOT/k256/ivf_opq" "$TRACE/events.bin" "$QUERIES" 128
build-ivf/uq_native_runner bench-batch-artifact \
  "$TRACE/events.bin" "$ARTIFACT_ROOT/k256/ivf_opq" "$QUERIES" 128 5
```

三个 artifact 都应通过 batch parity。`quality_sweep.csv` 使用全部 decision_count_s 为全局 FP/TP 分母；conditional FP 分母另为 FP+TN。`selected.csv` 给 balanced 与 strict 下正确剪枝最多的 beta，没有通过点就没有该方法/门控的选中行，不能将其补成零风险。静态 trace FP 不是主动搜索全轨迹 FP，也不是 Recall@10。

GIST 官方只有1000条 query。本轮 pilot 配置为 query_start=0,count=1000,exploratory=true，明确允许与开发集重叠；严格留出实验请分开 development 与 evaluation（例如前600/后400），设置 exploratory=false，脚本检测到重叠便拒绝。旧横向实验已使用全部1000条，新的留出仍需披露历史调参使用，不能称完全未见过的最终测试集。

## 5. 端到端 pilot

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/edge_estimation/ivf_experiments.py grid \
  --config configs/edge_estimation/ivf_pilot.json --artifacts "$ARTIFACT_ROOT/k256" \
  --index "$INDEX" --queries "$QUERIES" --ground-truth "$GROUND_TRUTH" \
  --runner build-ivf/uq_ivf_performance_runner --out "$PILOT"
```

一个 Kc 共57点（baseline5、PQ16、OPQ16、PQ+QJL20）。先看K256，再使用同样命令单独跑K1024；选择同一个Kc作为三方法主比较，不让各方法暗中用不同coarse。每个 repeat 内随机排序，各点单独进程；加载/哈希不计QPS，但总作业时间包含这些成本。每完成一点保存 points.csv；缺 complete 不算完整实验。

默认配置只比较新方法与同二进制 HNSW。要与原 PQ8、OPQ、PQ+QJL 同节点随机交错重测，使用 `ivf_pilot_with_controls.json`（72点），同时提供：

```bash
  --v0-runner /path/to/v0_performance_runner \
  --opq-runner /path/to/uq_opq_performance_runner \
  --pq-sidecar /path/to/original.pq \
  --qjl-companion /path/to/original.v0res \
  --opq-artifact /path/to/original-opq-artifact
```

这些控制组明确读原始方向量化 artifact，新方法明确读 coarse residual artifact。旧二进制必须与新构建使用同CPU、同编译优化/线程配置；不能直接使用旧CSV声称配对性能收益。控制组接口复用已经验证的 run_recall_qps_grid.command_for；本次没有真实 legacy/GIST 资产运行这一附加矩阵。控制组不支持的诊断指标留空，不伪装成0。

## 6. 高 recall、batch、最终重复与图表

```bash
python scripts/edge_estimation/ivf_experiments.py refine \
  --config configs/edge_estimation/ivf_pilot.json --points "$PILOT" --out "$FOLLOWUPS"
```

从95%/98%阈值点（达不到时选当前最大recall）选 beta，输出：

- `high_recall.json`：ef={950,1100,1300}；
- `batch.json`：batch={1,32,128,256}；
- `confirm.json`：选中邻域 repeat=3。

把相应配置交给同一个 grid 命令，使用新输出目录。高 recall 补测后，可对新目录再 refine，而不是在 pilot 尚未达到目标时就固定最终参数。confirm 仍是相同query split，重复只测运行波动，不生成新的独立统计样本。

```bash
python scripts/edge_estimation/ivf_experiments.py summarize --out "$PILOT" --plot
```

输出 points.csv、aggregated.csv（配置内QPS均值/中位数/标准差/min/max）、pareto.csv、thresholds.csv；可选 PNG/PDF 图。重复先聚合再选择前沿，避免挑最快单次重复。未达到98.5%明确写 NOT_REACHED。

`latency_p*` 是 search+本批prep摊销，`batch_completion_service_p95_ns` 是本批prep+累计搜索服务时间，二者都不含到达/排队等待；batch=1用于单请求服务时间。model/scratch/catalog字节和peak RSS分开；scratch为容器payload/近似map开销。`eligible_exact_distance_count`只覆盖hook触达的layer-0 candidates，未含入口/upper layers，baseline为null（未采样），不能称全索引精确距离总数。FP诊断由单独冻结trace生成，不放进QPS热循环。

## 7. Slurm与当前状态

```bash
export UQ_REPO=/path/to/checkout UQ_PYTHON=/path/to/python
sbatch --cpus-per-task=1 --mem=32G --time=12:00:00 \
  scripts/edge_estimation/submit_ivf_slurm.sh train \
  --config configs/edge_estimation/ivf_pilot.json --assets assets.json \
  --catalog "$CATALOG" --out "$ARTIFACT_ROOT"
```

quality/grid/refine同样传给worker。内存/时间为初始申请值，不是本轮实测GIST峰值。不要在登录节点训练/全图编码。本次未自动连接集群或提交作业。

当前本地已有GIST base/query/mapping；已有资产审计记录中的正式索引和ground truth为集群路径，未据此宣称当前可访问。完整GIST训练和正式曲线尚未执行。已有小规模集成测试、960维独立公式oracle和原生/脚本回归；详见 `docs/edge_estimation/IVF_IMPLEMENTATION_STATUS.md`。
