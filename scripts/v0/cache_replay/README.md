# 固定 PQ / exact 访问序列的 LUT cache replay

本工具回答：在相同 PQ/exact 访问顺序下，缩小 LUT footprint 可以减少多少时间和 L1D miss？
它不测 recall，也不输出端到端 QPS。使用 C++ 执行真实数据访问，Python 只负责编译、实验编排和汇总。

## 现有工具检查结果

检查了当前工作区和 `C:/ANU/IndividualProject/Implementation/hnswlib`：

- `examples/cpp/v0_fast_kernel_microbenchmark.cpp` 重复热缓存操作，不重放搜索访问序列。
- `scripts/v0/margin_retry/replay_margin_frontier.py` 重放剪枝判定统计，不测原生缓存访问。
- 工作区 `.jq-replay-staging/scripts/v0/jq_pruning_replay` 是理论 bound replay。
- baseline DCO trace 没有实际 active-PQ 的所有剪枝和 retry 次序，不能直接冒充本实验输入。

因此新增独立工具。`build` 把指定仓库的头文件复制到构建目录，在复制件的实际
`evaluateRawFast` lookup 前插入一个记录回调；原仓库头文件不改动。
如果目标代码的 hook 位置改变，构建会报错，避免静默漏录。

## 记录与计时范围

记录每条 query 的边界、每次实际 LUT lookup 的 sidecar edge index / current distance，
以及所有 exact L2 调用的 index internal node id，包括 upper-layer、entry、fallback、retry。
exact 包装器委托给原 HNSWLIB L2Space 选择的距离内核，不更改数值结果。
每个 query 录制后会关闭记录再搜索一次，逐项比较返回的距离和 label。

重放直接读取原 index 中的向量和原 sidecar 中的 edge record，保留它们的存储顺序；
不会把候选向量压缩复制到一个小的热缓存工作集。
PQ/exact 决策完全固定，新的估计值不改变访问顺序。

- 每个 query 重新构造 LUT，构表和文件加载都在计时区间之外。
- LUT 64-byte 对齐；`spacing` 是相邻有效 float 表项间的 float 步长。
- lookup + length/anchor 估计 + exact L2 在计时内，输出 checksum 防止编译器删掉工作。
- 输出是每次完整 trace 的 `elapsed_ns` 和每 query 的派生耗时，**不是完整搜索延迟**。
- 尚未重放 visited、队列、邻接扫描、失败 estimator 的 metadata-only 读取及原软件 prefetch。
  trace 事件读取本身也增加流量。因此它是缓存机制实验，不是完整搜索的逐内存指令仿真。
- 指针/缓存对齐可能与原进程不同；进程级随机化重复用于观察这种波动。

## 三种实验，必须区分

### A. 数值不变的布局对照

同一个 codebook、同一条 trace，改变 `--spacing 1 4 16`。
同一个 bit 下，各布局的 checksum 必须完全一致，否则驱动报错。

例如真实 M32,b6：spacing=1 是 8 KiB，spacing=4 是 32 KiB。
真实 M32,b4：spacing=1 是 2 KiB，spacing=16 是 32 KiB。
分散表项会同时改变空间局部性和 set 映射；这属于 footprint/layout 对照，不是单独的容量因果证明。
当前 M32,b8 的相应布局是 32/128/512 KiB，它本身不能无损变成 8 KiB。

### B. 当前 b8 sidecar 上的折叠敏感性实验

尚无训练好的低 bit sidecar 时，可以显式用 `--fold-bits 8 6 4`。
对于 b6/b4，把原 code 通过 `code & (K-1)` 映射到更小索引范围，LUT 使用原 codebook 的前 K 个 centroid。
该模式模拟较小 footprint 和更密集的 line 重用；**它不是重新训练的 PQ**，估计值不再有原来的质量含义。
输出 `folded_surrogate=true`，不能把结果写成真实 b6/b4 性能结果，也不能使用它推导 recall。
同一 folded bit 下，紧凑/扩展布局仍然数值一致，可以检验缓存机制。
固定原 sidecar record stride，不把 code 压缩收益混入 LUT 实验。

### C. 单独训练的真实低 bit / 小 M sidecar

指定 `--replay-sidecar /path/to/trained-b6.sidecar --fold-bits 6`。
新 sidecar 必须属于完全相同 index 和 edge ordering；原库加载时检查 index/adjacency 绑定。
新 sidecar 中 M、K、code、length、anchor 由其自身读取，不对旧 b8 code 做截断。
当指定 bit 等于新 sidecar 的实际 bit 时，`folded_surrogate=false`。
这仍然是原轨迹上的成本对照，不代表新配置实际搜索时的访问分布。
真正的等 recall QPS 必须另外运行真实搜索。

现有 `train_pq_codebook.py` 的 pilot contract 限定 nbits=8；本工具不绕过或修改训练合同，
也不声称已生成真实低 bit sidecar。训练/编码低 bit 是后续独立任务。

## 编译与本地验证

需要 Python 3.9+ 和 GCC/Clang C++17。Linux 建议在目标测试节点编译；使用 `-march=native`。
Windows 使用 MSYS2 UCRT64 g++，确保其 bin 在 PATH 中以加载运行库。
下面命令从当前工作区运行。Linux 改为实际仓库和编译器路径即可。

```powershell
python scripts/v0/cache_replay/run_replay.py build --repo C:/ANU/IndividualProject/Implementation/hnswlib --build tmp/cache-replay-build --cxx C:/msys64/ucrt64/bin/g++.exe
python scripts/v0/cache_replay/run_replay.py smoke --build tmp/cache-replay-build --out tmp/cache-replay-smoke-new
```

`smoke` 创建独立合成 index/sidecar/queries，测试录制开关结果一致、retry/legacy 和 no-retry/gate、
3 bits × 3 layouts × 2 process blocks、重复 checksum 一致，以及截断 trace 拒绝。
合成 fixture 耗时没有 GIST1M 性能意义。
输出目录必须不存在，避免覆盖已有实验。

## GIST1M 录制

把下列路径替换为同一实验的实际 index、sidecar、queries。先录 100 个 query 验证磁盘和内存，再扩大到 1000。
trace 为 32-byte header + 每事件 24 bytes；8 百万事件约 192 MB，另需加载 index、sidecar 和记录映射。

```powershell
python scripts/v0/cache_replay/run_replay.py record --build tmp/cache-replay-build --index /data/gist/index.bin --sidecar /data/gist/m32b8.sidecar --queries /data/gist/gist_query.fvecs --dimension 960 --query-start 0 --query-count 100 --ef 500 --k 10 --beta 1.4 --retry --prefetch legacy --out outputs/cache-replay/capture-ef500-b140
```

用 `--no-retry --prefetch gate` 等参数录制其他 operating point，每个使用不同输出目录。
保留 `manifest.json`、`trace.bin`、`record.json`；manifest 记录输入 SHA256、配置、源码和二进制来源。
Python 驱动每次重放先校验原输入和 trace 的 SHA256。大文件校验耗时不算实验时间。
二进制 trace v1 使用本机 little-endian uint64/double；支持的 Windows/Linux x86-64 ABI 相同。

## 正式计时

先针对当前 b8 trace 做缓存敏感性实验：

```powershell
python scripts/v0/cache_replay/run_replay.py replay --build tmp/cache-replay-build --trace-dir outputs/cache-replay/capture-ef500-b140 --fold-bits 8 6 4 --spacing 1 4 16 --blocks 5 --repeats 5 --warmups 2 --out outputs/cache-replay/timing-ef500-b140
```

Linux 正式计时加 `--cpu N`，N 为已分配的逻辑 CPU；节点空闲情况、SMT 邻核占用和频率策略需保持一致。
默认每个配置新进程，随机化运行次序，先 warmup 再测 repeats。汇总使用进程内中位数和进程间中位数，
保留所有原始数据；没有把进程内 repeats 伪装成独立样本或生成无依据的 CI。

有真实 b6 sidecar 后：

```powershell
python scripts/v0/cache_replay/run_replay.py replay --build tmp/cache-replay-build --trace-dir outputs/cache-replay/capture-ef500-b140 --replay-sidecar /data/gist/m32b6.sidecar --fold-bits 6 --spacing 1 4 --blocks 5 --out outputs/cache-replay/real-b6
```

## Linux 硬件计数器诊断

在单独输出目录重复上述 replay 命令，加 `--counters --cpu N`。
通过 `perf_event_open` 读取当前线程的 cycles、instructions、L1D read misses；每次 query 构表后开启、
重放结束后关闭，排除加载和构表。计数器开启/关闭附近仍有少量用户态指令，属于诊断开销。
诊断 pass 的计时不可与无 counters 的正式计时混用。无权限、事件不支持或 multiplexing 都会明确报错，
不会填造为 0。Windows 未实现 counters；空值 `null` 表示未测。
L2 miss 需要目标 CPU 的原生 PMU event，目前未实现；不把 LLC miss 冒称 L2 miss。
本地已验证 Windows 路径，Linux PMU 路径需在集群验证权限和事件支持。

## 结果解释

- `summary.json`：各配置 median ns/query、每个进程 block 的原始值、输入身份、实验顺序。
- 单配置 JSON：PQ/exact 事件数、LUT 大小、source/lookup bits、surrogate 标志、每次 elapsed/checksum、可选计数器。
- 同一 trace 的事件数应一致；同一 bit 的不同 spacing 数值必须一致。
- 重点比较 b6 spacing1 vs spacing4、b4 spacing1 vs spacing16，结合 L1D misses 和 cycles 同时变化。
- b8 vs folded b6/b4 仅回答硬件敏感性，不回答算法质量。
- 把 ns/query 的差值视为固定轨迹下的成本差，再去完整搜索验证；不能直接乘上现有 QPS 宣称最终加速。
