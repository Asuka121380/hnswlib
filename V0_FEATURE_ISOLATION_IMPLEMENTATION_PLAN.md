# V0 分支内功能隔离实现计划

## 1. 文档目的

本文定义 `v0-edge-quantisation` 分支中 V0 单层保守边量化剪枝的功能隔离方案。

隔离目标是同时保留并可独立构建以下能力：

1. 原始 HNSW baseline；
2. 已完成的 baseline trace；
3. V0 正式剪枝；
4. V0 shadow correctness validation。

V0 的代码加入后，必须继续满足：

- baseline build 不包含 V0 查询分支、V0 数据成员或 V0 统计开销；
- trace build 保持现有 baseline trace 语义；
- V0 performance build 不包含详细 trace 和 shadow exact-distance 开销；
- V0 shadow build 可以验证下界和剪枝安全性；
- 同一个 V0-enabled binary 中可以分别运行 baseline 查询与 V0 查询；
- 所有模式使用同一份基础 HNSW 搜索逻辑和更新逻辑，避免复制整套搜索实现。

---

## 2. 当前代码已有的隔离基础

当前 `baseline-trace` 已经实现了可复用的功能隔离框架。

### 2.1 CMake 编译选项

`CMakeLists.txt` 已定义：

```cmake
option(
    HNSWLIB_ENABLE_BASELINE_TRACE
    "Enable baseline HNSW search tracing."
    OFF
)
```

开关启用后，通过 header-only `INTERFACE` target 传播宏：

```cmake
target_compile_definitions(
    hnswlib
    INTERFACE HNSWLIB_ENABLE_BASELINE_TRACE=1
)
```

### 2.2 预处理器边界

`hnswlib/hnswalg.h` 中的 trace header、collector、计时和记录逻辑均位于：

```cpp
#ifdef HNSWLIB_ENABLE_BASELINE_TRACE
...
#endif
```

因此 trace-off build 会在编译前移除相关代码。

### 2.3 独立查询入口

现有接口分为：

```cpp
searchKnn(...)
searchKnnWithTrace(...)
```

普通查询入口保持可用，trace 查询入口只在 trace 编译开关启用时出现。

### 2.4 Runner 模式

`examples/cpp/real_data_trace_runner.cpp` 已支持：

- `build-index`；
- `performance`；
- `correctness`；
- `trace`。

其中 correctness 模式可同时执行 baseline 和 traced 查询并检查结果一致性。

### 2.5 独立构建目录

现有实验管线已经分别构建：

```text
hnsw-trace-on
hnsw-trace-off
```

正式 baseline 性能来自 trace-off binary。V0 将沿用这种构建目录隔离方式。

---

## 3. 总体隔离策略

V0 采用四层隔离：

1. **编译功能隔离**：CMake option 和预处理宏；
2. **API 行为隔离**：baseline、trace、V0 使用不同公共查询入口；
3. **热路径隔离**：使用编译期模板参数生成 baseline 和 V0 两套实例；
4. **实验环境隔离**：使用独立 build directory、runner mode 和输出目录。

逻辑结构如下：

```text
                        shared original HNSW logic
                                  |
              +-------------------+-------------------+
              |                   |                   |
        searchKnn()       searchKnnWithTrace()   searchKnnV0()
              |                   |                   |
        baseline path        baseline + trace       V0 gate
                                                      |
                                      +---------------+---------------+
                                      |                               |
                                 safe prune                    exact fallback
                                                                      |
                                                         original HNSW update
```

V0 不修改 baseline API 的行为，也不让 baseline trace 自动记录 V0 语义。

---

## 4. 新增 CMake 选项

在现有 `HNSWLIB_ENABLE_BASELINE_TRACE` 之后新增：

```cmake
option(
    HNSWLIB_ENABLE_EDGE_QUANT_V0
    "Enable V0 conservative edge-quantised pruning support."
    OFF
)

option(
    HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
    "Enable exact shadow validation for V0 pruning decisions."
    OFF
)
```

添加选项依赖：

```cmake
if(HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
   AND NOT HNSWLIB_ENABLE_EDGE_QUANT_V0)
    message(FATAL_ERROR
        "HNSWLIB_ENABLE_V0_SHADOW_VALIDATION requires "
        "HNSWLIB_ENABLE_EDGE_QUANT_V0=ON")
endif()
```

传播编译宏：

```cmake
if(HNSWLIB_ENABLE_EDGE_QUANT_V0)
    target_compile_definitions(
        hnswlib
        INTERFACE HNSWLIB_ENABLE_EDGE_QUANT_V0=1
    )
endif()

if(HNSWLIB_ENABLE_V0_SHADOW_VALIDATION)
    target_compile_definitions(
        hnswlib
        INTERFACE HNSWLIB_ENABLE_V0_SHADOW_VALIDATION=1
    )
endif()
```

三个选项保持独立：

```text
HNSWLIB_ENABLE_BASELINE_TRACE
HNSWLIB_ENABLE_EDGE_QUANT_V0
HNSWLIB_ENABLE_V0_SHADOW_VALIDATION
```

---

## 5. 构建矩阵

必须维护四个独立构建目录：

| 构建目录 | Baseline trace | V0 | Shadow | 用途 |
|---|---:|---:|---:|---|
| `build-baseline` | OFF | OFF | OFF | 原始 HNSW 性能基线 |
| `build-trace` | ON | OFF | OFF | 复现 baseline trace |
| `build-v0` | OFF | ON | OFF | V0 正式性能实验 |
| `build-v0-shadow` | ON | ON | ON | V0 正确性与诊断 |

### 5.1 Baseline build

```bash
cmake -S . -B build-baseline -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_BASELINE_TRACE=OFF \
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=OFF \
  -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=OFF
```

### 5.2 Trace build

```bash
cmake -S . -B build-trace -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_BASELINE_TRACE=ON \
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=OFF \
  -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=OFF
```

### 5.3 V0 performance build

```bash
cmake -S . -B build-v0 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_BASELINE_TRACE=OFF \
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON \
  -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=OFF
```

### 5.4 V0 shadow build

```bash
cmake -S . -B build-v0-shadow -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHNSWLIB_ENABLE_BASELINE_TRACE=ON \
  -DHNSWLIB_ENABLE_EDGE_QUANT_V0=ON \
  -DHNSWLIB_ENABLE_V0_SHADOW_VALIDATION=ON
```

禁止在一个 build directory 中反复切换功能选项后直接复用旧产物。每类实验必须使用固定目录和完整构建元数据。

---

## 6. 文件和组件规划

建议新增：

```text
hnswlib/
├── edge_quant_v0.h
├── edge_quant_v0_metadata.h
├── edge_quant_v0_metrics.h
└── edge_quant_v0_io.h

examples/cpp/
├── v0_edge_encoder.cpp
└── v0_search_runner.cpp

tests/cpp/
├── v0_feature_isolation_test.cpp
├── v0_metadata_test.cpp
├── v0_bound_test.cpp
└── v0_search_equivalence_test.cpp

scripts/v0/
├── build_binaries.ps1
├── run_correctness.ps1
└── run_performance.ps1
```

职责划分：

| 文件/组件 | 职责 |
|---|---|
| `edge_quant_v0.h` | 查询期 LUT、bound evaluator 和 V0 query context |
| `edge_quant_v0_metadata.h` | codebook、edge record、offset table 和格式定义 |
| `edge_quant_v0_metrics.h` | V0 聚合指标和 shadow validation record |
| `edge_quant_v0_io.h` | sidecar metadata 的装载、保存与校验 |
| `v0_edge_encoder.cpp` | 离线 codebook 训练和全边编码入口 |
| `v0_search_runner.cpp` | baseline/V0/shadow 对照和性能实验入口 |

V0 第一版使用 sidecar metadata，不修改原始 HNSW 序列化格式。

---

## 7. Header 和类成员隔离

在 `hnswlib/hnswalg.h` 顶部：

```cpp
#ifdef HNSWLIB_ENABLE_EDGE_QUANT_V0
#include "edge_quant_v0.h"
#include "edge_quant_v0_metadata.h"
#include "edge_quant_v0_metrics.h"
#endif
```

V0 索引成员只在 V0 build 中存在：

```cpp
#ifdef HNSWLIB_ENABLE_EDGE_QUANT_V0
std::shared_ptr<const EdgeQuantV0Metadata> edge_quant_v0_metadata_;
#endif
```

V0 公共接口也只在 V0 build 中出现：

```cpp
#ifdef HNSWLIB_ENABLE_EDGE_QUANT_V0
void loadEdgeQuantV0Metadata(const std::string& path);

std::priority_queue<std::pair<dist_t, labeltype>>
searchKnnV0(
    const void* query_data,
    size_t k,
    V0QueryMetrics* metrics = nullptr,
    BaseFilterFunctor* isIdAllowed = nullptr) const;
#endif
```

V0 关闭时：

- 不包含 V0 header；
- `HierarchicalNSW` 不增加 V0 数据成员；
- 不暴露 `searchKnnV0()`；
- 不装载 codebook 或 edge metadata；
- baseline 热路径不出现 V0 runtime flag。

---

## 8. 公共查询 API 隔离

最终保留以下接口：

```cpp
searchKnn(...)
searchKnnWithTrace(...)
searchKnnV0(...)
```

接口语义：

| API | 搜索行为 |
|---|---|
| `searchKnn()` | 始终执行原始 HNSW |
| `searchKnnWithTrace()` | 原始 HNSW 加 baseline trace |
| `searchKnnV0()` | 执行 V0 下界检查和安全剪枝 |

在 shadow build 中，`searchKnnV0()` 的 V0 判断保持不变，但对拟剪枝候选补算精确距离并记录验证结果。

`searchKnn()` 不应因 metadata 已装载或 V0 编译开启而自动改变行为。这样 V0-enabled binary 可以在同一进程中运行：

```cpp
baseline_result = index.searchKnn(query, k);
v0_result = index.searchKnnV0(query, k, &metrics);
```

然后直接进行结果一致性比较。

---

## 9. 内部热路径隔离

当前底层函数模板为：

```cpp
template <bool bare_bone_search = true, bool collect_metrics = false>
searchBaseLayerST(...)
```

计划增加编译期 V0 参数：

```cpp
template <
    bool bare_bone_search = true,
    bool collect_metrics = false,
    bool use_edge_quant_v0 = false
>
searchBaseLayerST(...)
```

baseline 实例：

```cpp
searchBaseLayerST<true, false, false>(...)
```

V0 实例：

```cpp
searchBaseLayerST<true, false, true>(...)
```

邻居扫描中：

```cpp
if (use_edge_quant_v0) {
    // V0 eligibility、LUT lookup、bound evaluation 和 safe prune
}
```

`use_edge_quant_v0` 是编译期常量。编译器应在 baseline 实例中完全删除 V0 分支。

V0 只能插入在：

```text
未访问邻居已按原始时机标记 visited
        ↓
原始 exact distance 调用之前
```

无法安全剪枝时，必须继续进入现有：

```cpp
fstdistfunc_(data_point, currObj1, dist_func_param_)
```

并复用原始 candidate queue、result heap 和 threshold 更新代码。

禁止为 V0 复制另一份完整的 `searchBaseLayerST()`。

---

## 10. Trace 与 V0 metrics 的隔离

现有 `BaselineTraceCollector` 假定每个 unique neighbour 都执行精确距离计算，并记录 `dist_qd`。

V0 剪枝后，该假设不再成立。因此：

- 不修改 baseline trace 的指标定义；
- 不把 V0 剪枝事件直接解释为 baseline DCO；
- 不让 baseline trace 的 `n_dist` 混入 bound evaluation 数；
- 为 V0 新建独立的 metrics 和 validation record。

建议结构：

```cpp
struct V0QueryMetrics {
    uint64_t bound_evaluated;
    uint64_t bound_pruned;
    uint64_t exact_fallback;
    uint64_t exact_only_fallback;
    uint64_t exact_distance_saved;
    uint64_t lower_bound_violation;
    uint64_t false_prune;
};
```

Shadow record 至少包含：

```text
query_id
current_node_id
candidate_id
s_c
tau
s_hat_d
delta_d
lower_bound
shadow_exact_s_d
would_prune
lower_bound_valid
false_prune
```

V0 performance build 只保留低成本 query-level 聚合计数器，不保存逐 DCO record。

---

## 11. Runner 隔离

新增 `v0_search_runner`，支持：

```text
--mode correctness
--mode shadow
--mode performance
```

### 11.1 Correctness 模式

对每个查询：

```cpp
baseline = index.searchKnn(query, k);
v0 = index.searchKnnV0(query, k, &metrics);
```

检查：

- top-k label 完全一致；
- top-k 精确距离一致；
- Recall@k 一致；
- V0 metadata 与基础索引匹配；
- 所有 V0 invariants 成立。

### 11.2 Shadow 模式

V0 对满足 `L_d > tau` 的候选仍计算：

```text
s_d = ||q - d||^2
```

但该 shadow distance 不得改变搜索状态。

检查：

```text
L_d <= s_d
L_d > tau  =>  s_d > tau
```

要求：

```text
lower_bound_violation == 0
false_prune == 0
```

### 11.3 Performance 模式

只允许使用：

```text
HNSWLIB_ENABLE_BASELINE_TRACE=OFF
HNSWLIB_ENABLE_EDGE_QUANT_V0=ON
HNSWLIB_ENABLE_V0_SHADOW_VALIDATION=OFF
```

关闭：

- 逐 DCO CSV；
- shadow exact distance；
- 详细计时；
- debug assertions；
- 大型 validation record。

性能模式分别运行 `searchKnn()` 和 `searchKnnV0()`，输出到不同目录，禁止覆盖。

---

## 12. CMake target 隔离

V0 target 只在 V0 开启时构建：

```cmake
if(HNSWLIB_ENABLE_EDGE_QUANT_V0)
    add_executable(
        v0_edge_encoder
        examples/cpp/v0_edge_encoder.cpp
    )
    target_link_libraries(v0_edge_encoder hnswlib)

    add_executable(
        v0_search_runner
        examples/cpp/v0_search_runner.cpp
    )
    target_link_libraries(v0_search_runner hnswlib)

    add_executable(
        v0_feature_isolation_test
        tests/cpp/v0_feature_isolation_test.cpp
    )
    target_link_libraries(v0_feature_isolation_test hnswlib)
    add_test(
        NAME v0_feature_isolation_test
        COMMAND v0_feature_isolation_test
    )
endif()
```

Shadow-only test 只在 shadow 开启时加入：

```cmake
if(HNSWLIB_ENABLE_V0_SHADOW_VALIDATION)
    add_executable(
        v0_shadow_validation_test
        tests/cpp/v0_shadow_validation_test.cpp
    )
    target_link_libraries(v0_shadow_validation_test hnswlib)
    add_test(
        NAME v0_shadow_validation_test
        COMMAND v0_shadow_validation_test
    )
endif()
```

现有 baseline trace targets 保持原条件，不改为依赖 V0。

---

## 13. 实验输出隔离

建议目录：

```text
results/
├── baseline/
├── baseline-trace/
├── v0-correctness/
├── v0-shadow/
└── v0-performance/
```

每次运行的 metadata 必须保存：

- Git commit；
- Git branch；
- working tree 状态；
- 编译器和版本；
- build type；
- 三个 CMake option；
- 数据集配置；
- 基础 HNSW index fingerprint；
- V0 sidecar fingerprint；
- codebook 配置；
- `efSearch`；
- query range；
- 是否启用 shadow。

任何跨模式比较都必须确认基础 HNSW index fingerprint 相同。

---

## 14. 实施阶段

## 14.1 阶段 1：隔离骨架

实现：

- 两个新 CMake option；
- option dependency 检查；
- 空的 V0 headers；
- 条件编译 include；
- `searchKnnV0()` 公共入口；
- 内部 `use_edge_quant_v0` 模板参数；
- V0 入口暂时执行与 baseline 完全相同的逻辑。

暂不实现：

- metadata；
- codebook；
- LUT；
- bound；
- pruning。

验收：

- 四种 build 均按预期配置；
- baseline build 不暴露 V0 API；
- trace 测试继续通过；
- V0-enabled build 中 `searchKnn()` 和空实现 `searchKnnV0()` 输出完全一致；
- shadow ON、V0 OFF 时 CMake 配置失败。

## 14.2 阶段 2：Metadata 隔离

实现：

- sidecar 数据结构；
- metadata load/save；
- index fingerprint；
- edge count 和 offset 校验；
- V0 索引成员。

V0 查询仍不剪枝。

验收：

- baseline build 大小和行为不受影响；
- metadata 未装载时 `searchKnnV0()` 明确报错；
- metadata 与错误索引配对时拒绝运行；
- serialize/deserialize round trip 一致。

## 14.3 阶段 3：Shadow-only bound evaluation

实现：

- query LUT；
- per-edge lookup；
- `s_hat_d`；
- `Delta_d`；
- `L_d`；
- V0 metrics；
- shadow exact-distance 验证。

此阶段即使 `L_d > tau` 也不真正跳过 exact distance。

验收：

- `lower_bound_violation == 0`；
- `false_prune == 0`；
- baseline 和 V0-shadow 搜索结果完全一致；
- baseline trace invariants 未被修改。

## 14.4 阶段 4：启用真实剪枝

实现：

```cpp
if (lower_bound > tau) {
    ++metrics.bound_pruned;
    ++metrics.exact_distance_saved;
    continue;
}
```

验收：

- shadow 验证仍为零错误；
- V0 和 baseline top-k 完全一致；
- Recall 一致；
- exact-distance count 按预期下降；
- `searchKnn()` 仍执行原始路径。

## 14.5 阶段 5：性能隔离与优化

实现：

- trace-free V0 performance build；
- thread-local metrics；
- LUT 和 metadata 热路径优化；
-稳定的运行脚本。

验收：

- baseline 与 V0 使用相同编译器和优化参数；
- baseline binary 不包含 V0 功能；
- V0 performance binary 不包含 trace/shadow；
- 报告 QPS、延迟、exact distance saved 和额外内存；
- 给出净加速或净减速结论。

---

## 15. 测试计划

### 15.1 配置测试

- 所有 option 默认 OFF；
- shadow ON 且 V0 OFF 配置失败；
- baseline build 不构建 V0 targets；
- trace OFF 时不构建 trace-only targets；
- V0 OFF 时不暴露 V0 API。

### 15.2 API 测试

- `searchKnn()` 在所有 build 中保持 baseline 语义；
- `searchKnnWithTrace()` 只在 trace build 中存在；
- `searchKnnV0()` 只在 V0 build 中存在；
- V0 metadata 未装载时行为明确；
- V0-enabled binary 可在同一索引上连续运行 baseline 和 V0。

### 15.3 搜索等价性测试

- 空 V0 骨架与 baseline 完全一致；
- shadow-only 与 baseline 完全一致；
- safe-pruning V0 与 baseline 完全一致；
- 多个 `efSearch` 下结果一致；
- bare-bone 和 filtered 路径分别测试；
- zero-length edge 走 exact-only 路径。

### 15.4 热路径测试

- result heap 未满时不执行 V0 bound；
- metadata 无效时不静默剪枝；
- `tau` 来自最新精确 result heap；
- bound-pruned candidate 不调用 exact distance；
- fallback candidate 只调用一次 exact distance；
- shadow exact distance 不改变候选队列或结果堆。

---

## 16. 提交拆分建议

建议按以下 commits 推进：

1. `Add V0 feature-isolation build scaffolding`
2. `Add isolated V0 search API and template path`
3. `Add V0 sidecar metadata structures and validation`
4. `Add V0 query LUT and conservative bound evaluator`
5. `Add V0 shadow validation and metrics`
6. `Enable conservative V0 pruning`
7. `Add V0 correctness and performance runners`
8. `Add V0 build and experiment scripts`

每个 commit 必须保持可构建，并运行与改动范围匹配的测试。

---

## 17. 完成标准

功能隔离完成需要同时满足：

1. baseline、trace、V0、V0-shadow 四种构建可独立生成；
2. baseline build 的预处理结果不包含 V0 数据结构和查询分支；
3. V0-enabled build 中 `searchKnn()` 仍是原始 baseline；
4. baseline trace 的原有字段、计数和 invariants 不因 V0 改变；
5. V0 使用独立 metrics 和 validation record；
6. V0 performance 不包含详细 trace 和 shadow 开销；
7. V0 fallback 复用原始 exact distance 和原始 HNSW 更新代码；
8. 结果目录和运行 metadata 能明确识别所用构建模式；
9. 所有配置和搜索等价性测试通过；
10. 后续 V0 实现可以在此隔离框架内逐步加入，而不污染 baseline。

