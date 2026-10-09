# 高维数据接入：已实现接口与执行说明

本实现接入 BioASQ1024、DBpedia te3-large1536，以及独立的可选3072 profile。
原 `final_study` 方法、数学公式、m32/8bit/100K训练边预算及 v1 文件保持原样。
新增数据准备源码会改变 build fingerprint，集群需要新建 v2 tools/performance 构建。

## 1. 新接口

| 模块 | 子命令 | 作用 |
|---|---|---|
| `scripts.datasets.prepare_highdim` | `probe` | 本地或远程Parquet footer检查；远程只接受HTTP206，最多4MiB footer，拒绝意外全量下载 |
| 同上 | `download` | 按固定来源/完整分片清单下载，原子发布，计算完整SHA256，生成sealed source-lock |
| 同上 | `lock` | 为已经下载的完整原始文件登记source-lock，无需重复下载 |
| 同上 | `convert` | 单线程分批读取、规范化、去重/留出、来源映射、清洗日志、sealed manifest |
| `scripts.datasets.highdim_controls` | `data` | 从已验证转换结果生成registry/splits/history/review ledger，使用实际N与候选容量 |
| 同上 | `protocol` | 从真实v1协议派生pilot或最终四集v2；最终版必须提供显式ef网格 |
| 同上 | `environment` | 生成先加载v1环境、再设置v2路径和旧资产引用函数的Bash包装 |

PyArrow是数据准备依赖，`requirements-highdim.txt`供独立环境使用，不安装到冻结的QPS环境。
下载使用Python标准库，不需要HF token或embedding API。来源定义见
`configs/edge_estimation/final_study/highdim-sources.json`：DBpedia固定完整revision、26/63个分片；
BioASQ使用`shuffle_train.parquet`、`test.parquet`、`neighbors.parquet`。

## 2. 下载和转换

在仓库根执行，下列变量均由操作者设置为目标机器绝对路径。
`PREP_PYTHON`指独立准备环境，`PYTHON`指既有测量环境。

```bash
"$PREP_PYTHON" -m scripts.datasets.prepare_highdim probe \
  --profile bioasq1024 --out "$CONTROL/bioasq-source-probe.json"

"$PREP_PYTHON" -m scripts.datasets.prepare_highdim download \
  --profile bioasq1024 --out "$DATA_ROOT/bioasq/raw"
"$PREP_PYTHON" -m scripts.datasets.prepare_highdim download \
  --profile dbpedia1536 --out "$DATA_ROOT/dbpedia1536/raw"

# 若原始文件已经完整下载，可用lock替代download；两者不重复执行。
# --raw-dir须符合上述来源清单的目录结构。
# "$PREP_PYTHON" -m scripts.datasets.prepare_highdim lock \
#   --profile bioasq1024 --raw-dir "$DATA_ROOT/bioasq/raw" \
#   --out "$CONTROL/bioasq.source-lock.json"

"$PREP_PYTHON" -m scripts.datasets.prepare_highdim convert \
  --profile bioasq1024 --source-lock "$DATA_ROOT/bioasq/raw/source-lock.json" \
  --normalize l2-f64-to-f32 --batch-size 4096 --out "$DATA_ROOT/bioasq/converted"
"$PREP_PYTHON" -m scripts.datasets.prepare_highdim convert \
  --profile dbpedia1536 --source-lock "$DATA_ROOT/dbpedia1536/raw/source-lock.json" \
  --normalize l2-f64-to-f32 --batch-size 4096 --holdout-count 10000 \
  --holdout-seed 20261010 --out "$DATA_ROOT/dbpedia1536/converted"
```

所有输出目录/协议都拒绝覆盖，失败保留`.partial-*`目录，且不发布成功manifest。
下载当前采用整次原子发布，无自动断点续传；失败后检查partial，换新目录重试，或核验完整文件后用`lock`登记。
远程probe对所有登记文件读取footer，不读取向量页；它不替代正式下载hash或集群连通性验证。

转换规则：float64求L2范数后cast小端float32并规范±0。有限极端值导致范数溢出/下溢时，
使用缩放后的float64范数，manifest明确记录该fallback规则。null、非有限和零向量记录并剔除；
维度错误、缺列、非浮点向量、无效/冲突ID立即失败。base/query全部使用相同变换。

BioASQ保留有效base顺序和重复base记录；query按向量去重并排除与base的向量重合。
base和query相同数值ID不自动判定为泄漏。DBpedia按源分片/row稳定顺序，
通过SQLite哈希索引和紧凑union-find将相同ID/文本/向量的传递关系合并，保留最早代表。
文本组按`text`正文完全相同判定，即使title不同也合并；同一ID的title/正文/向量冲突则失败。
不同有效记录共用冲突ID时拒绝输入。对最终代表集合用PCG64留出query，base中物理排除这些实体。
向量按batch读取，DBpedia中间数据落盘；不把整张表变成Python列表。

产物：`base.fvecs`、`query_candidates.fvecs`、两份row到原始ID的JSONL、`exclusions.jsonl`和
`conversion_manifest.json`。manifest保存实际规模、输入/输出SHA、dtype/schema（源锁中）、版本、
normalization、holdout、脚本身份、环境及wall time；输出向量/映射在不同batch下完全一致。
原始GT只保存来源证据，后续必须针对本项目输出重新计算。

## 3. 生成真实数据控制文件与审计门槛

```bash
"$PREP_PYTHON" -m scripts.datasets.highdim_controls data \
  --conversions "$DATA_ROOT/bioasq/converted/conversion_manifest.json" \
                "$DATA_ROOT/dbpedia1536/converted/conversion_manifest.json" \
  --out "$STUDY/control/data-pending"
```

这会生成actual-N registry、splits和**uncertain** review ledger；不能直接通过materialize。
BioASQ合格候选≥3000时500/500/2000、warmup100；2500–2999时test=N−1000；少于2500停止。
DBpedia正式控制文件要求恰好10000 heldout，2000/2000/6000、warmup100。

在`review-templates/`复制对应JSON到单独的审核文件，完成真实历史核查后填写：

```json
{
  "dataset_id": "bioasq1024_cos_v1",
  "source_sha256": "候选fvecs的完整SHA256",
  "range": [0, 3102],
  "status": "unused_confirmed",
  "review_complete": true,
  "reviewer": "实际审核者",
  "basis": "实际审查的历史目录、时间范围、查询用途、转换映射证据与限制"
}
```

range上界必须使用实际候选数，示例3102不是容量承诺。工具要求hash/range完全匹配；
自动生成模板不会替人作“从未使用”的声明。准备完成后生成新的控制目录：

```bash
"$PREP_PYTHON" -m scripts.datasets.highdim_controls data \
  --conversions "$DATA_ROOT/bioasq/converted/conversion_manifest.json" \
                "$DATA_ROOT/dbpedia1536/converted/conversion_manifest.json" \
  --review-evidence "bioasq1024_cos_v1=$CONTROL/bioasq.review.json" \
  --review-evidence "dbpedia_te3l1536_holdout_cos_v1=$CONTROL/dbpedia.review.json" \
  --history-root "$HISTORY_ROOT" --out "$STUDY/control/data-reviewed"
```

然后使用既有`final_study_data inventory/audit-queries/materialize/exact-gt/validate`，输入改为
`data-reviewed/registry.highdim.json`、`history.highdim.json`、`splits.highdim.json`。
materialize现在还会核验conversion和实际base/query绑定，将preprocessing身份写入dataset合同；
validate及run在存在该字段时验证其hash。没有新字段的旧合同保持兼容。

## 4. pilot、最终v2与旧资产

```bash
"$PYTHON" -m scripts.datasets.highdim_controls protocol \
  --base-protocol "$V1/control/protocol.json" --pilot \
  --out "$STUDY/control/protocol-pilot.json"

# 在新集dev pilot完成后填写显式网格JSON，不读test性能。
"$PYTHON" -m scripts.datasets.highdim_controls protocol \
  --base-protocol "$V1/control/protocol.json" --grids "$CONTROL/highdim-ef-grids.json" \
  --out "$STUDY/control/protocol.json"

"$PYTHON" -m scripts.datasets.highdim_controls environment \
  --v1 "$V1" --study "$STUDY" --repo "$REPO" --out "$STUDY/control/env-v2.sh"
```

`highdim-ef-grids.json`恰好含`bioasq1024_cos_v1`和`dbpedia_te3l1536_holdout_cos_v1`两个key，
值为递增、去重、≥k的ef整数列表。pilot默认50/100/200/400/600/900/1300/1800/2400；
这不是已验证覆盖98%的最终网格。最终协议只登记四集，完整复制原training/search/measurement预算。
3072 profile可单独转换，但不自动加入主推荐；其与1536的实体配对仍需专门派生版本，
不能将两次独立holdout默认当作同一实体划分。

生成的env先source旧env，再设置V2变量。`use_dataset gist1m/sift1m`保留V1 data/assets/ledger路径，
其他两集使用V2输入，输出`D`始终指V2。包装不创建研究目录、构建或提交作业；
继续按10.4实施方案的dev/batch/select/recommend/freeze/test顺序执行。

生产源码指纹已改变，先新建双构建并验收；旧base/GT/模型在合同和预算不变时原路径复用。
旧两集dev在最终共同协议下重新生成和测量。不要用旧protocol hash或重新seal旧结果迁移证据。

## 5. 本地测试与集群验收

```bash
"$PREP_PYTHON" tests/python/highdim_data_test.py
"$PREP_PYTHON" tests/python/highdim_integration_test.py \
  --build "$TOOLS_BUILD" --profile bioasq1024 --out "$TEST_ROOT/bioasq-smoke"
"$PREP_PYTHON" tests/python/highdim_integration_test.py \
  --build "$TOOLS_BUILD" --profile dbpedia1536 --out "$TEST_ROOT/dbpedia-smoke"
"$PYTHON" tests/python/edge_geometry_oracle_test.py \
  --native "$TOOLS_BUILD/uq_native_runner" --dimensions 128 960 1024 1536 3072 \
  --out "$TEST_ROOT/geometry"
```

完整集成测试的Python需同时能导入准备依赖、既有Faiss/绘图依赖；可用专门验证环境，
不改变冻结的集群软件环境。native依赖路径按平台设置。测试使用真实profile维度、
512 base/40 query的合成数据、2048条训练边和短迭代，不代表生产资产或正式QPS。
synthetic source必须显式`--allow-synthetic`，控制文件的小规模split仅允许synthetic输入，
该标记透传到dataset合同，正式runner拒绝它（只允许`--local-smoke`）。

oracle已扩展1024/1536/3072，保留默认128/960和原有绝对/相对容差，不放宽几何或负例验收。
Windows本地通过不替代Linux/OpenBLAS复验；新真实资产还必须dev17×OPQ/CR×batch1/8/128 parity，
随后再执行独占正式性能。查询错误/非有限/并列GT不通过时保留失败证据，不绕过验收。

原六后端离线质量/计时流程未删除或改成full-search方法。3072 paired实验、细分rotation/LUT计时、
全量真实数据准备和正式Slurm实验仍是后续独立步骤。
