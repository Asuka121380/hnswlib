# K256 与原横向对比的配对实验方案

## 夜间一键执行更新（2026-09-28）

用户选择一次提交。新增 ivf_k256_controlled_full.json，将下面 mid/high 的 efSearch 合并去重，161点在同一Slurm作业内统一随机排序，同节点同绑定核执行。mid/high文件保留用于分段分析，不再建议分别提交。

入口：bash scripts/edge_estimation/submit_ivf_controlled_overnight.sh --account db4ai --partition normal --qos normal

默认申请4 CPU/32G/12小时：4 CPU仅供并行编译；性能进程用taskset固定到分配范围内的一个CPU，各线程环境均为1。12小时为资源上限，不是预计耗时保证。完整网格仍是1次探索重复，不自动将161点放大为3次重复。

提交器把当前已提交HEAD通过git archive固定到独立运行目录。worker先清理旧模块影响，使用既有v0-pq环境CMake、系统编译器、wheel BLAS；检查输入身份、统一构建、重跑小图集成测试，再7点smoke与两个runner的无剪枝top-k对照，然后161点grid、汇总和出图。100-query baseline时间仅作诊断，不宣称runner性能等价。

输出根目录 ~/IndividualProject/results/ivf/controlled-overnight-*。正式结果为grid/points.csv、aggregated.csv、thresholds.csv、pareto.csv、recall_qps.png/pdf；根目录SUMMARY.md与COMPLETE.json代表整条链完成。失败保留FAILED.txt（阶段名）、分阶段日志及已有点；没有自动续跑。SOURCE在source/、编译信息在build/、环境信息在environment.txt。不复制GIST或已训练编码，不重新训练GIST模型。小图集成测试会生成独立合成测试模型。

新脚本已做本地语法与配置校验；Linux七方法运行仍由实际提交的smoke验收。不会从本地自动提交远程作业。

## 依据与已核实事实

本方案替代聊天中先前仅跑 27 点 IVF 高召回补测的建议。尚未提交本方案集群作业。

历史资料根目录：C:/ANU/IndividualProject/9.24/recall-qps-stage。
已读取 comparison-report-v1/README.md、tables/all_points.csv、method_summary.csv、threshold_summary.csv、原始 v1/v2 配置、manifest、代表性 command/performance JSON 与 Slurm 日志；已查看四张 figures/*.png。
历史有效结果为 v1 70 点 + v2 21 点，共 91 点，均为单次重复。失败的 164913 目录不纳入。
两轮历史 Slurm 日志均记录 cpusrv-1。当前 IVF pilot 的主机应读取其 manifest/作业日志确认，不能由先前训练所在节点推定。
历史包未提供完整 CMakeCache/编译命令，不能仅凭脚本示例证明旧二进制 native=OFF 或编译器与当前相同。

历史阈值最佳点（不是精确匹配 recall）：

| 门槛 | HNSW ef / QPS | PQ beta / ef / QPS | OPQ beta / ef / QPS | PQ+QJL theta / ef / QPS |
|---|---|---|---|---|
| 95% | 425 / 287.25 | 1.50 / 400 / 348.47 | 1.30 / 400 / 410.93 | 1.08 / 425 / 370.91 |
| 97% | 600 / 215.33 | 1.45 / 700 / 245.85 | 1.30 / 600 / 292.04 | 1.08 / 700 / 249.98 |
| 98% | 950 / 156.28 | 1.50 / 1000 / 161.23 | 1.35 / 1000 / 165.58 | 1.12 / 1000 / 172.08 |

旧 OPQ query rotation batch=128，且 query prep 纳入 QPS。旧 PQ+QJL 使用 v0 residual-threshold --theta；新 ivf_pq_qjl 使用新估计器 --beta。它们都被 CSV 命名为 beta，但不是天然等价的参数。
旧 OPQ 在 95/97% 门槛表现最好；98% 门槛旧 PQ+QJL 最快。所有历史方法都未达到98.5%，因此主目标先固定95/97/98%，98.5%只列诊断。

当前 IVF pilot-DvvMWP 共57点，全部完成。95%门槛 HNSW/PQ/OPQ/PQ+QJL QPS为299.96/346.98/427.67/274.17；97%门槛为208.82/210.49/303.95/179.08。上述历史表和当前表不得直接相除作为IVF收益。
同ef的HNSW旧/新QPS比值也随ef变化：ef200为583.20/529.92，ef425为287.25/299.96，说明不能用一个统一倍率校正旧结果。
当前PQ/OPQ的ef上限800、beta缺少旧关键点；先前QJL网格来自旧表示，已重新标定到balanced beta1.35、strict beta1.60。

## 问题与公平口径

七方法：HNSW、旧PQ8、旧OPQ、旧PQ+QJL、IVF+PQ、IVF+OPQ、IVF+PQ+QJL。
每个作业内同节点、同核绑定策略、单线程、随机交错。两阶段可能被分配不同节点，不能未经核对就将两轮QPS合并选前沿；记录hostname、CPU型号、commit、编译器、BLAS、配置及输入hash。
复用冻结index、base行mapping、官方1000 query和ground truth，k=10，warmup100。仍是exploratory；前100条已用于标定，不声明独立测试。
主码M32b8，QJL128bit；并非等总内存实验。报告sidecar/artifact总字节、RSS和query scratch。当前 IVF PQ/OPQ为52字节/边，QJL为76字节/边，另有共享模型。
旧/新OPQ查询旋转batch均为128，并计入prep；旧PQ/QJL逐query，新PQ/QJL批量准备LUT。主实验比较当前完整实现；不能把所有差异归因于粗中心本身。
性能前沿保留激进点；另外标注冻结trace质量门控。不能只筛新方法而不对旧方法施加同等准则。新OPQ1.20、PQ1.35和QJL1.30目前未通过balanced。
若要归因粗残差本身，还需相同kernel、训练边样本、随机种子和剪枝策略下的无coarse消融；当前七方法实验不是该消融。

## 可执行配置

configs/edge_estimation/ivf_k256_controlled_smoke.json：7点，100query，warmup20。只验接线与资产，不用于论文QPS。
configs/edge_estimation/ivf_k256_controlled_mid.json：92点，主测95/97%。
configs/edge_estimation/ivf_k256_controlled_high.json：69点，主测98%。

| 方法 | beta（旧QJL实为theta） | mid efSearch | high efSearch |
|---|---|---|---|
| HNSW | 无 | 200,425,600,750 | 850,950,1100 |
| PQ8 | 1.45,1.50,1.60 | 250,400,600,700 | 900,1000,1150 |
| IVF+PQ | 1.35,1.45,1.50,1.55,1.60 | 同PQ8 | 同PQ8 |
| OPQ | 1.30,1.35,1.40 | 250,400,600,700 | 900,1000,1150 |
| IVF+OPQ | 1.20,1.30,1.35,1.40 | 同OPQ | 同OPQ |
| PQ+QJL | 1.04,1.08,1.12 | 250,425,600,700 | 900,1000,1150 |
| IVF+PQ+QJL | 1.30,1.35,1.40,1.60 | 同PQ+QJL | 同PQ+QJL |

每行笛卡尔积。mid为4+12+20+12+16+12+16=92；high为3+9+15+9+12+9+12=69。共161个探索点。
这不是完全复制91点；它保留全部95/97/98%历史最佳参数组合，并补充新方法必要点。比只加入15个控制点更能避免将旧方法调参不足误认为IVF收益。
ef700用于找97%附近前沿；ef900/1000/1150直接承接旧高召回v2。先不默认扩展1300，只有目标仍未达到才成对追加所有相关方法。

## 运行前提与入口

使用现有 ivf_experiments.py grid；上述配置只用于grid，不应拿去train。
需重编译三个runner到同一新构建目录，并记录实际命令。保留当前成功的系统/usr/bin/c++、环境CMake、wheel OpenBLAS；module purge，native OFF，Release，PERFORMANCE_COMPARABLE_FLAGS ON。
除现有IVF构建选项外必须开启：
- HNSWLIB_ENABLE_V0_APPROX_REAL_PRUNING=ON
- HNSWLIB_ENABLE_V0_RESIDUAL_ESTIMATOR=ON
- HNSWLIB_ENABLE_V0_RESIDUAL_REAL_PRUNING=ON
构建v0_performance_runner、uq_opq_performance_runner、uq_ivf_performance_runner及验收工具。
不能直接把先前未编译legacy功能的IVF构建当成完整控制组构建。

历史manifest给出的旧资产（运行前验证存在及hash）：
- PQ: ~/IndividualProject/results/v0_offline/gist1m/20260726-98c5595-strict/gist1m_m32_nbits8_strict.v0meta
- QJL: ~/IndividualProject/results/v0_residual_estimator/p3-prototype-v1-092a1ef/companion/seed-42-k128.v0res
- OPQ: ~/IndividualProject/results/edge_estimation/c1-six-method-6e46627-20260926-165959/artifact-opq

grid命令除已有--config/--artifacts/--index/--queries/--ground-truth/--runner/--out外，加：
--v0-runner 新构建/v0_performance_runner
--opq-runner 新构建/uq_opq_performance_runner
--pq-sidecar 上述PQ路径
--qjl-companion 上述QJL路径
--opq-artifact 上述OPQ目录

先7点smoke；检查各method存在、旧PQ/QJL输出的method及beta/theta与请求一致、新runner剪枝计数非零、batch128、QPS计prep、正确率合理、hash/日志。旧PQ/QJL性能JSON不输出剪枝计数，标为不可用，不能由该输出证明实际剪枝次数。
另用v0 runner的--method baseline与IVF runner的--no-prune在相同ef400检查top-k及时间；前者不在现有七方法grid里，需单独运行。若baseline实现开销明显不同，需单列两个baseline，而非将runner差异归因于量化。
当前脚本支持legacy命令转发，但真实七方法集群联合运行仍未验收；本方案不宣称已通过。
完成smoke再跑mid和high，不重训K256，不启动K1024。

## 后续确认、batch与诊断

从配对探索结果中为每方法每个95/97/98%目标选择beta，并细扫ef使recall落在目标至目标+0.2个百分点；若无法落入，报告实际差值与两侧点，不伪造精确matched recall。
每方法每目标最多3个ef邻域，最多7*3*3=63个配置，各3次重复（最多189次）。用明确小配置文件表达选中(beta,ef)组合；不要使用会将所有beta与ef再做无关笛卡尔积的合并配置。
现有refine()只自动围绕95/98%，而本方案必须包含97%；不能直接把默认refine输出称为本方案最终确认集。
在同一作业中重测被选中点，聚合3次QPS均值/中位数/标准差，再选择前沿。重复是运行波动，不是新增独立query样本；不把缺乏重复的qps_std=0当作稳定性证明。
OPQ query batch后续做旧OPQ与IVF+OPQ成对的1/32/128/256对比；beta和ef固定于已选择的95/98%配置，不随batch重新调参。HNSW每组只作为共享baseline，batch completion不能冒充单query响应延迟。
QJL负面结果保留：比较正确/误剪、fallback原因、query prep、每候选成本、残差尺度、最终recall、编码/RSS；旧theta和新beta独立校准。可进一步固定PQ主码和coarse比较加/不加QJL，不因本轮落后直接删除该方法。
K1024在K256七方法配对结果明确后再启动；仍需重新训练/编码，但无须每次重训所有历史资产。

## 验证范围

本次仅生成方案与JSON配置，未启动远程作业，未改搜索算法，未宣称原始数据有新实验结果。
