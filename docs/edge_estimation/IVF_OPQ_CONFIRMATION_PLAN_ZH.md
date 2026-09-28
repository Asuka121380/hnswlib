# OPQ 收益确认与 query batch 对照（2026-09-28）

依据：本地 job145161 的161点完整结果，而非仅阈值汇总。旧OPQ和IVF+OPQ在beta1.30/ef400的QPS分别405.09/415.33；在ef600为287.37/294.38。原来的大幅收益点使用新方法beta1.20，但旧方法没有测1.20，存在搜索范围不对称。下一步必须补旧OPQ1.20，不能提前把全部收益归因于coarse。

入口：`bash scripts/edge_estimation/submit_ivf_controlled_overnight.sh --account db4ai --partition normal --qos normal --opq-confirm`。
复用已修复的统一构建、wheel运行库、7点smoke和baseline bridge，固定源码快照。在同一节点同一个绑定CPU上顺序执行两个阶段，阶段内每次repeat随机交错。申请4CPU/32G/12小时，编译可并行、搜索及BLAS单线程。无GIST重训、无K1024。

## 第一阶段：63配置，各3次，共189次

| 方法/beta | 95%附近ef | 97%附近ef | 98%附近ef |
|---|---|---|---|
| HNSW | 350,375,400 | 550,575,600 | 850,900,950 |
| 原OPQ与IVF+OPQ，beta1.20 | 425,450,475 | 650,675,700 | 1000,1075,1150 |
| 原OPQ与IVF+OPQ，beta1.30 | 350,375,400 | 575,600,625 | 900,950,1000 |
| 原OPQ与IVF+OPQ，beta1.35 | 350,375,400 | 575,600,625 | 900,950,1000 |

原/新OPQ使用完全相同(beta,ef)集合，query rotation batch128。HNSW batch1。每配置官方1000 query，预热100，k10，native OFF、Release、BLAS ON。
新增explicit cases支持避免无关的beta/ef笛卡尔积，保留旧grid模式。先聚合3次QPS中位数，再挑选点。可直接比较同参数与相近Recall两种口径。

## 第二阶段：自动固定参数后测batch，最多81次

对每方法每目标95/97/98%，在[目标,目标+0.2个百分点]内选中位QPS最高的点；没有点进入窗口但有更高Recall点时，选最近上侧点并标OUTSIDE_RECALL_WINDOW；目标不可达则标NOT_REACHED，不把低于目标的点拿来冒充。
OPQ两个版本分别固定各目标选中beta/ef，query batch1/32/128/256，各3次。HNSW各目标选中点batch1各3次。跨目标重复的配置去重，因此不超过(2*3*4+3)*3=81次。
该阶段仅解释已选配置的batch敏感性，不代表每个batch的独立调参最优曲线。两阶段都复用相同query，第二阶段是运行重复确认而非独立测试集。

## 结果与判读

`opq-confirm/`和`opq-batch/`分别包含原始逐点performance/results/latency、聚合表、前沿、阈值表和图。`opq-configs/`冻结两阶段实际配置；`opq-selected.json`包含每目标的选择与窗口状态；`OPQ_STAGE.txt`指出当前阶段；根目录`SUMMARY.md`和`COMPLETE.json`标识整条链完成。
主要确认：1)原OPQ1.20是否追回差距；2)相同beta/ef下coarse收益是否稳定；3)接近Recall时提升是否超过3次运行波动；4)batch1是否仍有优势。
beta1.20仍是探索参数，未通过先前新OPQ的静态balanced门控；旧OPQ的误剪门控尚未统一测量。beta1.30/1.35只是保守参数对照，不自动宣称两方法均满足同一风险阈值。
若最近上侧点仍偏离窗口，应进一步补ef；不通过插值声称测到了精确matched recall。3次重复不足以证明普遍显著性，报告每次值、中位数、均值、标准差和实际Recall。

本地验证：5项配置/选择回归、既有训练编码到搜索集成测试、显式cases的9次真实native搜索与3次聚合、shell语法检查通过。远程新实验尚未提交；脚本不能保证12小时内一定完成。
