# PhDPaper3 当前交接

## 1. 项目目标

PhDPaper3 是可复现的 SDWPF 时间序列预测科研实验工程。统一入口是
`scripts/run.py`，正式模型为 LSTM、Crossformer 和 STCN；训练、
评估、指标和汇总逻辑由共享路径提供。

## 2. 当前稳定架构

父调度器按模型 YAML 选择环境，并为每个模型启动独立 worker。Crossformer 使用
Node Shared 和 `env_tslib`；STCN 使用 `env_tsl` 和公共 `k=5` 物理图。训练前
执行 resolved-shape 检查；默认 Full-shape 与覆盖 batch 检查分开报告。

长期公共状态只有：

- 批次：`results/_runs/<run-id>/status.json`；
- 模型：`results/<model>/<run-id>/run_info.json`；
- 独立检查：`results/_checks/<check-id>/status.json` 和模型结果文件。

顶层状态只有 `PENDING/RUNNING/PASS/FAILED/SKIPPED`，失败分类保持粗粒度，
具体原因放在 `error.code` 和 `details`。新状态使用 schema v2，读取器兼容
schema v1。`validation_status.json` 只是 worker 内部临时通信文件，不是长期
公共接口；Resume、完成判断和 summarize 不依赖它。

## 3. 已完成

- 公共状态写入集中到 `src/runtime/status.py` 的 `normalize_status_payload()`
  和 `write_status()`；新运行、worker 和 repeatability 状态统一写 schema v2。
- profile 与 classification 已分离；Resume/Skip completed 只依赖稳定状态、
  profile 和必要 artifacts。
- 已有结果可通过 summarize 重新生成汇总；`model_comparison.csv` 是论文/Excel
  两行分组表头版本，`model_comparison_flat.csv` 是程序读取版本。
- 保留环境调度、Smoke、Full-shape、Repeatability、Crossformer、STCN、LSTM
  和共享数据流程。

## 4. 当前限制

正式训练仍需要 `dataset/` 中两个协议命名的 parquet 文件。没有对应 CUDA 机器
和数据时，不声称完成正式 GPU 实验；正式 Full 训练未执行时标记 `NOT RUN`。

## 5. 下一步

接手新模型时先读 `MODEL_INTEGRATION_INDEX.md`，再读公共配置、共享数据/训练/
评估路径和调度器；只新增模型代码与模型 YAML，运行相关检查和测试。

## 6. 项目维护原则

本项目是科研实验工程，不是企业平台。优先正确、可运行、公平、可复现、容易
维护和容易接入模型。不要过度设计，不要为猜测的未来需求提前增加复杂抽象；
公共接口要少且稳定，只有存在两个以上真实调用方时才考虑新增公共抽象。模型
特殊需求优先放在 Adapter 或模型 YAML 中；公共数据、Trainer、Evaluator、
Metrics、图和汇总逻辑只实现一次；配置优先于硬编码。

不要重新引入 StudySpec、ModelSpec、证书、Manifest、Readiness、复杂协议层或
模型专属 Trainer/Evaluator。

## 7. 文档更新规则

- `README.md` → 稳定用户入口、常用命令和主要输出。
- `MODEL_INTEGRATION_INDEX.md` → Codex 读取顺序、模型接入规则和共享接口。
- `docs/COMMAND_REFERENCE.md` → 由 `command_schema.py` 自动生成，禁止手工修改。
- `HANDOFF.md` → 当前任务状态、限制、下一步、维护原则和踩坑。

CLI 改变时修改 `command_schema.py` 并运行生成器；普通模型参数只改模型 YAML；
公共实验参数只改 `experiment.yaml`，只有长期语义或用户操作变化才更新手写
文档。新增模型只有共享接入方式变化时才修改模型索引。普通重构、变量改名或
测试数量变化不更新所有 Markdown。

## 8. 不要再踩的坑

- 不要用 batch 1 Shape PASS 冒充默认正式 Shape PASS。
- 不要从 k=4 STCN checkpoint Resume 到 k=5。
- 不要从 metrics JSON 的 display 字段读取完整指标，也不要把 NaN R2/MAPE 填成 0。
- 不要用 validation monitor 代替测试 Horizon Score。
- 不要让 Time-Series-Library 的 `models` 覆盖项目 `src/models`；纯 `tsl` 模型不得绑定它。
- 不要手工修改 `COMMAND_REFERENCE.md`，不要无意义更新所有 Markdown。
- 不要提交 dataset、results、logs、checkpoint 或外部库；不要强制推送或批量递归删除。

## 9. 新会话接手顺序

先读取本文件和 `MODEL_INTEGRATION_INDEX.md`，检查 `git status`，再按索引读取
配置、共享路径、调度器和相关测试。修改前确认没有触及模型、数据流程、训练
协议、损失函数、评估公式、公共 `k=5` 图或汇总 CSV 契约。

## 10. RA-DS-PFD Crossformer 当前任务

- P0 状态：`PASS`。
- P0 完成提交：`7687a418bf94b74c5db91edaf55bbbeebe9959a4`。
- P0 已完成当前 Crossformer Adapter、upstream Crossformer、旧原型、双尺度插入点、
  Segment Merging、Decoder、图方向、Static/Relation Bias、Entmax 状态和 formal
  shape 显存风险的审计。
- P0 详细审计曾随完成提交进入 Git 历史；当前工作树不长期保留阶段快照或阶段索引，
  以维持四份 Markdown 和单一当前状态来源的维护规则。
- 当前无 P0 技术阻塞。P4-A 开始前仍需解决 Entmax 依赖；P2 不引入 Entmax。
- P1 状态：`PASS_WITH_NOTES`。已新增受版本控制的 local canonical Backbone、Node
  Shared Adapter、模型 YAML 和 P1 测试；实现只复用真实 upstream canonical 子模块，
  并在 Scale0/Scale1 的 Cross-Time 后、Cross-Dimension 前执行严格 identity。
- P1 canonical equivalence 已通过 strict/bijective state transfer；小形状与当前 formal
  双尺度 CPU 检查的 full-channel 和最终功率输出最大绝对差均为 `0`（`atol=1e-6`、
  `rtol=1e-6`）。奇数 merge 检查使用 `lookback=60`、Scale0 `5` 段、Scale1 `3` 段，
  padding/数值与 upstream 一致；shared checkpoint reload、forward/backward finite
  和现有 Crossformer/isolation 回归均通过。
- 当前执行语义已收口：`spatial_disabled=true` 的 P1 concrete adapter 继承
  `NodeSharedForecastModel`，使用公共 `node_shared_microbatch`；134 个节点范围为
  `0:32,32:64,64:96,96:128,128:134`。`spatial_disabled=false` 的 P2 concrete
  adapter 继承普通 `ForecastModel`，保持 `full_spatiotemporal`；其内部
  `spatial_edge_chunk_size` 与公共 node chunk 独立。P1/P2 共享同一 `backbone.*`
  canonical implementation。
- P2 状态：`PASS_WITH_NOTES`。P2 代码、focused tests、完整 pytest、P1 canonical
  回归以及 LSTM/Crossformer/STCN 回归均已通过；正式训练仍未执行。
- Relation Resource 是模型局部的只读 NPZ 契约，不改公共 `GraphResource`。文件使用
  项目相对路径和版本化文件名；artifact 内部 `schema_version=1`；只允许
  `node_ids`, `edge_index`, `edge_static_features`, `edge_feature_names` 四类数据字段，
  使用 `allow_pickle=False`；节点顺序、`[2,E] int64` source/target、无重复/自环、
  `(target,source)` 稳定排序、target 非零入度、`[E,13] float32` finite 和精确字段名
  都在模型构造时 fail closed 校验。公共图仍保持 `adjacency[source,target]`、
  `edge_index[0]=source`、`edge_index[1]=target`、`k=5`。
- The P2 relation resource now follows the project-wide minimal-validation
  principle in `MODEL_INTEGRATION_INDEX.md`: structural correctness checks remain,
  while unused file-hash and certificate-style identity machinery is removed.
- 2026-08-05 minimal-validation simplification acceptance：公共环境哈希删除、
  P2 Relation Resource file-only 配置及结构校验修改完成；focused tests
  `45 passed`，完整 pytest `181 passed, 3 skipped`，command reference、
  P1 CUDA preflight 和 P2 opt-in fixture build/forward 均通过。本文档其他位置的
  `57 passed, 3 skipped`、`170 passed, 3 skipped`、`177 passed, 3 skipped`
  和 `8 passed` 均为各自阶段的历史验收证据，不代表当前完整测试总数。
- 13 个 static edge feature 名称固定为：
  `semantic_similarity`, `semantic_overlap_ratio`, `distance_kernel_weight`,
  `normalized_distance`, `relative_x`, `relative_y`, `delta_elevation`,
  `terrain_slope`, `terrain_slope_angle`, `is_semantic_edge`, `is_distance_edge`,
  `is_both_edge`, `has_elevation`。TrueUnion 的 semantic∪distance 去重和最终排序由
  预构建 artifact 提供，模型不读取 parquet、target、mask、未来数据或功率序列。
- PFD0 只解析 `DataInfoView.feature_columns` 中的 `Wspd`。`level[t]` 是当前输入窗口
  的 Wspd level；`diff1[0]=0`，`diff1[t]=level[t]-level[t-1]`。编码使用旧原型的
  左侧 zero padding，Scale1 奇数段复制最后 segment；formal `S0=12,S1=6`。
  Value 只来自这两个候选，禁止传播功率、target/mask、邻居完整 hidden state 或
  turbine embedding。
- Relation spatial 使用 target 自身 `[B,N,C,S,D]` Cross-Time token 生成 Query，
  source 的 PFD0 token 生成 Key/Value；按 `edge_index[1]` target 入边 softmax，
  `edge_index[0]` source gather 后 scatter 到 target，不建立 dense `N×N` attention。
  StaticEdgeMLP 和 OrderedRelationBias 只加 logits；两个 scale 各调用一次，分别执行
  `T + gamma * Dropout(Wo(M))`，两个独立 gamma 初始值均为 `0.1`，无 variable gate。
- `spatial_disabled=true` 仍接受旧 P1 配置，完全不读取 relation artifact、不创建 PFD0
  或 P2 persistent buffer，且 state_dict/canonical 数值路径保持 P1 严格等价。P2 只接受
  `pfd_mode=pfd0`；P1 checkpoint 与 P2 配置的 model-config identity 不匹配时由现有
  checkpoint compatibility 拒绝 Resume。当前正式 YAML 保持 legacy disabled 状态，
  因为 134 节点正式 relation artifact 不应作为仓库 fixture 或私有数据提交；P2 opt-in
  配置和 3 节点合成 fixture 已被测试。
- 测试证据：focused `57 passed, 3 skipped`（env_tslib 中正式 tsl 缺失的既有 skip）；
  完整 `170 passed, 3 skipped`；env_tsl 单独 STCN 回归 `3 passed`；命令 reference
  检查通过，受跟踪 Markdown 仍为四份。P1 public resolved batch-1、formal default
  shape 和 preflight 均 PASS；这些命令读取的是 legacy disabled YAML，不宣称为 P2
  relation formal PASS。
- 额外使用 P0 审计目录中 E=1536、train-only、无 future target 的旧 TrueUnion 数据，
  临时转换为上述 NPZ 后执行了 formal `B=32,L=144,N=134,C=16,H=10` P2 forward/backward，
  output/gradients finite；临时 artifact 已从工作区删除，未提交。该结果不等同于
  正式训练或把临时 artifact 作为公共资源发布。
- P2 明确未实现：7 个额外安全气象变量、56 candidates、CausalPropagationFeatureBank、
  learned_change12、PFD1、Entmax、Top-k、Straight-Through、PFD2–PFD5 Selector、
  source dynamic context、候选稀疏选择和正式 Full 训练。
- 下一阶段只允许：`P3：7 个安全气象变量、56 个稳定因果候选、
  CausalPropagationFeatureBank、learned_change12 和 PFD1 Dense Candidate Propagation`。

## 11. 共享 controlled-nonstrict 修复与当前 P2 状态

`main` 是自定义模型长期分支，继续保留并推进 RA-DS-PFD。共享修复链为
`bd64db6`、`76ac9ac`，以及后续的隔离 evaluate-only worker 和 AMP-safe
Repeatability 补丁；均已同步到 `baseline-models`。全局
`torch.use_deterministic_algorithms(True)` 已关闭；固定 Python/NumPy/PyTorch/CUDA
seed、DataLoader generator 和数据顺序保留，`cudnn.deterministic=true`、
`cudnn.benchmark=false` 保留，TF32 已关闭。父调度器在每个 worker 启动前注入
`PYTHONHASHSEED=<resolved training.seed>` 与 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
evaluate-only 复用同一 worker 路径，不在父进程直接初始化 seed。

当前 AMP float16 的 Repeatability 固定默认值为 prediction `atol/rtol=5e-3`、
metric `atol/rtol=2e-4`，仍可由 CLI 显式覆盖，容差写入报告。完整三模型默认双跑已
通过：Run A 为 `LSTM, Crossformer, STCN`，Run B 为反序；前两者为
`EXACT`，STCN 为 `NUMERICAL`，不同 worker PID、exact 字段、预测、validation/test
metrics 均通过。

共享验收已在本机完成：主工作树完整 pytest `177 passed, 3 skipped`（3 个 skip 是
`env_tslib` 中正式 `tsl` 只在 `env_tsl` 可用），`env_tsl` 的 STCN 回归 `3 passed`，
command reference check 通过；LSTM/Crossformer/STCN 的真实数据 preflight、
`B=32,L=144,N=134,C=16` `FORMAL_DEFAULT_SHAPE`、B=32 Smoke 均通过。STCN 使用
CUDA AMP float16，formal shape 峰值为 `5479.02 MiB allocated / 7086 MiB reserved`，
Smoke（含 1 optimizer update）峰值为 `4568.45 / 5118 MiB`，未出现 strict 路径的
20+ GiB 显存问题；不得降低 batch 或静默切换设备。

RA-DS-PFD P2 分支只增加了 controlled-nonstrict finite 回归测试，没有在缺少正式资源
时制造 checkpoint/chunk 修复。公共 YAML 仍为 `spatial_disabled=true` 的 P1/legacy
路径；启用 P2 的真实小夹具 `N=3,E=4`、`pfd_mode=pfd0`、`d_model=64`、`B=32`、
AMP float16、edge chunk 128 在两个独立进程中完成 forward/backward/optimizer step，
output/loss/gradient finite，状态哈希一致，峰值 `92.48 / 116 MiB`；P2 focused
测试为 `8 passed`。这只是 enabled P2 smoke 证据，不是 Full。

在本任务开始前工作区没有正式 134 节点 enabled P2 relation NPZ；当时的 P2 正式 GPU、
20 updates、Full 和正式 P2 Repeatability 因此标记为 `NOT RUN`。该历史状态不代表本次
comparison foundation 的一次性形状验收。没有 OOM 证据，因此未实现 activation
checkpointing，也未改变模型训练协议。该 P2 专项只进入 `main`，不得将 `baseline-models`
整体合入 `main`，也不得把 RA-DS-PFD 代码带入 `baseline-models`。

本次公共 NodeShared execution 已同步到 main：sample batch 与 node chunk 分离，唯一
公共 `runtime.node_shared_chunk_size` 默认值为 `32`；LSTM/Crossformer 使用 shared
node microbatch，134 个节点自然分为 `32/32/32/32/6`；STCN 与 RA-DS-PFD P2
保持完整 `[B,L,N,C]` 执行，RA-DS-PFD P1 使用同一公共 node microbatch。NodeShared
检测到原生 BatchNorm 时保留 full-node，不改为 LayerNorm；正式 compatible
NodeShared OOM 只能统一下调公共 chunk，禁止 per-model rescue。

## 12. P2 comparison foundation（2026-08-06）

- TrueUnion 构建器、旧图转换器和 NPZ 写入入口为
  `src/models/ra_ds_pfd_crossformer/relation_builder.py` 与
  `scripts/build_ra_ds_pfd_relation.py`；正式本地派生资源为
  `dataset/derived/ra_ds_pfd/ra_ds_pfd_trueunion_sdwpf_v1.npz`。
- 正式重建结果为 `N=134,E=1536`，semantic/distance/both 为 `1340/670/474`；
  与具有 `node_order_hash` 的旧目录 `unified_relation_c86b63279060c5c7` 的
  node_ids、edge_index、字段名 exact，静态特征 `atol=rtol=1e-5` allclose。较早的
  `unified_relation_863fb48bb6389791` 缺少可证明的节点顺序字段，转换按协议拒绝。
- 四轴允许值为：`spatial_query_mode={per_variable,node_pooled}`、
  `propagation_encoder_mode={segment_fusion,cross_time_then_fusion}`、
  `turbine_embedding_mode={relation_only,temporal_and_relation}`、
  `bias_scaling_mode={direct,learnable_per_scale}`。Current P2 是四轴第一项组合；
  Old-concept endpoint 是四轴第二项组合。
- 16 个小形状组合均完成 build/forward/backward；正式 `B=32,L=144,N=134,C=16,
  horizon=10` 两端点在等价 `edge_chunk=512` 下 output/loss/required gradients finite。
  这些历史峰值是 PyTorch allocator 统计，不是 WDDM 下可与物理 VRAM 总量比较的
  authoritative physical-memory evidence；本次审计已将其替换，不再作为正式显存值。
- 本任务未实现 P3/P4；未运行 R0–R7 Full、多 seed 对照或正式训练。
- `P2_COMPARISON_FOUNDATION = PASS_WITH_NOTES`；`OLD_RESOURCE_SAFE_TO_DELETE = NO`
  （未能证明的旧目录仍需用户自行保留/处置）。

## 13. Shared NodeShared software closeout（2026-08-07）

公共 NodeShared software closeout complete：`runtime.node_shared_chunk_size` 的
唯一运行值来源是 `configs/experiment.yaml`，production code 不提供独立
fallback；Resume compatibility 只约束当前 `ExecutionPlan` 实际使用
`node_shared_microbatch` 的模型，`full_nodes`/`full_spatiotemporal`（包括
RA-DS-PFD P2）不因 chunk 值变化失效。Formal shape validation 已改为 resolved
AMP semantics；P2 数学路径、relation resource 和 edge chunk 未改变，P1/P2 的
execution adapter 语义见上文。

此前 target GPU 被其他任务占用，因此当时 LSTM/Crossformer/STCN/RA-DS-PFD
GPU gate 均为 `NOT EXECUTED — target GPU currently occupied`；real CUDA/cuDNN
LSTM recurrent-dropout two-pass replay 当时 pending。`node_shared_chunk_size` 保持
`32`，未下调到 `16` 或 `8`。P0/P1/P2、relation resource 和 P2 comparison
foundation 状态继续以本 HANDOFF 前文为准；2026-08-11 的 GPU 收口见下文。

## 14. RA-DS-PFD R0-R7 Matrix Foundation（2026-08-10）

- `R0_R7_MATRIX_FOUNDATION = PASS`：该状态仅表示 machine-readable matrix、resolver
  和 CPU structural/software foundation 已通过权威验收。
- `configs/experiments/ra_ds_pfd_r0_r7.yaml` is the sole machine-readable R0-R7 matrix
  source; the resolver validates schema and relational invariants without maintaining a
  second full matrix copy. Resolver 为 `src/models/ra_ds_pfd_crossformer/r0_r7_suite.py`；权威矩阵测试为
  `tests/test_ra_ds_pfd_r0_r7_suite.py`。R0 严格保持 canonical P1 identity；R1
  为 current P2 mapping；R2 为 old endpoint mapping；R3/R4/R5/R6 分别只改变
  Query、Propagation Encoder、Turbine Embedding、Bias Scaling；R7 只改变 Query
  与 Propagation Encoder 两轴。
- R0 使用 `NodeShared` execution；R1-R7 使用 `full_spatiotemporal` execution。
  R1-R7 共享 canonical backbone、P2 参数、relation resource 和
  `spatial_edge_chunk_size=512`；本次 GPU closeout 后该值成为统一的
  GPU-validated common selected edge chunk。
- focused CPU acceptance：`67 passed`。完整 CPU-only acceptance：`247 passed,
  3 skipped`；skip 为既有 env_tslib 中正式 `tsl` 不可用的 STCN 测试。权威命令为
  `pytest -q --ignore=tests/test_full_shape.py`。
- 2026-08-11 `RA-DS-PFD GPU Memory Evidence Audit` 已完成；正式 R1 resolver、
  relation resource、AMP、shape 和执行语义均通过，R1 formal F/B、one-step、20-step
  均 PASS。WDDM physical FB 证据统一来自独立 `nvidia-smi` device-wide
  `memory.used` sampling；PyTorch allocator peaks 只保留为独立诊断，绝不解释为
  physical VRAM。全部 authoritative workload 前均执行 `nvidia-smi` cleanliness gate；
  没有其他 CUDA compute workload，桌面 C+G 占用计入 baseline。

## 15. RA-DS-PFD WDDM GPU / P2 / Stage A closeout（2026-08-11）

- GPU：单卡 NVIDIA GeForce RTX 4070 Ti SUPER，WDDM，driver `591.86`；
  `nvidia-smi` physical total `16376 MiB`，Python `3.11.15`，Torch `2.5.1+cu124`，
  CUDA runtime `12.4`。正式 shape 为 `B/L/N/C/H=32/144/134/16/10`，AMP
  `float16`，`controlled_nonstrict`，seed `2026`。
- Measurement：sampler 为
  `nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits -lms 50`；
  baseline 是 workload 启动前最后 20 个有效样本的中位数，并要求该窗口范围不超过
  `64 MiB`；peak 是 workload 期间/结束后采样 `memory.used` 的最大值；delta 是
  `peak - baseline`。该 physical FB 指标是 device-wide WDDM evidence，不是 per-process
  resident VRAM。
- R1 memory finalization：formal F/B `baseline/peak/delta=2576/16032/13456 MiB`，
  one-step `1011/15973/14962 MiB`，20-step `1385/15963/14578 MiB`；三轮均
  output/loss/required gradients finite、正式 Adam 成功、无 CUDA OOM。20-step
  losses 全部 finite，结束后 device-wide used 回落，无异常持续上涨。对应 allocator
  diagnostics：F/B `18927.97/19460.0 MiB`，one-step `18927.97/19460.0 MiB`，
  20-step `18964.02/19464.0 MiB`（allocated/reserved）。

| Variant | Execution | Params | Physical FB baseline | Physical FB peak | Physical FB delta | PyTorch peak allocated | PyTorch peak reserved | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R0 | node_shared_microbatch | 532,068 | 904 | 4,069 | 3,165 | 2,711.90 | 2,836.0 | PASS |
| R1 | full_spatiotemporal | 603,150 | 766 | 16,025 | 15,259 | 18,927.97 | 19,460.0 | PASS |
| R2 | full_spatiotemporal | 760,816 | 660 | 14,016 | 13,356 | 12,841.31 | 13,008.0 | PASS |
| R3 | full_spatiotemporal | 611,600 | 691 | 13,634 | 12,943 | 12,456.03 | 12,604.0 | PASS |
| R4 | full_spatiotemporal | 750,990 | 915 | 16,045 | 15,130 | 19,308.46 | 20,620.0 | PASS |
| R5 | full_spatiotemporal | 604,510 | 553 | 16,015 | 15,462 | 18,927.73 | 19,460.0 | PASS |
| R6 | full_spatiotemporal | 603,166 | 638 | 16,026 | 15,388 | 18,927.97 | 19,460.0 | PASS |
| R7 | full_spatiotemporal | 759,440 | 673 | 14,003 | 13,330 | 12,841.54 | 13,008.0 | PASS |

- Stage A 统一通过；R0 节点范围为 `32/32/32/32/6`，R1-R7 全部使用
  `spatial_edge_chunk_size=512`。最大 physical FB peak 为 R4 `16045 MiB`，最大
  physical delta 为 R5 `15462 MiB`；二者均未超过 device total，且没有 variant
  触发真实 CUDA OOM。`GPU_MEMORY_EVIDENCE_AUDIT = PASS`，
  `P2_MEMORY_FINALIZATION = PASS`，`R0_R7_GPU_STAGE_A = PASS`；selected common
  edge chunk = `512`。
- 该表是 single-step Stage A，不是 R0-R7 Full。R0-R7 Full、multi-seed、formal
  test-set comparison、P3/P4 均未运行；Hybrid Self-View node chunk 与 activation
  checkpointing 均未实现。

## 16. RA-DS-PFD R0-R7 Execution Foundation（2026-08-11）

- `R0_R7_FULL_READINESS_MEMORY_GATE = PASS`：在每个 variant 各自独立 Python
  process、每轮前 `nvidia-smi` cleanliness check、正式 train split/DataLoader/loss/
  Adam/AMP/Trainer execution primitive 下，R4 与 R5 均完成 20/20 optimizer updates。
  全部 loss 与 required gradients finite，optimizer state 每步恰好前进一次，无
  CUDA OOM、NaN/Inf 或持续单调 physical FB 泄漏。配置保持
  `B/L/N/C/H=32/144/134/16/10`、seed `2026`、AMP `float16`、
  `controlled_nonstrict`、relation resource
  `dataset/derived/ra_ds_pfd/ra_ds_pfd_trueunion_sdwpf_v1.npz` 与 common
  `spatial_edge_chunk_size=512`。
- R4 20-step device-wide physical FB `baseline/peak/delta=1380/16051/14671 MiB`；
  baseline last-20 range `1380..1417 MiB`。PyTorch allocator diagnostics 为
  `19353.26/20630.0 MiB` allocated/reserved。R5 为
  `591/16017/15426 MiB`；baseline range `588..599 MiB`；allocator diagnostics
  `18966.15/19468.0 MiB`。Allocator 数值不作为 physical VRAM。
- `R0_R7_EXECUTION_FOUNDATION = PASS`：正式 suite entry 为
  `scripts/run_ra_ds_pfd_r0_r7.py`，支持严格互斥的 `--variant Rk`、
  `--variants R1,R3,...`、`--all` 与 `--dry-run`。`--all` 稳定按 R0..R7；默认
  fail-fast。runner 不提供 batch、epochs、seed、AMP、loss、split 或 edge-chunk
  专用覆盖，只调用既有 `scripts/run.py train` → orchestrator → isolated worker →
  Trainer。
- `configs/experiments/ra_ds_pfd_r0_r7.yaml` 仍是唯一 matrix SSOT。runner 通过
  `resolve_r0_r7_variants()` 生成临时 `runtime/model` YAML；临时文件位于系统临时
  目录，结束/异常后清理，不写入 `configs/models/`。每个 variant 使用
  `<base-run-id>__Rk` 独立 identity；result path、run_info/status 与持久化
  `model_config.yaml` 可直接识别 variant。Resume 复用公共 checkpoint compatibility，
  比较 resolved model-config 内容而非临时路径，因此 R3 只能恢复 R3，R4 config 会被
  R3 checkpoint 拒绝。
- Execution Foundation 验收：suite + runner focused `18 passed`；CLI/orchestration/
  execution/config/resume focused `113 passed`；完整 CPU-only regression
  `255 passed, 3 skipped`（3 个 skip 为 env_tslib 中无正式 `tsl` 的既有 STCN
  条件 skip）。`--all --dry-run` PASS，R0 active node chunk `32` 且 edge chunk
  `not applicable`；R1-R7 均为 `full_spatiotemporal`、edge chunk `512`，resolved
  四轴与 suite resolver 一致。未启动正式训练。
- Explicitly NOT RUN / NOT IMPLEMENTED：`R0-R7 Full = NOT RUN`，
  `multi-seed = NOT RUN`，`formal test comparison = NOT RUN`，`P3/P4 = NOT RUN`，
  `Hybrid = NOT IMPLEMENTED`，`activation checkpointing = NOT IMPLEMENTED`。

## 17. RA-DS-PFD R0-R7 Execution Foundation Final Closeout（2026-08-11）

- `R0_R7_EXECUTION_FOUNDATION_CLOSEOUT = PASS`。suite runner 已支持单一
  `--smoke` 布尔开关；该开关只转发既有公共 train smoke profile，不提供或重定义
  suite 专用 smoke epochs/update/eval 参数。
- 单 R1 real smoke 使用 `ra_ds_pfd_r0_r7_r1_smoke_seed2026__R1`，结果为
  `PASS / SMOKE`。全 suite 使用独立 base identity
  `ra_ds_pfd_r0_r7_all_smoke_seed2026`，R0-R7 均为 `PASS / SMOKE`；每个 result
  directory identity 唯一且 artifacts 完整。该验收是 execution/provenance integration
  smoke，不是 20-step memory gate，也不是 Formal Full。
- Temporary model YAML lifecycle 为 PASS：runner 在一个系统临时目录中逐项生成
  resolver 的 `runtime/model` 文档，每个 variant 结束后删除 YAML，suite 结束后删除
  空目录。公共 worker 持久化的 `model_config.yaml` 现在保留完整、已验证的
  `runtime/model` 文档，八个 suite smoke 均与 resolver 输出精确一致，可直接传给
  `scripts/run.py train --model-config` 重放。
- `command.json` 不回写、不篡改原始 `argv`；`model_config_path` 继续记录真实临时
  invocation path，新增 `replay_model_config_path` 指向同一 result directory 内实际存在的
  `model_config.yaml`。Resume compatibility 继续比较 checkpoint 与当前 resolved model
  config 内容，不依赖临时路径；实际 R1 smoke checkpoint 与当前 R1 resolver compatible，
  R3/R4 不兼容回归继续 PASS。
- Closeout 验收：suite/runner focused `21 passed`；CLI/provenance/documentation focused
  `40 passed`；完整 CPU-only regression `258 passed, 3 skipped`，3 个 skip 仍是
  env_tslib 中无正式 `tsl` 的既有 STCN 条件 skip；generated command reference
  consistency PASS。
- `R0-R7 Formal Full = NOT RUN`。本 closeout 完成后停止；Formal Full 只能在下一批由
  用户显式启动。

## 18. R0-R7 Replay Metadata Portability Closeout（2026-08-11）

- `R0_R7_REPLAY_METADATA_CLOSEOUT = PASS`。`replay_model_config_path` 从 absolute
  result path 收口为 result-directory-relative `model_config.yaml`；整目录 archive/rename
  relocation test PASS。原始 `argv` 与 `model_config_path` provenance 未修改，历史
  artifacts 未迁移，checkpoint/resume 语义未修改。
- `R0-R7 Formal Full = NOT RUN`。
