# PhDPaper3 当前交接

## 1. 项目目标

PhDPaper3 是可复现的 SDWPF 时间序列预测科研实验工程。统一入口是
`scripts/run.py`，正式模型为 NodeSharedLSTM、Crossformer 和 STCN；训练、
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
- 保留环境调度、Smoke、Full-shape、Repeatability、Crossformer、STCN、NodeSharedLSTM
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
- P2 状态：`PASS_WITH_NOTES`。P2 代码、focused tests、完整 pytest、P1 canonical
  回归以及 NodeSharedLSTM/Crossformer/STCN 回归均已通过；正式训练仍未执行。
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
通过：Run A 为 `NodeSharedLSTM, Crossformer, STCN`，Run B 为反序；前两者为
`EXACT`，STCN 为 `NUMERICAL`，不同 worker PID、exact 字段、预测、validation/test
metrics 均通过。

共享验收已在本机完成：主工作树完整 pytest `177 passed, 3 skipped`（3 个 skip 是
`env_tslib` 中正式 `tsl` 只在 `env_tsl` 可用），`env_tsl` 的 STCN 回归 `3 passed`，
command reference check 通过；NodeSharedLSTM/Crossformer/STCN 的真实数据 preflight、
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

当前工作区没有正式 134 节点 enabled P2 relation NPZ（要求的 `N=134,L=144,C=16,
H=10` 真实 artifact），因此 P2 正式 GPU、20 updates、Full 和正式 P2 Repeatability
标记为 `NOT RUN`，不得把 3 节点夹具或历史临时审计资源冒充正式验收。没有 OOM 证据，
因此未实现 activation checkpointing，也未改变 edge chunk、模型结构、batch、loss 或
训练协议。该 P2 专项只进入 `main`，不得将 `baseline-models` 整体合入 `main`，也不得
把 RA-DS-PFD 代码带入 `baseline-models`。
