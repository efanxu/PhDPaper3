# PhDPaper3 当前交接

## 1. 项目目标

PhDPaper3 是可复现的 SDWPF 时间序列预测科研实验工程。统一入口是
`scripts/run.py`，正式模型为 LSTM、Crossformer、STCN、DLinear、TSMixer、
SegRNN、iTransformer、TimesNet、TimeMixer、Transformer、LightTS、TiDE、
FreTS、FiLM、Informer、Autoformer、STID、DCRNN、AGCRN、GraphWaveNet、
GRUGCN、RNNEncGCNDec、PureGCN、PatchTST、Nonstationary Transformer、
FEDformer 和 EvolveGCN；训练、
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
- 保留环境调度、Smoke、Full-shape、Repeatability、共享数据流程和全部基准
  模型接入。

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

## 10. baseline-models 当前共享策略

`baseline-models` 是长期基准模型分支，只接入和维护基准模型；不得将该分支整体
合入 `main`。已采用 `controlled_nonstrict` reproducibility：全局 strict CUDA
algorithms 已关闭，固定 Python/NumPy/PyTorch/CUDA seed、DataLoader generator 和
数据顺序保留，`cudnn.deterministic=true`、`cudnn.benchmark=false` 保留，TF32 已
关闭。父调度器在 worker 启动前注入 `PYTHONHASHSEED` 和
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。
evaluate-only 也复用同一现有 worker 调度路径，启动前注入相同环境变量，避免父
进程直接执行 seed 初始化。

Repeatability 现在对 resolved/model config、初始权重、数据 split、batch/update
数量、epoch、环境、DataLoader 顺序、prediction key 和 horizon 集合执行 exact
检查；loss、validation/test metrics 和 predictions 使用报告中的固定 atol/rtol。
Run B 按模型名称反转顺序，结果仍按模型名称比较。

当前基准模型接入状态为 LSTM、Crossformer、STCN、DLinear、TSMixer、SegRNN、
iTransformer、TimesNet、TimeMixer、Transformer、LightTS、TiDE、FreTS、FiLM、
Informer、Autoformer、STID、DCRNN、AGCRN、GraphWaveNet、GRUGCN、RNNEncGCNDec、
PureGCN、PatchTST、Nonstationary Transformer、FEDformer 和 EvolveGCN。第二批软件接入完成；Informer(distil=true) 因官方 BatchNorm
按公共 planner 使用 full_nodes；STID 保留 node identity，使用
full_spatiotemporal；其余兼容模型按 NodeShared planner；本轮 GPU validation
已在 target GPU 上按固定 acceptance sequence 执行。16 个 Time-Series-Library
adapters 中，14 个不含原生 BatchNorm 的 adapter 使用公共 NodeShared
microbatch executor；Informer(distil=true) 与 PatchTST 因官方 BatchNorm 使用
full_nodes。LSTM、Crossformer 作为独立 NodeShared baseline，不计入上述 16 个
adapter 统计；STCN 和 STID 保持 full_spatiotemporal。第三批 TSL 图模型软件接入完成；
DCRNN/GraphWaveNet/GRUGCN/RNNEncGCNDec 复用公共 k=5 physical graph；AGCRN
保留官方 adaptive learned graph；5 个模型均使用 full_spatiotemporal，不参与
NodeShared node chunk。默认
graph checkpoint compatibility now follows each built model's
`uses_public_graph_resource` capability, so shared physical-graph consumers
reject changed public graph configuration while AGCRN remains compatible。
`runtime.node_shared_chunk_size=32`；
public batch=32 和 `runtime.node_shared_chunk_size=32` 均未改变；
sample batch 与 node chunk 分离，134 个节点自然分为 `32/32/32/32/6`。
STCN 保持 `[B,L,N,C]` 的 full spatiotemporal 执行；NodeShared 若检测到原生
BatchNorm 则保留完整节点执行，不改为 LayerNorm。正式 compatible NodeShared
发生 OOM 时只能统一下调公共 chunk，禁止按模型救援。当前主机为 NVIDIA
GeForce RTX 4070 Ti SUPER、CUDA 12.4；外部 `Time-Series-Library` 源码仅作为
只读运行时依赖，不纳入仓库。本轮正式 Full 已完成 LSTM 与 Crossformer；其余
模型按用户要求暂缓，不将此分支用于 RA-DS-PFD 开发。

### NodeShared software closeout

NodeShared software closeout complete：`runtime.node_shared_chunk_size` 的唯一
运行值来源是 `configs/experiment.yaml`，production code 不提供独立 fallback；
Resume compatibility 只约束当前 `ExecutionPlan` 实际使用
`node_shared_microbatch` 的模型，`full_nodes`/`full_spatiotemporal` 不因 chunk
值变化失效。Formal shape validation 已改为 resolved AMP semantics。

本轮 target GPU 已完成全部 Smoke、formal shape 和 repeatability gate；real
CUDA/cuDNN LSTM recurrent-dropout two-pass replay 已通过现有真实 CUDA
repeatability 路径验证。pre-node_shared_microbatch 产生的
baseline runtime results/logs 已明确作废，不支持复用，且未新增 legacy result
compatibility layer。`capture_prediction` 的多 ForecastBatch 重组问题已修复；
real LSTM recurrent-dropout CPU repeatability regression 和 actual Crossformer
chunk-equivalence regression 已加入。`node_shared_chunk_size` 保持 `32`，未
下调到 `16` 或 `8`。

第四批软件接入完成：PureGCN 是基于 TSL GraphConv 的项目定义 spatial-only
结构控制 baseline，使用公共 k=5 physical graph；PatchTST 保留官方 BatchNorm，
因此使用 full_nodes；Nonstationary Transformer 与 FEDformer 使用 NodeShared
planner；EvolveGCN 使用公共 physical graph 和 full_spatiotemporal，保持官方
evolving-GCN semantics。GPU formal validation 状态按实际执行记录。

### GPU validation closeout (2026-08-11)

- GPU：NVIDIA GeForce RTX 4070 Ti SUPER；PyTorch `2.5.1+cu124`；CUDA `12.4`；总显存 `16375.5 MB`。运行时 resolve/preflight：`env_tslib=18`、`env_tsl=9`，全部 PASS。
- 公共配置：graph `k=5`；train/validation/test batch `32`；`node_shared_chunk_size=32`；AMP 为 YAML resolved `true/float16`；seed `2026`；nodes `134`；lookback `144`；horizon `10`。
- 原始 GPU gate 中 `27` 个模型完成了 Smoke / FORMAL_DEFAULT_SHAPE；只有 dual-gate PASS 模型进入 repeatability。原始 gate 为 Smoke `17 PASS / 10 FAIL`、FORMAL_DEFAULT_SHAPE `17 PASS / 10 FAIL`，随后 repeatability `15 PASS / 2 FAIL`。
- 本轮 `baseline_gpu_fix_smoke_seed2026`：`10/10 PASS`、`0 OOM`；`baseline_gpu_fix_formal_shape_seed2026`：`10/10 PASS`，真实 shape 均为 `[32,144,134,16] -> [32,134,10]`，profile 均为 `FORMAL_DEFAULT_SHAPE`。本轮通过模型：`autoformer fedformer film timesnet frets graphwavenet informer nonstationary_transformer timemixer transformer`。
- 本轮 `baseline_gpu_fix_repeatability_seed2026`：`10 PASS / 2 FAIL`。PASS：`autoformer fedformer film timesnet frets graphwavenet informer nonstationary_transformer timemixer transformer`；FAIL：`evolvegcn`（predictions，max abs `6.42791748046875`）、`puregcn`（predictions，max abs `1.750457763671875`）。未放宽既有 tolerance，未发现 selection tie；GraphWaveNet 为 fixed tolerance 内的 `NUMERICAL` PASS。
- Autoformer/FEDformer/FiLM/TimesNet 保持 public AMP=`true`，其 read-only upstream CUDA FFT forecast path 在 project-owned FP32 compatibility island 内执行，以避开 CUDA FP16 cuFFT 对 resolved non-power-of-two transform length 的限制。
- missing-gradient validator 现在只要求至少一个 participating finite gradient，并拒绝全空梯度或 NaN/Inf；本轮六个模型均有真实 finite gradients，部分 `grad=None` 仅来自当前 forecasting forward 未使用的 upstream 参数。
- Formal Full：`lstm=PASS`、`crossformer=PASS`；H3/H6/H10 artifacts、metrics、checkpoint reload 和 performance 均完整且有效。其余 13 个模型为 `NOT RUN — user deferred after Crossformer`；STCN 曾因 scheduler race 启动，随后在 Full 完成前终止，不计为 PASS。
- Formal shape peak allocated GPU memory（MB）：`Informer=4040.7`（full_nodes）、`PatchTST=9541.3`（full_nodes）、`STCN=4568.4`（full_spatiotemporal）、`STID=136.1`（full_spatiotemporal）、`DCRNN=1546.2`、`AGCRN=3879.7`、`GraphWaveNet=6821.0`、`GRUGCN=2151.6`、`RNNEncGCNDec=2396.8`、`PureGCN=161.2`、`EvolveGCN=437.3`。Full peak：`LSTM=731.5`、`Crossformer=2593.8`。
- 本轮无 OOM；本轮未运行任何 Formal Full。剩余 blocker 为 EvolveGCN/PureGCN repeatability FAIL，以及 13 个 deferred Full。

### PureGCN / EvolveGCN repeatability diagnosis (2026-08-11)

- 两模型均使用正式 `env_tsl`、当前 YAML、公共 `k=5` graph、`[32,144,134,16]` input、seed `2026`；诊断重新核对了 initial state hash，分别为 PureGCN `71a12a369c4b5c7699361db0de39131d809c8d42395e71760cb302314d32daec`、EvolveGCN `936fe4895a0f6412565e7490bd9387d24ae33cfb38be6bfd1052e98b8aa427b6`。
- PureGCN：CPU FP32 repeated eval forward exact；CUDA FP32 max abs `2.384185791015625e-7`、AMP max abs `4.8828125e-4`。stage probe 首个非 exact 点是 `graph_conv1.aggr_module`；CUDA FP32 8/8 gradient tensors diverge，首个参数 `graph_conv1.bias`，max grad abs `2.7939677238464355e-9`。
- EvolveGCN：CPU FP32 repeated eval forward exact；CUDA FP32 max abs `2.682209014892578e-7`、AMP max abs `4.8828125e-4`。stage probe 首个非 exact 点是 `upstream.encoder.rnn_cells.0.aggr_module` 的第一个时间步；CUDA FP32 11/11 gradient tensors diverge，首个参数 `upstream.input_encoder.0.weight`，max grad abs `7.450580596923828e-9`。
- 两模型的 forward 均未 mutation state 或 cache（`cached=false`，cache 保持 `None`）。用完全相同 gradient 隔离 Adam step 后，model state 和 optimizer state 均 exact；正式 CUDA AMP 两-update path 在 `update 0` 已出现非零 state/prediction/loss 差异，故不是 optimizer 首因。
- 只读源码链为：TSL `GraphConv` / `EvolveGCNHCell` → PyG `MessagePassing(aggr="add")` → `SumAggregation` → `torch_geometric.utils._scatter.scatter` → CUDA `Tensor.scatter_add_`；`norm=mean` 的 TSL degree normalization 也使用 `Tensor.scatter_add_`。`torch.use_deterministic_algorithms(True)` 在本次两模型 forward/backward 未抛出错误，因此低层 CUDA kernel 归因记为 STRONGLY SUPPORTED，而非 strict-probe CONFIRMED。
- 结论：failure layer = CUDA graph forward aggregation（CONFIRMED）；项目 RNG/input/state/cache/optimizer bug = 未发现；根因为 upstream CUDA graph reduction 的受控数值非确定性（STRONGLY SUPPORTED）。Repeatability 继续为 PureGCN `FAIL`、EvolveGCN `FAIL`，tolerance 和正式协议不变；本轮不新增永久测试、不运行 Formal Full。`GPU acceptance closeout` 保持 `IN PROGRESS`。

### Baseline software integration status

Baseline CPU software integration: PASS。
GPU acceptance closeout: IN PROGRESS。

`env_tslib` full CPU regression：PASS，0 failures，0 errors（240 passed，22 expected skips）。
`env_tsl` full CPU regression：PASS，0 failures，0 errors（210 passed，52 expected skips）。

GPU Smoke / FORMAL_DEFAULT_SHAPE：本轮 10 个受影响模型全部 PASS；GPU Repeatability：`10 PASS / 2 FAIL`。Formal Full：`PARTIAL / USER DEFERRED`，本轮固定在 Repeatability 结束。
