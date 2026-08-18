# PhDPaper3 当前交接

## 1. 项目与当前 main

PhDPaper3 是可复现的 SDWPF 时间序列预测科研实验工程。`main` 是长期承载
自定义模型的分支，当前维护范围包括共享路径上的 LSTM、Crossformer、STCN，
以及 RA-DS-PFD Crossformer 的 P1/P2、冻结的 R0-R7 suite 和 P3-A
Global Top-K Auto-PFD Foundation。公共实验协议、模型数学实现、R0-R7
variant 定义、P3 propagation seam、GPU 策略和已有结果都属于当前兼容边界。

正式训练需要 `dataset/` 中两个协议命名的 parquet 文件。没有对应数据或目标
CUDA 环境时，不把 shape、smoke、CPU 回归或单步 GPU gate 解释为正式 Full。

## 2. 当前公共架构

公共训练、评估、检查、repeatability 和 summarize 执行链是：

```text
scripts/run.py
  -> orchestrator
  -> isolated worker
  -> shared Trainer / Evaluator
```

`scripts/run_ra_ds_pfd_r0_r7.py` 只是 R0-R7 的 variant-selection wrapper：它从
`configs/experiments/ra_ds_pfd_r0_r7.yaml` 解析 variant，生成临时 `runtime/model`
YAML，最终调用 `scripts/run.py train`。它没有自己的 Trainer、Evaluator、数据
流程、公共实验参数或结果系统；临时 YAML 不写入 `configs/models/`，结果目录内
持久化的 `model_config.yaml` 可用于重放。

父调度器按模型 YAML 选择运行环境，为每个模型启动独立 worker；公共配置来自
`configs/experiment.yaml`，模型 YAML 只负责 `runtime` 和模型结构参数。实际
DataLoader worker seeding 的唯一 owner 是 `src/data/dataloader.py::_worker_init()`；
`src/engine/reproducibility.py` 负责全局 controlled-nonstrict seed、RNG 状态和
loader seed，不保留第二个 worker seeding 实现。

长期公共状态为：

- 批次：`results/_runs/<run-id>/status.json`；
- 模型：`results/<model>/<run-id>/run_info.json`；
- 独立检查：`results/_checks/<check-id>/`。

顶层状态使用 schema v2 的小型状态词汇，失败细节放在 `error` 和 `details`；
读取器兼容 schema v1。Resume、完成判断和 summarize 不依赖 worker 内部的
`validation_status.json`。

scripts/run_ra_ds_pfd_p3.py 是 P3-A 的同类 thin wrapper：它从
configs/experiments/ra_ds_pfd_p3.yaml 解析 frozen R2，生成临时 model YAML，
最终调用 scripts/run.py train；它不拥有第二套 Trainer、Evaluator、数据流程
或结果系统。本批只执行过 CPU dry-run，未启动训练。

## 3. RA-DS-PFD 当前状态

P1 与 P2 均为 `PASS_WITH_NOTES`：

- P1 的 `spatial_disabled=true` concrete adapter 使用公共 `NodeShared` 微批，
  134 个节点按 `32/32/32/32/6` 执行；公共 chunk 唯一来源是
  `configs/experiment.yaml` 中的 `runtime.node_shared_chunk_size=32`。
- P2 的 enabled 路径使用 `pfd_mode=pfd0`、`full_spatiotemporal` 和模型局部的
  relation resource；relation NPZ 只包含 `node_ids`、`edge_index`、
  `edge_static_features`、`edge_feature_names`，结构、节点顺序、边方向、排序、
  dtype、shape、finite 和字段名均在构造时校验。
- 当前公共 YAML 保持 legacy disabled 路径；仓库不提交正式 134 节点 enabled P2
  relation artifact。P2 opt-in 使用合成小 fixture，正式训练仍未执行。

R0-R7 的唯一 matrix source 是
`configs/experiments/ra_ds_pfd_r0_r7.yaml`，resolver 是
`src/models/ra_ds_pfd_crossformer/r0_r7_suite.py`。R0 保持 canonical P1 identity；
R1 是 current P2 mapping；R2 是 old endpoint mapping；R3/R4/R5/R6 分别只改变
Query、Propagation Encoder、Turbine Embedding、Bias Scaling；R7 只改变 Query 与
Propagation Encoder。R0 使用 `node_shared_microbatch`，R1-R7 使用
`full_spatiotemporal`，R1-R7 的共同 `spatial_edge_chunk_size` 为 `512`。

R0-R7 matrix/resolver、公共 runner、CPU foundation、suite smoke、持久化 model
config 重放和 result-directory relocation 已通过验收。GPU Stage A 与 R4/R5 的
20-step execution foundation gate 也已通过；这些是 memory/execution evidence，
不是 R0-R7 Formal Full。

P3-A 从 frozen R2 派生，模型仍使用 ra_ds_pfd_crossformer 和
full_spatiotemporal：

- Candidate Bank 固定 13 个 scalar base features，各自生成 level 和 diff1
  两个 history-only 候选，共 26 个；Wdir、Ndir、Wdir_w 因当前 model-visible
  x 已标准化且公共数据接口不变，暂不进入候选空间。
- Selector 是全局共享的 [26] learnable logits 与 differentiable softmax
  score vector，top_k=2 只承担 ranking/readout 和后续 freeze contract，
  不代表本批已有科学 Top-2 结果。
- 每个候选只有一个小型 Linear(seg_len -> d_model, bias=False) 和一个
  identity embedding；Scale0/Scale1 Cross-Time、Scale1 merging 和 position
  embedding 均在 candidate 轴上共享。
- Self View 继续接收完整多变量历史；P3 selector 只生成 propagation K/V。
  relation resource、R2 的四个 architecture axes、relation bias 和 graph
  均不变。P3 提供 model-owned read-only selection report，不写 score 文件。

## 4. 当前已验证结果

P3-A focused tests：18 passed。冻结 R0-R7 suite/runner regression：
21 passed。所有 RA-DS-PFD 相关测试：111 passed。当前 CPU-only regression：
276 passed, 4 skipped；skip 是既有 CUDA 或正式 tsl 环境条件，不是失败。
python scripts\generate_command_reference.py --check 通过；P3 runner CPU
dry-run 通过且未写正式结果。

R0-R7 的当前明确状态是：

```text
R0-R7 Formal Full = NOT RUN
```

同样尚未运行 multi-seed 和 formal test-set comparison；已有 smoke、shape、
Stage A、P3 dry-run 或 CPU foundation 均不得改写为 Formal Full。

## 5. 尚未实现与下一步

P3-A Discovery 尚未运行，未导出正式 feature scores、rank 或 Top-2，也未使用
test set 做选择。P3 Formal Full、multi-seed、PredictionTop2、RandomTop2、
AllPropagation、physics loss、dynamic graph 和 variable-specific temporal
response 均未实现或未运行。

Next：P3-B Discovery ——在 validation-best checkpoint 上训练并导出真实
propagation selection scores / rank / Top-2，不使用 test set 进行选择。

## 6. 当前兼容约束与不可踩的坑

- 不修改公共 batch、lookback、horizon、seed、AMP、loss、optimizer、graph `k`、
  `node_shared_chunk_size`、`spatial_edge_chunk_size`、R0-R7 axes、Trainer、
  Evaluator、loss accumulation、checkpoint compatibility、repeatability tolerance、
  status schema 或 result layout。
- 公共图保持 `adjacency[source,target]`、`edge_index[0]=source`、
  `edge_index[1]=target`、`k=5`；NodeShared 与 full-spatiotemporal 的边界不能互换。
- Resume 比较 resolved model-config 内容而不是临时 YAML 路径；不得把不同 Rk 的
  checkpoint 互相恢复，也不得破坏 result-directory-relative 的重放路径。
- 不用 batch-1 shape、smoke、CPU pytest、单步 GPU gate 或 allocator peak 冒充
  默认 formal shape、正式 GPU 训练或 Formal Full；不把 NaN 指标填成 0。
- 不重新引入 StudySpec、ModelSpec、manifest、certificate、readiness protocol、
  模型专属 Trainer/Evaluator 或新的 Markdown 文档；不要手工编辑生成的
  docs/COMMAND_REFERENCE.md。
- P3-A 只允许从 frozen R2 派生；pfd_mode=pfd0 的 R0-R7 路径继续使用
  self.pfd0，P3 使用 self.p3_propagation，不得 mask Self 输入或改变关系图。
- Candidate Bank 只读取 ModelInput.x 和 DataInfoView.feature_columns，不得读取
  target、mask、future weather 或预测窗口；GPU validation 当前为
  PENDING_GPU，因为 GPU 被其他进程占用，本批没有启动 CUDA workload。
- 不提交 dataset、results、logs、checkpoint 或外部库，不强制推送，不批量递归删除。
