# PhDPaper3 当前交接

## 1. 项目与当前 main

PhDPaper3 是可复现的 SDWPF 时间序列预测科研实验工程。`main` 是长期承载
自定义模型的分支，当前维护范围包括共享路径上的 LSTM、Crossformer、STCN，
以及 RA-DS-PFD Crossformer 的 P1/P2、冻结的 R0-R7 suite 和 P3-A
Global Top-K Auto-PFD Foundation；P3-A2.1 architecture closure 已完成。公共实验协议、
模型数学实现、R0-R7
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
或结果系统。本轮 B0 复用同一 resolver 生成临时 model YAML，再调用现有
scripts/run.py check --full-shape；未启动 epoch training 或 Discovery。

scripts/run_ra_ds_pfd_p3_b1.py 是 P3-B1 的 thin wrapper：它从
configs/experiments/ra_ds_pfd_p3_b1.yaml 解析 canonical P3，生成 B1-LD/B1-L
临时 model YAML，调用同一 scripts/run.py train，并在训练成功后从同一 run
目录的 best.pt 生成 p3_selection_best.json。它不拥有第二套 Trainer、
Evaluator、数据流程或结果系统；dry-run 只解析配置和打印公共 train command。

scripts/run_ra_ds_pfd_p3_b2.py 是 P3-B2 的 thin wrapper：它从
configs/experiments/ra_ds_pfd_p3_b2.yaml 解析 canonical P3，生成 B2_K1、
B2_K2、B2_K3、B2_K4、B2_K6、B2_K8 的临时 model YAML，调用同一
scripts/run.py train，并在训练成功后复用 best.pt -> p3_selection_best.json
的 selection readout。每个 arm 只允许改变 p3.top_k；runner 不拥有第二套
Trainer、Evaluator、数据流程或结果系统。

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

P3-A2 architecture closure = COMPLETE。P3 继续从 frozen R2 派生，模型仍使用
ra_ds_pfd_crossformer 和 full_spatiotemporal：

- Candidate Bank 固定 13 个 scalar base features，各自生成 level 和 diff1
  两个 history-only 候选，默认共 26 个；Wdir、Ndir、Wdir_w 因当前
  model-visible x 已标准化且公共数据接口不变，暂不进入候选空间。
- Selector 是全局共享的一组 [M] learnable logits，使用
  entropy-regularized fixed-cardinality Top-K relaxation；它共享于所有
  turbines、samples、timestamps、edges 和两个尺度。默认 K=2，架构支持
  1 <= K <= M。forward 通过 no-grad scalar dual threshold bisection 求
  relaxed gate，backward 使用 fixed-cardinality constraint 的 analytic
  implicit differentiation；raw gate 的 sum 由 threshold solve 满足，传播仍使用
  relaxed gate 归一化后的 mixture weights。
- 当前 selector 参数为 model-owned `selector_temperature=0.1` 和
  `selector_bisection_iterations=64`，随 resolved model config 进入现有
  checkpoint compatibility。Hard Top-K 只用于 deterministic ranking/readout，
  不参与 canonical forward。
- 每个候选只有一个小型 Linear(seg_len -> d_model, bias=False) 和一个
  identity embedding；Scale0/Scale1 Cross-Time、Scale1 merging 和 position
  embedding 均在 candidate 轴上共享。
- Candidate temporal operator registry 已建立；当前注册且仅注册 `level`、
  `diff1`。默认 operator basis 为 `level + diff1`，validator 支持非空、无重复
  的当前注册 operator 子集，level-only 会动态生成 13 个候选。
- Self View 继续接收完整多变量历史；P3 selector 只生成 propagation K/V。
  relation resource、R2 的四个 architecture axes、relation bias 和 graph
  均不变。P3 propagation 的 candidate token、Scale0 Cross-Time 和 Scale1
  Cross-Time 均继承 frozen R2 的 `spatial_dropout`；Self/backbone 仍使用
  `dropout`。P3 提供 model-owned read-only selection report，不写 score 文件。

P3-B1 infrastructure = PASS。B1 resolver 只允许从 canonical P3 改变
`p3.candidate_transforms`：B1-LD 为 13 x {level,diff1}，M=26、K=2；B1-L
为 13 x {level}，M=13、K=2。两个 arm 的唯一实验变量是 candidate temporal
operator basis；top_k、selector settings、candidate_features、R2 axes、
relation resource、spatial edge chunk 和其他 model fields 均 fail closed。
B1 execution preparation 已覆盖完整解析链：B1 suite -> canonical P3 ->
R0-R7 -> `base_model_config.file` ->
`configs/models/ra_ds_pfd_crossformer.yaml` -> runtime/model YAML；两个临时
model YAML 的 runtime 均为 `environment: tslib`。selection readout 固定只使用
`best.pt`，并要求 checkpoint manifest 的 `model_config`（兼容已有
`model_config_identity`）与 run directory 的 `model_config.yaml` model mapping
结构一致；缺失或不一致时 fail closed，不写出 selection artifact。正式
Discovery 尚未运行，尚未冻结任何 operator basis。

P3-B2 infrastructure = PASS。B2 resolver 从 canonical P3 派生并 fail closed
锁定用户指定的 Level + Diff1 basis：13 x 2，M=26；K grid 为
`[1,2,3,4,6,8]`，唯一实验变量是 `p3.top_k`。candidate features、operator
basis、selector temperature/iterations、R2 architecture、relation、公共
训练参数和其他 model fields 均保持一致。每个未来真实 B2 run 都复用
`best.pt -> p3_selection_best.json`，并打印 selected propagation features、
score、rank、base-variable scores 和 operator scores。`p3_b2_k_selection.json`
只使用公共 validation checkpoint monitor；test set 永不参与 K selection，
缺少任一 K arm 时只能报告 INCOMPLETE，不能声明 winner。不同 K 的 normalized
score 只用于该 run 内的 readout，不直接解释为跨 K 的 feature importance。

## 4. 当前已验证结果

P3 selector focused tests：36 passed；B1 focused tests：13 passed；B2 focused
tests：23 passed；其中
gradcheck（K=1/2/3）3 项、central finite-difference（K=1/2/3）3 项、
constraint-gradient（K=1/2/3）3 项、translation-invariance、monotonic ranking、
K=M zero-gradient、best.pt-only selection、model-config provenance、B2 K grid、
validation-only aggregation 和 incomplete-grid gate 均通过。所有 P3 CPU 测试：
98 passed；所有 RA-DS-PFD 相关测试：191 passed。当前 CPU-only regression：
356 passed, 3 skipped；skip 是既有正式 tsl 环境条件，不是失败。唯一真实
CUDA/full-shape 测试 `tests/test_full_shape.py` 按本轮 GPU 禁令排除。
`python scripts\generate_command_reference.py --check` 通过；P3-B2 的六臂
CPU dry-run 通过，且未写正式结果。

P3-B0 GPU Feasibility = PASS。目标 GPU 为 NVIDIA GeForce RTX 4070 Ti SUPER
(16376 MiB)；公共 `check --full-shape` 使用 formal default shape、
`full_spatiotemporal` 和 resolved AMP float16，完成 forward/loss/backward：
输入 `[32,144,134,16]`，输出 `[32,134,10]`，M=26、K=2，输出、loss、selector
logits、candidate projections、shared Scale0/Scale1 temporal encoder 与
relation/spatial trainable gradients 均 finite，无 CUDA OOM。公共 allocator
记录为 peak allocated 18105.886 MB、peak reserved 18274.000 MB，GPU total
16375.500 MB。

R0-R7 的当前明确状态是：

```text
R0-R7 Formal Full = NOT RUN
```

同样尚未运行 multi-seed 和 formal test-set comparison；已有 smoke、shape、
Stage A、P3 dry-run 或 CPU foundation 均不得改写为 Formal Full。

## 5. P3-A2/P3-B1/P3-B2 状态与下一步

P3-A2.1 STATUS = PASS；P3-B0 STATUS = PASS；P3-B1 infrastructure STATUS = PASS。
P3-A2 formal architecture closure
已完成：Global K-configurable Selector 保持 entropy-regularized fixed-cardinality
relaxation，forward 为 scalar threshold numerical solve，backward 为 implicit
differentiation，且已通过 numerical gradient validation。默认 K=2，K configurable
1..M；operator registry 当前为 level、diff1；default candidate bank 为
level + diff1；selector remains propagation-only。

P3-B1 formal Discovery 尚未运行，未导出正式 feature scores、rank 或 Top-K，也
未使用 test set 做选择。P3-B2 infrastructure 已完成，但没有真实 K-selection
结果；Level+Diff1 是当前 B2 的 frozen default operator basis，不是 B1 的实验胜出
结论。P3 Formal Full、multi-seed、PredictionTopK、RandomTopK、
AllPropagation、physics loss、dynamic graph 和 variable-specific temporal
response 均未实现或未运行。R0-R7 仍冻结且未被本轮修改。

GPU validation for P3-B0 = PASS（formal default shape）；P3-B1 GPU smoke =
NOT RUN；B1-LD/B1-L Formal Discovery = NOT RUN；B1 operator decision =
NOT RUN。P3-B2 GPU smoke = NOT RUN；P3-B2 Formal K-selection = NOT RUN；
selected K* = NOT DECIDED。上述状态均因 GPU occupied，本轮只完成 CPU-only
infrastructure implementation。

Next：GPU 空闲后建议先运行 P3-B2 六臂 smoke；本轮没有启动该命令，也没有自动
启动六个 Formal Full：

```powershell
python scripts\run_ra_ds_pfd_p3_b2.py --all --run-id ra_ds_pfd_p3_b2_smoke_seed2026 --device cuda --smoke
```

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
  target、mask、future weather 或预测窗口；P3-B0 已通过公共 formal default
  forward/loss/backward gate；P3-B1 只完成 infrastructure 和 CPU 验收，未启动
  Discovery 或 Formal Full。
- P3 selection readout 只读取 `best.pt`；checkpoint manifest 中的 model config
  必须与 run directory 的 `model_config.yaml` 一致，不为此扩展公共 checkpoint
  compatibility 或增加 hash/certificate。
- 不提交 dataset、results、logs、checkpoint 或外部库，不强制推送，不批量递归删除。
