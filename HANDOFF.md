# PhDPaper3 当前交接

## 1. 当前 baseline scope

`baseline-models` 当前正式 scope 为 26 个模型：

```text
agcrn autoformer crossformer dcrnn densegcn dlinear fedformer film frets
graphwavenet grugcn informer itransformer lightts lstm
nonstationary_transformer patchtst rnnencgcndec segrnn stcn stid tide
timemixer timesnet transformer tsmixer
```

模型配置由 `configs/models/*.yaml` 动态发现；当前环境分布为
`env_tslib=19`、`env_tsl=7`。`densegcn` 已存在，`puregcn` 和 `evolvegcn`
不再属于当前 model choices，也没有兼容 alias。

## 2. 项目与稳定架构

PhDPaper3 是可复现的 SDWPF 时间序列预测科研实验工程。统一入口是
`scripts/run.py`；DataLoader、Trainer、Evaluator、metrics、loss、checkpoint、
scheduler、repeatability、环境调度、结果聚合、节点顺序和公共图资源均由共享路径
提供。模型只实现 `build_model(model_config, data_info)`，消费 `ModelInput` 并返回
`[B,N,H]`。

父调度器按模型 YAML 选择环境并启动独立 worker。Node Shared 模型由公共执行器按
`runtime.node_shared_chunk_size=32` 处理；真正的跨节点模型保持
`full_spatiotemporal`。公共状态写入 `results/_runs/<run-id>/status.json`、
`results/<model>/<run-id>/run_info.json` 和 `results/_checks/<check-id>/status.json`。

当前冻结公共协议：

```text
nodes=134, lookback=144, horizon=10, eval_horizons=3/6/10
physical_knn k=5, binary, symmetrize=true, self_loops=false
train/validation/test batch=32, epochs=20, seed=2026
AMP=true/float16
loss=masked_score_aligned_hybrid
optimizer=Adam, lr=0.001
reproducibility_mode=controlled_nonstrict
torch.use_deterministic_algorithms(False)
```

## 3. DenseGCN replacement closeout（2026-08-11）

DenseGCN 来自旧仓库 `benchmark_models/src/models/pure_gcn.py` 的
`PureGCNForecastModel` / `WeightedGraphConv`。迁移保留的结构为：历史与特征展平、
input projection、GELU、两层 dense `einsum("ij,bjd->bid", A, X)` 后接 Linear、
Dropout、residual、LayerNorm，以及线性 readout。参数量保持 `156746`。

旧项目的 Gaussian distance weighting、self-loop 和 row normalization 属于旧图
预处理，没有迁移。DenseGCN 使用当前 `src/resources/graph.py` 生成的公共 `k=5`
binary graph，在模型构造阶段将 `edge_index + edge_weight` 一次性还原为节点顺序不变
的 `[134,134]` dense adjacency，并注册为 persistent buffer。forward 不使用 TSL
`GraphConv`、PyG `MessagePassing`、`torch_scatter` 或 `scatter_add_`。运行环境为
`env_tslib`，执行模式为 `full_spatiotemporal`。

固定 acceptance sequence 全部 PASS：

- `densegcn_interface_small_seed2026`：`INTERFACE_SMALL PASS`。
- environment preflight：`env_tslib PASS`。
- model/data preflight：`PASS`。
- `densegcn_gpu_smoke_seed2026`：`RESOLVED_SHAPE PASS`、Smoke `PASS`、`0 OOM`。
- `densegcn_formal_shape_seed2026`：`FORMAL_DEFAULT_SHAPE PASS`；YAML-default
  batch `32`，`[32,144,134,16] -> [32,134,10]`，AMP `true/float16`，peak GPU
  allocated `129.5224609375 MB`。
- Smoke 短训练 peak GPU allocated `171.10400390625 MB`。
- `densegcn_repeatability_seed2026`：`EXACT PASS`；A/B 独立 worker PID
  `28704/22244`，initial weight hash 均为
  `9c1fad9e76333528ec883e5d96f1cd0d8457c4c8749885dd396d23e6e542992d`，
  batch order exact，first-step loss 均为 `0.9624511003494263`，train history exact
  （epoch 1 train loss `0.9743914008140564`、monitor `55.16749385179107`、
  learning rate `0.001`、2 updates），prediction max abs/rel 均为 `0.0`，
  validation/test metric differences 均为 `0.0`，best epoch 均为 `1`，
  `selection_tie=false`。

Repeatability 使用正式 `controlled_nonstrict`、
`deterministic_algorithms=False` 和 AMP float16；prediction tolerance
`atol/rtol=0.005/0.005`、metric tolerance `0.0002/0.0002` 均未修改。

## 4. PureGCN / EvolveGCN 历史结论

PureGCN 和 EvolveGCN 已从当前正式 baseline scope 删除。它们并非无法运行：两者
都曾通过 Smoke 和 `FORMAL_DEFAULT_SHAPE`；删除原因仅是其 upstream sparse CUDA
graph aggregation 路径在固定 `controlled_nonstrict` tolerance 下持续
Repeatability FAIL。诊断链为 TSL `GraphConv` / `EvolveGCNHCell` → PyG
`MessagePassing` / `SumAggregation` → CUDA `Tensor.scatter_add_`。历史 results、
repeatability artifacts 和诊断凭据保留在磁盘中，没有删除。

## 5. 当前验收状态

```text
Baseline scope: 26 models
GPU Smoke / FORMAL_DEFAULT_SHAPE / Repeatability: 26/26 PASS
pre-Full eligible: 26/26

Formal Full completed:
- LSTM
- Crossformer

Formal Full remaining: 24 models
```

本轮没有运行或恢复任何 Formal Full；DenseGCN 的 `FORMAL FULL NOT RUN`。

scope replacement 后 CPU 回归：

- `env_tslib`：`245 passed, 19 expected skips`，0 failures，0 errors。
- `env_tsl`：`212 passed, 52 expected skips`，0 failures，0 errors（2 warnings）。

## 6. 维护与接手规则

接手新模型时先读 `MODEL_INTEGRATION_INDEX.md`。不要重新实现共享训练/评估路径，
不要修改冻结公共协议来救援单个模型，不要让 Time-Series-Library 的 `models`
覆盖项目 `src/models`，也不要提交 dataset、results、logs、checkpoint 或外部库。

文档保持少而明确：README 只记录稳定用户入口；
`MODEL_INTEGRATION_INDEX.md` 只在接入契约变化时修改；
`docs/COMMAND_REFERENCE.md` 只在 `command_schema.py` 变化时重新生成；本轮后二者均
无需修改。下一阶段如果继续正式实验，只处理剩余 24 个 Formal Full。
