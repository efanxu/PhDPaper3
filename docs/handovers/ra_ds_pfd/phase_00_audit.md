# RA-DS-PFD Crossformer P0 审计快照

## 阶段目标与状态

本阶段只完成当前框架、当前 Crossformer Adapter、本机 upstream
Time-Series-Library Crossformer 和 `model_validation.zip` 旧原型的真实代码审计，
并冻结 P1 的插入点、shape、图方向、bias 语义、依赖和文件边界。没有实现
RA-DS-PFD、PFD0、Feature Bank、Selector 或任何生产模型代码。

```text
状态: PASS
completion_commit: SELF
```

本机 upstream 源码存在；旧压缩包存在并已隔离审计；动态 trace、canonical formal
shape forward/backward、focused tests、文档一致性检查和全量非数据测试均有真实结果。

## 开始状态

```text
开始分支: main
开始 HEAD: 2122dbaa3fe6df5b5c193dd49d2da2e765be7580
开始工作树: ?? model_validation.zip
远端: origin git@github.com:efanxu/PhDPaper3.git
仓库根目录: D:/PaperProject/PhDPaper3
```

`model_validation.zip` 是本任务范围内的用户输入，开始时已存在且未跟踪；本阶段
没有覆盖、清理、暂存或提交它。解压目录为仓库外的
`D:/PaperProject/PhDPaper3_old2/p0_model_validation_20260731/`。
审计完成后，该单个 archive 已移动到上述仓库外目录保存；SHA256 保持为
`5002F8BCD47657C0AE03A54A7BCCFF28DF47702CD394AACEB4D8F09CDBFED248`，未暂存或提交。

## 读取的文件

```text
MODEL_INTEGRATION_INDEX.md
HANDOFF.md
configs/experiment.yaml
configs/models/crossformer.yaml
configs/environments.yaml
src/cli/command_schema.py
src/runtime/config.py
src/data/loader.py
src/data/split.py
src/data/normalization.py
src/data/window.py
src/data/dataset.py
src/data/dataloader.py
src/engine/reproducibility.py
src/engine/checkpoint.py
src/engine/trainer.py
src/engine/evaluator.py
src/runtime/performance.py
src/models/base.py
src/models/loader.py
src/cli/orchestrator.py
src/runtime/environments.py
src/integrations/time_series_library.py
src/models/crossformer/model.py
src/models/stcn/model.py
src/resources/graph.py
tests/test_crossformer_model.py
tests/test_time_series_library_isolation.py
tests/test_graph_resources.py
scripts/run.py
```

开始时不存在 `docs/handovers/ra_ds_pfd/`、`PHASE_INDEX.md` 和本快照。

## upstream source identity

```text
source_root: D:/PaperProject/PhDPaper3/Time-Series-Library
source revision: UNKNOWN
source Git worktree: 不存在独立 .git；父仓库 .gitignore 第 17 行忽略整个目录
parent Git tracked file count under Time-Series-Library: 0
```

本机实际 upstream 文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `models/Crossformer.py` | `F5892F70EB3FC320F69FA0A9B2019B68537CC5A5E5AA2EC544C15127B19E36F0` |
| `layers/Crossformer_EncDec.py` | `EE2CC5C8D3A44F4DE67CE986E6B442488826A20FA4D8F87991941A75587936A1` |
| `layers/Embed.py` | `17E7C3577324C41A0DA427A199C955B782FDE905AABB1F7CBC3C4E15EBD4AE35` |
| `layers/SelfAttention_Family.py` | `6F6592AED0753E342FC04C01DB70528C2EB191EEB067CD5FD4C8D348FDDCB794` |
| `models/PatchTST.py` | `29835D4FEDDDD3CBEE26CC60D6A54C04513F5A9CA32E4A0D08DE2C9E059084E5` |

`configs/environments.yaml`、`src/integrations/time_series_library.py` 和当前
Adapter 共同确认：Crossformer 运行时为 `tslib`/`env_tslib`，source root 为上述
目录；controlled loader 临时暴露 upstream `layers`/`models` 别名，加载后恢复项目
自己的 `models` 包。

## 旧 model_validation.zip

```text
archive: D:/PaperProject/PhDPaper3/model_validation.zip
SHA256: 5002F8BCD47657C0AE03A54A7BCCFF28DF47702CD394AACEB4D8F09CDBFED248
zip entry count: 1360
unsafe path entry count: 0
extract root: D:/PaperProject/PhDPaper3_old2/p0_model_validation_20260731
```

顶层目录为 `checkpoints/`, `checkpoints_smoke/`, `configs/`, `diagnostics/`,
`graphs/`, `reports/`, `results/`, `results_smoke/`, `src/`。只读取了旧
`src/unified_rel_crossformer` 的模型、图、attention、segment 和测试源码，没有运行
旧 runner、训练脚本或归档脚本，也没有复制旧目录到当前仓库。

旧关键源码 hash：

| 文件 | SHA256 |
| --- | --- |
| `src/unified_rel_crossformer/model.py` | `0A50F3586465DD35AF0AE7DD5216B73035C0DE217773E70E7897D0EE22D77DCB` |
| `src/unified_rel_crossformer/graph_union.py` | `FC73ED41F7B83A80760521B9F07B8C85892B784DC3B588B496FABEAB758E1ABD` |
| `src/unified_rel_crossformer/sparse_relation_attention.py` | `DBEE82BCDD1853230FCC04E0BE1186C3423EBA45E6A8E156D8C9C830226965B3` |

旧 union manifest 的真实 edge count：semantic only `1340`，distance only `670`，
semantic+distance 去重后 `1536`，both-edge `474`。这是旧 artifact 数值，不是当前
公共图的 formal `E`。

## 当前 Crossformer Adapter 审计

证据文件：`src/models/crossformer/model.py`。

| 项目 | 真实结论 |
| --- | --- |
| `build_model` 字段 | `_CONFIG_FIELDS` 只接受 `d_model`, `n_heads`, `d_ff`, `e_layers`, `dropout`, `factor`；`_validate_config` 检查未知、缺失、正值、整除和 dropout。 |
| 使用的 `DataInfoView` 字段 | `num_nodes`, `num_features`, `lookback`, `max_pred_len`, `feature_columns`, `input_power_column`, `input_power_index`, `project_root`；不使用 `node_ids`/`graph_config`。 |
| 输入布局 | `ModelInput.x` 必须是 `[B,L,N,C]`；可选 time/node/adjacency/static 输入非空会被拒绝。 |
| 节点展开 | `x.permute(0,2,1,3).reshape(B*N,L,C)`，所有节点共享一个 upstream 实例。 |
| upstream 输入/输出 | `[B*N,L,C]` 输入；检查 `[B*N,H,C]` 输出。long-term forward 内部先产生 padded output，再裁剪最后 `H`。 |
| 目标通道 | 按名称解析得到的 `data_info.input_power_index` 取 `Patv_clean_for_input`，不是旧硬编码；恢复为 `[B,N,H]`。 |
| 不适合作为空间 Backbone 的原因 | Adapter 只包住完整 `upstream.forward`，不暴露 encoder token、单层 Cross-Time、单层 Cross-Dimension、Segment Merging 或 decoder 多尺度列表，不能在两者之间插入 RA。 |

现有 `tests/test_crossformer_model.py` 已覆盖配置拒绝、shape、backward、功率通道、
Node Shared 参数量、节点置换/相同历史、非 finite 输出和缺失 upstream source；未覆盖
内部 token layout、空间关闭等价、边方向、target softmax、Static Edge/Relation Bias
和正式 RA 显存。

## upstream Crossformer 真实 forward trace

### 静态链路

```text
models/Crossformer.py:Model.forward
  -> Model.forecast
  -> layers/Embed.py:PatchEmbedding
  -> enc_pos_embedding + pre_norm
  -> layers/Crossformer_EncDec.py:Encoder
  -> encode_blocks[0]: scale_block(win_size=1)
       -> TwoStageAttentionLayer.time_attention (Cross-Time)
       -> dim_sender / dim_receiver (Cross-Dimension via router)
  -> encode_blocks[1]: scale_block(win_size=2)
       -> SegMerging
       -> TwoStageAttentionLayer.time_attention
       -> dim_sender / dim_receiver
  -> Encoder list [input, block0 output, block1 output]
  -> Decoder 的 e_layers+1 个 DecoderLayer
  -> dec_out[:, -pred_len:, :]
```

`Crossformer.py:Model.__init__` 实际 hardcode `seg_len=12`, `win_size=2`，创建
`PatchEmbedding`、encoder、decoder position embedding 和 `e_layers+1` decoder layers。
`PatchEmbedding.forward` 是右侧 `ReplicationPad1d`、非重叠 `unfold(size=12,step=12)`、
线性 value projection 加 positional embedding。

`SelfAttention_Family.py:TwoStageAttentionLayer.forward` 的 canonical 顺序为：

```text
time_in -> time_attention -> residual/dropout -> norm1
         -> MLP1 residual/dropout -> norm2
dim_send -> dim_sender(router) -> dim_receiver -> residual/dropout -> norm3
         -> MLP2 residual/dropout -> norm4 -> final_out
```

源码中 `time_in` 为 `(b*ts_d, S, D)`，`dim_send` 为 `(b*S, C, D)`。因此整个
Cross-Time/Cross-Dimension 是一个 atomic layer，outer hook 不能提供所需插点。

### 动态 trace

使用固定 `torch.manual_seed(20260731)`、`env_tslib python -`、小形状
`B=1,L=24,C=4,H=10,D=16,heads=4,d_ff=32,e_layers=2,factor=2`，通过现有
`load_time_series_library_model_class` 加载；forward hooks 观察 embedding、两个
scale block、time/dim attention、encoder 和 decoder。探针仅由 PowerShell here-string
经 stdin 执行，没有写入仓库。

关键真实输出：

```text
PatchEmbedding input/output: [1,4,24] -> [4,2,16]
pre_norm: [1,4,2,16] -> [1,4,2,16]
block0: [1,4,2,16] -> [1,4,2,16]
block0 time: [4,2,16] -> [4,2,16]
block0 dim sender: router [2,2,16], dim [2,4,16]
block1 merge: [1,4,2,16] -> [1,4,1,16]
block1 time: [4,1,16] -> [4,1,16]
block1 dim sender: router [1,2,16], dim [1,4,16]
encoder list: [[1,4,2,16], [1,4,2,16], [1,4,1,16]]
decoder layer prediction: [1,4,12] each
decoder output: [1,12,4]
final output: [1,10,4]
```

输出、输入梯度和参数梯度均 finite；参数量 `34580`。动态结果与静态代码一致。

## token shape 表（formal default）

当前真实配置：`B=32,L=144,N=134,C=16,H=10,D=64,heads=4,e_layers=2,seg_len=12,
win_size=2`。upstream batch `B_u=B*N=4288`，`head_dim=16`。

| 阶段 | shape | 语义 |
| --- | --- | --- |
| 原始输入 | `[B,L,N,C]=[32,144,134,16]` | batch、历史、节点、输入变量 |
| Adapter 展平 | `[B*N,L,C]=[4288,144,16]` | 每节点作为 upstream batch |
| padding 后 | `[B*N,C,pad_in_len]=[4288,16,144]` | `pad_in_len=ceil(L/12)*12` |
| segment embedding 后 | `[B*N,S0,D]=[4288,12,64]` | `S0=12`，随后为 `[B*N,C,S0,D]` |
| Layer 0 Cross-Time 前后 | outer `[4288,16,12,64]`；internal `[B*N*C,S0,D]=[68608,12,64]` | 沿 segment 做 time attention |
| Layer 0 Cross-Dimension 前后 | outer 不变；internal `[B*N*S0,C,D]=[51456,16,64]` | router sender/receiver 沿变量 |
| 第一次 merge 前后 | `[4288,16,12,64] -> [4288,16,6,64]` | `S1=ceil(S0/2)=6` |
| Layer 1 Cross-Time | outer `[4288,16,6,64]`；internal `[68608,6,64]` | 沿 `S1` |
| Layer 1 Cross-Dimension | internal `[B*N*S1,C,D]=[25728,16,64]`；outer 不变 | 沿变量 |
| encoder list | `[[4288,16,12,64],[4288,16,12,64],[4288,16,6,64]]` | decoder 使用三项 |
| Decoder 输入 | `[B*N,C,S_out,D]=[4288,16,1,64]` | `S_out=ceil(H/12)=1` |
| Decoder layer prediction | `[B*N,C,12]=[4288,16,12]` | padded prediction |
| Decoder 输出 | `[B*N,12,C]=[4288,12,16]` | 三层相加后 rearrange |
| 最终 upstream / Adapter 输出 | `[4288,10,16] -> [32,134,10]` | 裁剪 H，再取 power index 15 |

## Scale0 / Scale1 插入点冻结

Scale0 的真实位置是 `encoder.encode_blocks[0].encode_layers[0]` 中 Cross-Time 的
`norm2` 后、`dim_send = rearrange(...)` 前：

```text
Layer 0 Cross-Time canonical stage
→ Scale0 RA-DS-PFD
→ Gate-Free residual（P2，gamma_0=0.1）
→ Layer 0 Cross-Dimension canonical stage
```

输入/输出必须保持 local `[B,N,C,S0,D]`（formal `[32,134,16,12,64]`）或 upstream
等价 `[B*N,C,S0,D]`。原始 time attention、`norm1`/`MLP1`/`norm2`、dim sender/receiver、
`norm3`/`MLP2`/`norm4`、router、dropout 和 residual 顺序必须保留。

Scale1 的真实位置是 `encoder.encode_blocks[1].merge_layer` 之后、Layer 1 Cross-Time
的 `norm2` 后、其 `dim_send` 前：

```text
Layer 1 Segment Merging
→ Layer 1 Cross-Time canonical stage
→ Scale1 RA-DS-PFD
→ Gate-Free residual（P2，gamma_1=0.1）
→ Layer 1 Cross-Dimension canonical stage
```

输入/输出保持 local `[B,N,C,S1,D]`（formal `[32,134,16,6,64]`）。必须在本地
backbone 组合等价的 merge/time/spatial/dimension；不能用 outer hook，也不能修改
upstream 文件。`spatial_disabled=true` 时不创建传播张量，hook 严格 identity，使
canonical path 可对齐。

## Segment Merging 与 Decoder

`SegMerging.forward` 对 `win_size=2` 的真实公式：

```text
pad = 0                         if S % 2 == 0
pad = 2 - (S % 2)               otherwise
若 pad > 0，复制 x[:, :, -pad:, :] 到末尾
S_next = (S + pad) / 2 = ceil(S / 2)
```

奇数 segment 是复制最后 segment，不是零 padding 或截断。formal：
`S0=ceil(144/12)=12`，`S1=ceil(12/2)=6`。P1 必须增加 odd-S 测试。

`Encoder.forward` 返回 `[input, block0, block1]`。`Decoder.forward` 按索引消费三项，
每层使用 `S_out=ceil(H/seg_len)=1` 的 decoder token，产生 padded 12-step prediction，
逐层相加后得到 `[B_u,12,C]`；`Model.forward` 取最后 10 步。decoder 输出全部变量，
Adapter 再按 `input_power_index` 选择目标通道。P1 可通过现有 controlled loader 复用
upstream module 导出的 `Decoder`, `DecoderLayer`, `TwoStageAttentionLayer`,
`AttentionLayer`, `FullAttention`，不改 upstream。

## TrueUnion、图方向与公共接口

旧 `graph_union.py:union_graphs` 真实做 semantic/distance `(source,target)` 集合 union、
去重、按 `(target,source)` 稳定排序、生成 13 维 edge features，并检查 target indegree。
旧完整 union 是 `E=1536`。旧保存的 `adjacency.npy` 写入
`adjacency[target,source]`，与当前公共 adjacency 约定相反；P2 必须以明确
`edge_index[source,target]` 为真值。

当前 `src/resources/graph.py` 只有 `adjacency`, `edge_index`, `edge_weight`，没有
semantic provenance、13 维 static edge features 或 ordered relation embedding；当前
`DataInfoView` 不携带 raw history、target、mask、train boundary。在模型内直接读取
parquet/build semantic graph 会违反公共数据边界。因此冻结：

```text
P1: 不改公共图接口，spatial_disabled backbone 不需要图。
P2: 当前 GraphResource 不足以独立表达旧 TrueUnion；实现前要确定最小、向后兼容的
    shared relation-resource 或 train-only 预构建资源契约。P0 不实施该变更。
```

当前图代码明确：`adjacency[source,target]=1`，`edge_index[0]=source`，
`edge_index[1]=target`；`symmetrize=true` 取 `max(A,A.T)`，`self_loops=false`，
`torch.nonzero` 给出确定 row-major 顺序，dense graph 不产生重复边。当前 public
STCN 仅注册并传递这些 buffer；RA 必须自己显式实现 gather/scatter。

三节点方向验证：

```text
边 0->1, 2->1, 1->2
edge_index = [[0,2,1], [1,1,2]]
source gather = [v0,v2,v1]
target query  = [q1,q1,q2]
target=1 的两条边一起 softmax；target=2 的一条边单独 softmax
message[1] += a(0->1)v0 + a(2->1)v2
message[2] += a(1->2)v1
```

softmax 必须按 `edge_index[1]` 的 target 入边分组，不能按 source。旧
`SparseRelationAttention` 的 `q[:,target]`、`k/v[:,source]`、target scatter
`scatter_add_` 与该语义一致。

## Static Edge / Ordered Relation / Gate 结论

旧原型的 Static Edge 特征是 `[E,13]`，名称为：
`semantic_similarity`, `semantic_overlap_ratio`, `distance_kernel_weight`,
`normalized_distance`, `relative_x`, `relative_y`, `delta_elevation`,
`terrain_slope`, `terrain_slope_angle`, `is_semantic_edge`, `is_distance_edge`,
`is_both_edge`, `has_elevation`。`StaticEdgeMLP(13,d_ff,heads)` 输出 `[E,heads]`。

`relation_embedding[N,R]` 对有序 `(source,target)` 形成
`[e_t,e_s,e_t-e_s,abs(e_t-e_s),e_t*e_s]`，输入 `[E,5R]`，
`OrderedRelationBias` 输出 `[E,heads]`；A→B 与 B→A 区分。attention content
`[B,E,S,heads]` 只加 edge/relation bias（broadcast `[1,E,1,heads]`）形成 logits。
Value 是 `v[:,source]`；edge/relation bias 不乘 Value，也不乘聚合 Message。

旧 physics bias 虽也只进 logits，但 `RAW_PHYSICS_FEATURES` 包含 `Pab1,Pab2,Pab3`，
必须排除。旧 `VariableConditionedResidualInjection` 包含 `gate_mlp`/sigmoid
per-variable gate，最终只允许两个 gate-free `gamma_0=0.1`、`gamma_1=0.1` residual。
P2 只保留 edge provenance、ordered pair、sparse gather、target softmax/scatter、
logits-only bias；不迁移旧 data/Trainer/evaluator/checkpoint/registry/config/scaler/
results/status/environment 系统、Physics Bias、Pab、Variable Gate 或完整邻居 hidden Value。

## entmax 依赖

```text
解释器: D:\Apps\Miniconda3\envs\env_tslib\python.exe
Python: 3.11.15
torch: 2.5.1+cu124; CUDA available=True
find_spec('entmax'): None
pip show entmax: not found, exit code 1
installed metadata/license: UNKNOWN（未安装）
requirements/project declaration: 无 entmax
状态: ENTMAX_NOT_AVAILABLE
```

P0 未安装依赖。P4/PFD2 必须在实现前决定受控依赖/license 或模型本地等价实现，不能
直接假设 `import entmax` 可用。

## formal shape、E/S0/S1 与显存

真实配置：`B=32,L=144,N=134,C=16,H=10,D=64,heads=4,e_layers=2,seg_len=12,
win_size=2,k=5,symmetrize=true,self_loops=false,binary`；weights/input 默认 fp32，
训练 `amp=true, amp_dtype=float16`，GPU autocast float16。真实数据为 `[52559,134,16]`。
公共 physical graph 真实 `E=760`，无 self-loop、无重复边、对称，入/出度 5–9；
`B*N=4288`，`B*N*C=68608`，`S0=12`，`S1=6`。

单张量理论占用（不是训练总显存；fp32/fp16 分别按 4/2 bytes；训练 activation
可能由 autograd 保留）：

| 张量 | shape | elements | fp32 MiB | fp16 MiB | chunk/生命周期 |
| --- | --- | ---: | ---: | ---: | --- |
| raw x | `[32,144,134,16]` | 9,879,552 | 37.688 | 18.844 | batch/variable；forward→backward |
| scale0 token | `[4288,16,12,64]` | 52,690,944 | 201.000 | 100.500 | encoder/decoder list |
| scale1 token | `[4288,16,6,64]` | 26,345,472 | 100.500 | 50.250 | encoder/decoder list |
| scale0 time score | `[68608,4,12,12]` | 39,518,208 | 150.750 | 75.375 | batch-node/variable chunk |
| scale0 dim score | `[51456,4,16,16]` | 52,690,944 | 201.000 | 100.500 | batch-node/segment chunk |
| scale1 time score | `[68608,4,6,6]` | 9,879,552 | 37.688 | 18.844 | batch-node/variable chunk |
| scale1 dim score | `[25728,4,16,16]` | 26,345,472 | 100.500 | 50.250 | batch-node/segment chunk |
| RA edge logits/attention | `[32,760,12,4]` | 1,167,360 | 4.453 | 2.227 | edge/segment/head |
| RA one q/k/v gather | `[32,760,12,4,16]` | 18,677,760 | 71.250 | 35.625 | edge chunk；q/k/v each同量级 |
| `[B,E,C,S,heads]`, C=16 | `[32,760,16,12,4]` | 18,677,760 | 71.250 | 35.625 | edge/variable/segment chunk |
| candidate bank d_prop=16 | `[32,134,12,56,16]` | 46,104,576 | 175.875 | 87.938 | P3 variable/segment/candidate chunk |
| candidate bank d_prop=32 | `[32,134,12,56,32]` | 92,209,152 | 351.750 | 175.875 | P3 variable/segment/candidate chunk |
| edge candidate expansion d_prop=16 | `[32,760,12,56,16]` | 261,488,640 | 997.500 | 498.750 | 应避免；selector-first/edge chunk |
| edge candidate expansion d_prop=32 | `[32,760,12,56,32]` | 522,977,280 | 1,995.000 | 997.500 | 应避免；selector-first/edge chunk |

`d_prop` 尚未冻结，16/32 只是量级示例。P3/P4 不得先物化 edge candidate expansion；
P5 必须 selector-first、edge/variable/segment chunk，并做小形状 chunk/non-chunk 等价。

canonical formal baseline（不是 RA full-shape）：GPU `NVIDIA GeForce RTX 4070 Ti SUPER`，
total `16375.5 MiB`，固定 seed，`[32,144,134,16]`、AMP float16，输出 `[32,134,10]`，
parameter `532068`，forward `2.5303s`，backward `0.7627s`，peak allocated
`10361.7256 MiB`，reserved `10526.0000 MiB`，输出/loss/输入梯度/参数梯度 finite。
P0 没有 RA 模型，不能把该 baseline 当作 RA 峰值。

## P1 精确文件与接口冻结

新增文件：

```text
src/models/ra_ds_pfd_crossformer/__init__.py
src/models/ra_ds_pfd_crossformer/model.py
src/models/ra_ds_pfd_crossformer/backbone.py
configs/models/ra_ds_pfd_crossformer.yaml
tests/test_ra_ds_pfd_crossformer_backbone.py
```

P1 允许修改现有文件：`NONE`。禁止修改当前 Crossformer/STCN/NodeSharedLSTM、
`src/data/*`, `src/engine/*`, `src/cli/*`, `src/runtime/*`, `src/resources/graph.py`、
Time-Series-Library、dataset/results/logs/checkpoint 和 zip。现有 controlled loader
返回的 Crossformer module 已提供 decoder 组件，优先不改 integration loader。

冻结接口：

```text
build_model(model_config, data_info) -> ForecastModel
ModelInput.x: [B,L,N,C]
output: [B,N,H]
local tokens: [B,N,C,S,D]
apply_spatial(scale: 0|1, self_tokens[B,N,C,S,D],
              propagation_tokens|None) -> self_tokens[B,N,C,S,D]
```

`spatial_disabled=true` 不创建 graph/propagation activation，在两处 hook 严格 identity。
YAML 只放 `d_model,n_heads,d_ff,e_layers,seg_len,win_size,factor,dropout,spatial_disabled`；
batch、loss、optimizer、horizon、feature list 留在公共配置/DataInfoView。

P1 测试必须有：小形状 shape/finite backward；固定 seed、dropout=0、显式复制 upstream
参数的 token/decoder canonical 对齐；odd merge 复制最后 segment；checkpoint reload；
feature-name power index；现有 Crossformer/isolation/graph 回归。P1 不实现 PFD0。

## 明确未实现、测试结果和限制

```text
RA-DS-PFD model/backbone/YAML 不存在
PFD0、TrueUnion resource、56 candidates、CausalPropagationFeatureBank、learned_change12 不存在
PFD1/PFD2/PFD3/PFD4/PFD5 selector 不存在
Gate-Free residual、entmax、Selector-first memory optimization 不存在
RA full-shape forward/backward 和正式 Full training 未运行
```

测试解释器：`D:\Apps\Miniconda3\envs\env_tslib\python.exe`。

| 命令 | 真实结果 |
| --- | --- |
| `python -m pytest -q tests/test_crossformer_model.py tests/test_time_series_library_isolation.py tests/test_graph_resources.py` | `12 passed in 2.51s` |
| `python scripts/generate_command_reference.py --check` | `command reference is up to date` |
| `python -m pytest -q tests/test_documentation_consistency.py` | `7 passed in 0.35s` |
| `python -m pytest -q` | `138 passed, 3 skipped in 12.76s` |

3 个 skip 是 `tests/test_stcn_model.py` 明确要求 formal `tsl` package（只在 env_tsl），
不是新增失败。旧模型回归 `NOT RUN`：旧源码隔离且当前仓库没有旧模型 worker 入口，
遵守安全要求没有执行旧 runner/训练脚本；当前 canonical/graph/loader isolation 回归
已通过。

限制：upstream 是被父仓库忽略的普通源码目录，没有独立 revision，只能冻结文件 hash；
`E=760` 是当前 physical graph，不是未来 TrueUnion E；RA runtime peak 尚无真实值。

## Git、下一阶段与绝对不要重复的坑

本阶段只应提交 `HANDOFF.md`、`docs/handovers/ra_ds_pfd/PHASE_INDEX.md` 和本文件。
不能自引用最终 commit，使用 `completion_commit: SELF`；提交后用：

```powershell
git log --diff-filter=A --format="%H" -- docs/handovers/ra_ds_pfd/phase_00_audit.md
```

P1 只允许实现本快照冻结的 local canonical Backbone、adapter、YAML 和测试；P0 commit/
push 真实成功后才可开始。不要整体复制旧项目；不要用 current Adapter 当插层 Backbone；
不要把旧 `adjacency[target,source]` 当成当前 `adjacency[source,target]`；不要按 source
softmax；不要将 edge/relation bias 乘 Value/Message；不要迁移 Pab/Physics Bias/Variable
Gate/完整邻居 hidden Value；不要硬编码 power index；不要用 canonical baseline 冒充 RA
full-shape；不要降低正式 shape 规避 OOM。
