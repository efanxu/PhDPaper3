# RA-DS-PFD Crossformer 阶段索引

状态词仅使用：`NOT_STARTED`、`IN_PROGRESS`、`BLOCKED`、`PASS`、
`PASS_WITH_NOTES`、`FAILED`。

| 阶段 | 名称 | 状态 | 开始 HEAD | 完成提交 | 阶段快照 | 关键结论 | 下一阶段门禁 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 当前框架、旧原型与 upstream Crossformer 真实审计和接口冻结 | PASS | `2122dbaa3fe6df5b5c193dd49d2da2e765be7580` | `SELF` | `docs/handovers/ra_ds_pfd/phase_00_audit.md` | canonical layout、Scale0/Scale1、Segment Merging、Decoder、图方向、bias 语义、`E=760`、entmax 状态和 P1 文件范围已由本机代码/命令冻结 | P1 只能实现 local canonical Backbone；必须先通过 spatial-disabled canonical 对齐 |
| P1 | 可插入 canonical Crossformer Backbone | NOT_STARTED | — | — | — | — | P0 `PASS` commit/push 已验证；forward/backward、checkpoint reload、canonical 对齐通过 |
| P2 | TrueUnion + Ordered Relation + PFD0 | NOT_STARTED | — | — | — | 需先解决向后兼容的 shared relation-resource 边界；P0 未实施 | sparse/dense 等价、target softmax、logits-only bias、gate-free residual |
| P3 | 56 候选 Feature Bank + PFD1 | NOT_STARTED | — | — | — | — | 因果、零和 learned kernel、segment 对齐和无功率传播 |
| P4-A | PFD2 Global Sparse Selector | NOT_STARTED | — | — | — | entmax 当前为 `ENTMAX_NOT_AVAILABLE` | Entmax、hard top-k、ST 梯度、确定性 eval |
| P4-B | PFD3 Scale-Specific Selector | NOT_STARTED | — | — | — | — | 两尺度独立选择和退化测试 |
| P4-C | PFD4 Scale + Relation Selector | NOT_STARTED | — | — | — | — | relation/static edge 条件梯度和退化测试 |
| P4-D | PFD5 Full Selector | NOT_STARTED | — | — | — | — | source context、无泄漏、条件消融 |
| P5 | Full-shape 显存优化 | NOT_STARTED | — | — | — | 当前仅有 canonical baseline，未实现 RA memory path | chunk/non-chunk 等价、正式 shape forward/backward、峰值证据 |
| P6 | 公共验收和最终提交 | NOT_STARTED | — | — | — | — | INTERFACE_SMALL 至 Repeatability 全部有证据 |

## SELF 解析规则

P0 的 `完成提交=SELF` 是阶段快照自引用的稳定约定。提交后使用：

```powershell
git log --diff-filter=A --format="%H" -- docs/handovers/ra_ds_pfd/phase_00_audit.md
```

得到 P0 首次新增快照的 commit SHA；最终回复必须另外报告 `git rev-parse HEAD` 的
真实值。
