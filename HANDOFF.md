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

当前基准模型接入状态仍为 NodeSharedLSTM、Crossformer 和 STCN；正式 Full 训练与
正式 GPU 显存验收尚未运行。当前主机为 NVIDIA GeForce RTX 4070 Ti SUPER、CUDA
12.4；链接工作树缺少外部 `Time-Series-Library` 源码，因此完整 pytest 的
Crossformer 相关 4 项暂标记为未运行环境阻塞，其余共享测试通过。尚未完成项是
在对应 `env_tslib`/`env_tsl` 和真实数据上按固定序列完成基准模型的 GPU 验收；不
将此分支用于 RA-DS-PFD 开发。
