"""The single argparse schema for the project command line.

This module deliberately contains parser construction and the mapping from
public command-line names to experiment YAML paths.  Command implementations
consume the resulting namespace; they do not define a second parser.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CONFIG_PATH = Path("configs/experiment.yaml")


@dataclass(frozen=True)
class PublicOverrideSpec:
    """Metadata shared by argparse, config resolution and documentation."""

    dest: str
    option: str
    yaml_path: tuple[str, ...]
    value_type: str
    help: str
    nargs: str | None = None


_OVERRIDE_HELP = (
    "未提供时采用 configs/experiment.yaml；显式提供时仅覆盖本次运行。"
)


PUBLIC_OVERRIDE_SPECS: tuple[PublicOverrideSpec, ...] = (
    PublicOverrideSpec(
        dest="lookback",
        option="--lookback",
        yaml_path=("data", "lookback"),
        value_type="int",
        help=f"覆盖 data.lookback；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="batch_size",
        option="--batch-size",
        yaml_path=("training", "train_batch_size"),
        value_type="int",
        help=f"覆盖 training.train_batch_size；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="epochs",
        option="--epochs",
        yaml_path=("training", "epochs"),
        value_type="int",
        help=f"覆盖 training.epochs；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="loss",
        option="--loss",
        yaml_path=("training", "loss"),
        value_type="str",
        help=f"覆盖 training.loss；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="learning_rate",
        option="--learning-rate",
        yaml_path=("training", "learning_rate"),
        value_type="float",
        help=f"覆盖 training.learning_rate；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="train_ratio",
        option="--train-ratio",
        yaml_path=("split", "train_ratio"),
        value_type="float",
        help=f"覆盖 split.train_ratio；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="val_ratio",
        option="--val-ratio",
        yaml_path=("split", "val_ratio"),
        value_type="float",
        help=f"覆盖 split.val_ratio；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="test_ratio",
        option="--test-ratio",
        yaml_path=("split", "test_ratio"),
        value_type="float",
        help=f"覆盖 split.test_ratio；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="eval_horizons",
        option="--eval-horizons",
        yaml_path=("data", "eval_horizons"),
        value_type="int",
        nargs="+",
        help=f"覆盖 data.eval_horizons（可传多个整数）；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="feature_columns",
        option="--feature-columns",
        yaml_path=("data", "feature_columns"),
        value_type="str",
        nargs="+",
        help=f"覆盖 data.feature_columns（可传多个列名）；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="seed",
        option="--seed",
        yaml_path=("training", "seed"),
        value_type="int",
        help=f"覆盖 training.seed；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="num_workers",
        option="--num-workers",
        yaml_path=("runtime", "num_workers"),
        value_type="int",
        help=f"覆盖 runtime.num_workers；{_OVERRIDE_HELP}",
    ),
    PublicOverrideSpec(
        dest="amp",
        option="--amp",
        yaml_path=("training", "amp"),
        value_type="bool",
        help=f"显式启用或禁用 training.amp（使用 --amp/--no-amp）；{_OVERRIDE_HELP}",
    ),
)

PUBLIC_OVERRIDE_BY_DEST = {spec.dest: spec for spec in PUBLIC_OVERRIDE_SPECS}
PUBLIC_OVERRIDE_BY_PATH = {
    ".".join(spec.yaml_path): spec for spec in PUBLIC_OVERRIDE_SPECS
}


_TRAIN_OVERRIDES = tuple(spec.dest for spec in PUBLIC_OVERRIDE_SPECS)
_EVALUATE_OVERRIDES = (
    "lookback",
    "batch_size",
    "loss",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "eval_horizons",
    "feature_columns",
    "num_workers",
)
_SHAPE_OVERRIDES = ("lookback", "batch_size", "eval_horizons", "feature_columns")


def _type_for(spec: PublicOverrideSpec):
    return {"int": int, "float": float, "str": str, "bool": bool}[spec.value_type]


def _add_public_overrides(
    parser: argparse.ArgumentParser,
    *,
    names: Iterable[str] = _TRAIN_OVERRIDES,
) -> None:
    group = parser.add_argument_group(
        "公共实验覆盖（默认来自 configs/experiment.yaml）"
    )
    for name in names:
        spec = PUBLIC_OVERRIDE_BY_DEST[name]
        kwargs = {
            "dest": spec.dest,
            "default": None,
            "help": spec.help,
        }
        if spec.nargs is not None:
            kwargs["nargs"] = spec.nargs
        if spec.value_type == "bool":
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = _type_for(spec)
        group.add_argument(spec.option, **kwargs)


def _add_task_context(
    parser: argparse.ArgumentParser,
    *,
    include_run_id: bool = True,
    include_output_root: bool = True,
) -> None:
    group = parser.add_argument_group("任务定位")
    group.add_argument("--model", required=True, help="模型目录名，例如 node_shared_lstm")
    if include_run_id:
        group.add_argument("--run-id", help="本次运行目录名；训练和评估未提供时自动生成")
    group.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备；auto 根据 CUDA 可用性选择",
    )
    if include_output_root:
        group.add_argument(
            "--output-root",
            type=Path,
            help="结果根目录；未提供时使用项目 results/",
        )


def _add_config_files(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("配置文件")
    group.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="公共实验配置；默认 configs/experiment.yaml",
    )
    group.add_argument(
        "--model-config",
        type=Path,
        help="模型结构配置；未提供时自动使用 configs/models/<model>.yaml",
    )


def _build_train(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train",
        help="训练一个模型并保存 checkpoint、配置和指标",
        description="训练一个 PhDPaper3 预测模型。",
    )
    _add_task_context(parser)
    _add_config_files(parser)
    _add_public_overrides(parser)
    group = parser.add_argument_group("训练任务专属参数")
    group.add_argument("--resume", type=Path, help="从已有 checkpoint 继续训练")
    group.add_argument("--smoke", action="store_true", help="执行明确限制的短训练")
    group.add_argument("--smoke-epochs", type=int, help="--smoke 的 epoch 上限")
    group.add_argument("--smoke-max-train-updates", type=int, help="--smoke 的训练更新上限")
    group.add_argument("--smoke-max-eval-batches", type=int, help="--smoke 的评估 batch 上限")


def _build_evaluate(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "evaluate",
        help="使用 checkpoint 进行评估",
        description="加载 checkpoint，并检查其配置与当前最终配置兼容后进行评估。",
    )
    _add_task_context(parser)
    _add_config_files(parser)
    _add_public_overrides(parser, names=_EVALUATE_OVERRIDES)
    group = parser.add_argument_group("评估任务专属参数")
    group.add_argument("--checkpoint", type=Path, required=True, help="待评估的 checkpoint 或其目录")
    group.add_argument(
        "--split",
        choices=("validation", "test", "both"),
        default="both",
        help="输出哪个数据划分的评估结果；默认 both",
    )


def _build_check(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "check",
        help="检查模型导入、完整张量形状和反向传播",
        description="构造模型并执行 forward/backward 形状检查。",
    )
    _add_task_context(parser, include_run_id=False, include_output_root=False)
    _add_config_files(parser)
    _add_public_overrides(parser, names=_SHAPE_OVERRIDES)
    group = parser.add_argument_group("检查任务专属参数")
    group.add_argument("--full-shape", action="store_true", help="使用 YAML/覆盖后的正式 batch size")


def _build_preflight(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "preflight",
        help="验证配置、数据（可选）和模型导入",
        description="在正式训练前验证最终配置、数据文件和模型结构。",
    )
    _add_task_context(parser, include_run_id=False, include_output_root=False)
    _add_config_files(parser)
    _add_public_overrides(parser, names=_SHAPE_OVERRIDES)
    group = parser.add_argument_group("预检任务专属参数")
    group.add_argument("--no-data", action="store_true", help="只检查配置和模型，不读取正式 parquet")


def _build_repeatability(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "repeatability",
        help="用相同配置执行两次短运行并比较结果",
        description="重复两次明确限制的短运行；两次使用完全相同的命令行覆盖。",
    )
    _add_task_context(parser, include_run_id=False)
    _add_config_files(parser)
    _add_public_overrides(parser)
    group = parser.add_argument_group("重复性任务专属参数")
    group.add_argument(
        "--prediction-atol",
        type=float,
        default=1e-6,
        help="预测数组允许的最大绝对误差，默认 1e-6",
    )
    group.add_argument(
        "--metric-atol",
        type=float,
        default=0.0,
        help="指标允许的最大绝对误差，默认 0",
    )


def _build_batch(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "batch",
        help="按顺序运行多个模型",
        description="按同一份公共配置和同一组命令行覆盖运行多个模型。",
    )
    group = parser.add_argument_group("任务定位")
    group.add_argument(
        "--models",
        nargs="+",
        default=["node_shared_lstm"],
        help="模型名称列表；默认 node_shared_lstm",
    )
    group.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备；auto 根据 CUDA 可用性选择",
    )
    group.add_argument("--output-root", type=Path, help="结果根目录；未提供时使用项目 results/")
    _add_config_files(parser)
    _add_public_overrides(parser)
    task = parser.add_argument_group("批处理任务专属参数")
    task.add_argument("--smoke", action="store_true", help="所有模型执行明确限制的短训练")
    task.add_argument("--continue-on-error", action="store_true", help="一个模型失败后继续其他模型")
    task.add_argument("--skip-completed", action="store_true", help="跳过已有 best.pt 的 batch_<model> 运行")
    task.add_argument("--smoke-epochs", type=int, help="--smoke 的 epoch 上限")
    task.add_argument("--smoke-max-train-updates", type=int, help="--smoke 的训练更新上限")
    task.add_argument("--smoke-max-eval-batches", type=int, help="--smoke 的评估 batch 上限")


def build_parser() -> argparse.ArgumentParser:
    """Build the sole user-facing project parser."""

    parser = argparse.ArgumentParser(
        prog="python scripts/run.py",
        description="PhDPaper3 的统一命令入口。",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="可用命令",
        metavar="{train,evaluate,check,preflight,repeatability,batch}",
    )
    _build_train(subparsers)
    _build_evaluate(subparsers)
    _build_check(subparsers)
    _build_preflight(subparsers)
    _build_repeatability(subparsers)
    _build_batch(subparsers)
    return parser


def build_reference_parser() -> argparse.ArgumentParser:
    """Build the small parser owned by the documentation utility."""

    parser = argparse.ArgumentParser(
        prog="python scripts/generate_command_reference.py",
        description="从统一命令 schema 生成命令参考文档。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验 docs/COMMAND_REFERENCE.md 是否与当前 schema 一致",
    )
    return parser


def override_spec_for_dest(dest: str) -> PublicOverrideSpec | None:
    """Return public override metadata for an argparse destination."""

    return PUBLIC_OVERRIDE_BY_DEST.get(dest)
