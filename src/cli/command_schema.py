"""The sole user-facing argparse schema for the project."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from runtime.losses import LOSS_NAMES


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
    additional_yaml_paths: tuple[tuple[str, ...], ...] = ()

    @property
    def yaml_paths(self) -> tuple[tuple[str, ...], ...]:
        return (self.yaml_path, *self.additional_yaml_paths)


_OVERRIDE_HELP = "未提供时采用 configs/experiment.yaml；显式提供时仅覆盖本次运行。"


PUBLIC_OVERRIDE_SPECS: tuple[PublicOverrideSpec, ...] = (
    PublicOverrideSpec("lookback", "--lookback", ("data", "lookback"), "int", f"覆盖 data.lookback；{_OVERRIDE_HELP}"),
    PublicOverrideSpec("batch_size", "--batch-size", ("training", "train_batch_size"), "int", f"覆盖 training.train_batch_size；{_OVERRIDE_HELP}"),
    PublicOverrideSpec(
        "eval_batch_size",
        "--eval-batch-size",
        ("training", "val_batch_size"),
        "int",
        f"同时覆盖 training.val_batch_size 和 training.test_batch_size；{_OVERRIDE_HELP}",
        additional_yaml_paths=(("training", "test_batch_size"),),
    ),
    PublicOverrideSpec("epochs", "--epochs", ("training", "epochs"), "int", f"覆盖 training.epochs；{_OVERRIDE_HELP}"),
    PublicOverrideSpec("loss", "--loss", ("training", "loss"), "str", f"覆盖 training.loss；{_OVERRIDE_HELP}"),
    PublicOverrideSpec("learning_rate", "--learning-rate", ("training", "learning_rate"), "float", f"覆盖 training.learning_rate；{_OVERRIDE_HELP}"),
    PublicOverrideSpec("train_ratio", "--train-ratio", ("split", "train_ratio"), "float", f"覆盖 split.train_ratio；{_OVERRIDE_HELP}"),
    PublicOverrideSpec("val_ratio", "--val-ratio", ("split", "val_ratio"), "float", f"覆盖 split.val_ratio；{_OVERRIDE_HELP}"),
    PublicOverrideSpec("test_ratio", "--test-ratio", ("split", "test_ratio"), "float", f"覆盖 split.test_ratio；{_OVERRIDE_HELP}"),
    PublicOverrideSpec("eval_horizons", "--eval-horizons", ("data", "eval_horizons"), "int", f"覆盖 data.eval_horizons（可传多个整数）；{_OVERRIDE_HELP}", nargs="+"),
    PublicOverrideSpec("feature_columns", "--feature-columns", ("data", "feature_columns"), "str", f"覆盖 data.feature_columns（可传多个列名）；{_OVERRIDE_HELP}", nargs="+"),
    PublicOverrideSpec("seed", "--seed", ("training", "seed"), "int", f"覆盖 training.seed；{_OVERRIDE_HELP}"),
    PublicOverrideSpec("num_workers", "--num-workers", ("runtime", "num_workers"), "int", f"覆盖 runtime.num_workers；{_OVERRIDE_HELP}"),
    PublicOverrideSpec("amp", "--amp", ("training", "amp"), "bool", f"显式启用或禁用 training.amp（使用 --amp/--no-amp）；{_OVERRIDE_HELP}"),
)

PUBLIC_OVERRIDE_BY_DEST = {spec.dest: spec for spec in PUBLIC_OVERRIDE_SPECS}
PUBLIC_OVERRIDE_BY_PATH = {
    ".".join(path): spec
    for spec in PUBLIC_OVERRIDE_SPECS
    for path in spec.yaml_paths
}

_TRAIN_OVERRIDES = tuple(spec.dest for spec in PUBLIC_OVERRIDE_SPECS)
_EVALUATE_OVERRIDES = (
    "lookback",
    "eval_batch_size",
    "loss",
    "train_ratio",
    "val_ratio",
    "test_ratio",
    "eval_horizons",
    "feature_columns",
    "num_workers",
    "amp",
)
_SHAPE_OVERRIDES = (
    "lookback",
    "batch_size",
    "eval_batch_size",
    "eval_horizons",
    "feature_columns",
)


def _type_for(spec: PublicOverrideSpec):
    return {"int": int, "float": float, "str": str, "bool": bool}[spec.value_type]


def _add_public_overrides(
    parser: argparse.ArgumentParser,
    *,
    names: Iterable[str] = _TRAIN_OVERRIDES,
) -> None:
    group = parser.add_argument_group("公共实验覆盖（默认来自 configs/experiment.yaml）")
    for name in names:
        spec = PUBLIC_OVERRIDE_BY_DEST[name]
        kwargs: dict[str, object] = {
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
        if spec.dest == "loss":
            kwargs["choices"] = LOSS_NAMES
        group.add_argument(spec.option, **kwargs)


def _add_task_context(
    parser: argparse.ArgumentParser,
    *,
    allow_multiple_models: bool,
    include_run_id: bool = True,
    include_output_root: bool = True,
) -> None:
    group = parser.add_argument_group("任务定位")
    model_kwargs: dict[str, object] = {
        "required": True,
        "help": "模型目录名；可一次指定一个或多个模型" if allow_multiple_models else "模型目录名",
    }
    if allow_multiple_models:
        model_kwargs["nargs"] = "+"
    group.add_argument("--model", **model_kwargs)
    if include_run_id:
        group.add_argument("--run-id", help="本次运行的共同实验 ID")
    group.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备；auto 根据 CUDA 可用性选择",
    )
    if include_output_root:
        group.add_argument("--output-root", type=Path, help="结果根目录；未提供时使用项目 results/")


def _add_config_files(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("配置文件")
    group.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="公共实验配置；默认 configs/experiment.yaml")
    group.add_argument("--model-config", type=Path, help="单模型结构配置；多模型时自动读取 configs/models/<model>.yaml")


def _add_run_modes(parser: argparse.ArgumentParser, *, include_resume: bool = True) -> None:
    group = parser.add_argument_group("运行模式")
    modes = group.add_mutually_exclusive_group()
    if include_resume:
        modes.add_argument("--resume", action="store_true", help="从每个模型目录的有效 last.pt 继续运行")
    modes.add_argument("--overwrite", action="store_true", help="先归档旧结果，再使用原 ID 重新运行")
    modes.add_argument("--id-suffix", help="追加安全后缀，例如 rerun1；生成 <run-id>__rerun1")


def _build_train(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="训练一个或多个模型并保存结果", description="训练一个或多个 PhDPaper3 预测模型；每个模型在独立 Python 子进程中运行。")
    _add_task_context(parser, allow_multiple_models=True)
    _add_config_files(parser)
    _add_public_overrides(parser)
    _add_run_modes(parser)
    group = parser.add_argument_group("训练任务专属参数")
    group.add_argument("--fail-fast", action="store_true", help="第一个模型失败后立即停止后续模型")
    group.add_argument("--smoke", action="store_true", help="执行明确限制的短训练")
    group.add_argument("--smoke-epochs", type=int, help="--smoke 的 epoch 上限")
    group.add_argument("--smoke-max-train-updates", type=int, help="--smoke 的训练更新上限")
    group.add_argument("--smoke-max-eval-batches", type=int, help="--smoke 的评估 batch 上限")
    group.add_argument(
        "--environment-preflight-only",
        action="store_true",
        help="只预检本批次所需运行环境，不启动模型 worker",
    )


def _build_evaluate(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evaluate", help="使用 checkpoint 进行评估", description="加载 checkpoint，并检查其配置与当前最终配置兼容后进行评估。")
    _add_task_context(parser, allow_multiple_models=False)
    _add_config_files(parser)
    _add_public_overrides(parser, names=_EVALUATE_OVERRIDES)
    group = parser.add_argument_group("评估任务专属参数")
    group.add_argument("--checkpoint", type=Path, help="待评估的 checkpoint 或其目录；省略时从 --run-id 自动查找")
    group.add_argument("--split", choices=("validation", "test", "both"), default="both", help="输出哪个数据划分的评估结果；默认 both")


def _build_check(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("check", help="检查一个或多个模型的形状和反向传播", description="每个模型在独立 Python 子进程中执行 forward/backward 形状检查。")
    _add_task_context(parser, allow_multiple_models=True, include_run_id=False, include_output_root=False)
    _add_config_files(parser)
    _add_public_overrides(parser, names=_SHAPE_OVERRIDES)
    group = parser.add_argument_group("检查任务专属参数")
    group.add_argument("--full-shape", action="store_true", help="使用 YAML/覆盖后的正式 batch size")
    group.add_argument(
        "--environment-preflight-only",
        action="store_true",
        help="只预检本批次所需运行环境，不启动模型 worker",
    )


def _build_preflight(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("preflight", help="验证一个或多个模型的配置和数据", description="每个模型在独立 Python 子进程中验证最终配置、数据文件和模型结构。")
    _add_task_context(parser, allow_multiple_models=True, include_run_id=False, include_output_root=False)
    _add_config_files(parser)
    _add_public_overrides(parser, names=_SHAPE_OVERRIDES)
    group = parser.add_argument_group("预检任务专属参数")
    group.add_argument("--no-data", action="store_true", help="只检查配置和模型，不读取正式 parquet")
    group.add_argument(
        "--environment-preflight-only",
        action="store_true",
        help="只预检本批次所需运行环境，不启动模型 worker",
    )


def _build_repeatability(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("repeatability", help="用独立子进程执行两次短运行并比较结果", description="每个模型由两个完全独立的 Python 子进程完成 A/B 短运行。")
    _add_task_context(parser, allow_multiple_models=True)
    _add_config_files(parser)
    _add_public_overrides(parser)
    _add_run_modes(parser, include_resume=False)
    group = parser.add_argument_group("重复性任务专属参数")
    group.add_argument("--prediction-atol", type=float, default=1e-6, help="预测数组允许的最大绝对误差，默认 1e-6")
    group.add_argument("--metric-atol", type=float, default=0.0, help="指标允许的最大绝对误差，默认 0")
    group.add_argument(
        "--environment-preflight-only",
        action="store_true",
        help="只预检本批次所需运行环境，不启动重复性 worker",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the sole user-facing project parser."""

    parser = argparse.ArgumentParser(prog="python scripts/run.py", description="PhDPaper3 的统一命令入口。")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="可用命令",
        metavar="{train,evaluate,check,preflight,repeatability}",
    )
    _build_train(subparsers)
    _build_evaluate(subparsers)
    _build_check(subparsers)
    _build_preflight(subparsers)
    _build_repeatability(subparsers)
    return parser


def build_reference_parser() -> argparse.ArgumentParser:
    """Build the parser owned by the documentation utility."""

    parser = argparse.ArgumentParser(prog="python scripts/generate_command_reference.py", description="从统一命令 schema 生成命令参考文档。")
    parser.add_argument("--check", action="store_true", help="只校验 docs/COMMAND_REFERENCE.md 是否与当前 schema 一致")
    return parser


def override_spec_for_dest(dest: str) -> PublicOverrideSpec | None:
    return PUBLIC_OVERRIDE_BY_DEST.get(dest)
