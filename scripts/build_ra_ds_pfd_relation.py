"""Build or migrate the RA-DS-PFD TrueUnion relation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from _bootstrap import ROOT  # noqa: F401
from models.ra_ds_pfd_crossformer.relation_builder import (
    RelationBuildError,
    RelationConsistencyError,
    build_trueunion_from_project,
    compare_against_old_graph,
    convert_old_graph,
    public_node_ids_from_project,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the train-only TrueUnion resource or convert an old graph directory."
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resources/ra_ds_pfd_trueunion_v1.yaml"),
        help="resource YAML containing construction parameters and the output path",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-from-data", action="store_true")
    mode.add_argument("--convert-old-graph", type=Path)
    parser.add_argument("--verify-against-old-graph", type=Path)
    parser.add_argument("--output", type=Path, help="optional explicit NPZ output path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    try:
        if args.build_from_data:
            graph, output = build_trueunion_from_project(
                root,
                config,
                device=args.device,
                output_path=args.output,
            )
            verification = None
            if args.verify_against_old_graph is not None:
                expected = public_node_ids_from_project(root)
                verification = compare_against_old_graph(
                    graph,
                    args.verify_against_old_graph,
                    expected_node_ids=expected,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            payload = {
                "mode": "build-from-data",
                "output": str(output),
                "node_count": len(graph.node_ids),
                "edge_count": graph.edge_count,
                "semantic_edge_count": graph.semantic_edge_count,
                "distance_edge_count": graph.distance_edge_count,
                "both_edge_count": graph.both_edge_count,
                "verification": verification,
            }
        else:
            try:
                expected = public_node_ids_from_project(root)
            except (FileNotFoundError, RelationBuildError):
                # Conversion is intentionally usable without private formal
                # data when the old manifest carries its own proved node_ids.
                expected = None
            output = args.output
            if output is None:
                # Conversion without --output follows the resource YAML output.
                import yaml

                resource = yaml.safe_load(config.read_text(encoding="utf-8"))
                output = root / str(resource["output"]["file"])
            if not output.is_absolute():
                output = root / output
            graph = convert_old_graph(
                args.convert_old_graph,
                output,
                expected_node_ids=expected,
            )
            verification = None
            if args.verify_against_old_graph is not None:
                verification = compare_against_old_graph(
                    graph,
                    args.verify_against_old_graph,
                    expected_node_ids=expected,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            payload = {
                "mode": "convert-old-graph",
                "output": str(Path(output).resolve()),
                "node_count": len(graph.node_ids),
                "edge_count": graph.edge_count,
                "semantic_edge_count": graph.semantic_edge_count,
                "distance_edge_count": graph.distance_edge_count,
                "both_edge_count": graph.both_edge_count,
                "verification": verification,
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0
    except RelationConsistencyError as exc:
        print(json.dumps(exc.report, ensure_ascii=False, indent=2, default=str), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI reports a concise build failure
        print(f"TrueUnion build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
