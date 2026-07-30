# Handoff

## Architecture

`scripts/run.py` is the only public entry point. `src/cli/command_schema.py`
defines all public arguments. The parent scheduler reads each model's
`runtime.environment`, resolves its target Python, writes one request, and
starts one worker subprocess per model. Data lives in `dataset/` and results
live under `results/<model>/<run_id>/`.

## Core capabilities

- one or many models through `train --model ...`;
- isolated model processes, per-model logs and atomic run status;
- fail-closed new runs, checkpoint resume, archive-based overwrite and ID suffixes;
- complete epoch checkpoint state, deterministic resume and independent A/B repeatability;
- shared training/evaluation timing, throughput, parameter and GPU-memory metrics;
- generated command reference and documentation consistency tests.
- supports `tslib` and `tsl` runtime environments;
- defaults to `tslib` when a model YAML omits `runtime`;
- automatically resolves interpreters and can mix both environments in one batch;
- environment preflight can run without creating model result directories or workers;
- paper comparison CSVs contain H3, H6 and H10 metrics when those horizons are configured;
- repeatability runs one complete multi-model A batch followed by one complete B batch and compares every test horizon;
- keeps each model in an independent process;
- treats the local `Time-Series-Library` source as read-only;
- provides `tsl` through `env_tsl`.
- provides a real Node Shared Time-Series-Library integration path and a pure
  `tsl` graph-model path; the parent scheduler switches workers between them.
- builds shared SDWPF graph resources once at model construction, then stores
  sparse edge buffers in graph adapters for checkpoint-safe device movement.

## Known limitations

Formal training still requires the two local protocol-named parquet files in
`dataset/`. The current public configuration keeps AMP CUDA-only and the
loader materializes the cleaned SDWPF arrays in memory. No formal GPU claim is
made without running the corresponding command on a CUDA machine.

## Next step

Add a model directory and a `runtime`/`model` YAML, implement only
`build_model(model_config, data_info)`, then run the generated command checks
and the relevant tests. Ordinary model parameters do not require public
documentation changes.

## Keep out of the project

Do not add a second CLI, model-specific training infrastructure, batch alias,
`--models`, StudySpec/ModelSpec, certificates, declarations, manifests,
readiness protocols, or experiment/report Markdown files.
