# Handoff

## Architecture

`scripts/run.py` is the only public entry point. `src/cli/command_schema.py`
defines all public arguments. The parent scheduler writes one request and
starts one hidden worker subprocess per model with `sys.executable`; the
worker calls the shared single-model pipeline. Data lives in `dataset/` and
results live under `results/<model>/<run_id>/`.

## Core capabilities

- one or many models through `train --model ...`;
- isolated model processes, per-model logs and atomic run status;
- fail-closed new runs, checkpoint resume, archive-based overwrite and ID suffixes;
- complete epoch checkpoint state, deterministic resume and independent A/B repeatability;
- shared training/evaluation timing, throughput, parameter and GPU-memory metrics;
- generated command reference and documentation consistency tests.

## Known limitations

Formal training still requires the two local protocol-named parquet files in
`dataset/`. The current public configuration keeps AMP CUDA-only and the
loader materializes the cleaned SDWPF arrays in memory. No formal GPU claim is
made without running the corresponding command on a CUDA machine.

## Next step

Add a model directory and structure YAML, implement only
`build_model(model_config, data_info)`, then run the generated command checks
and the relevant tests.

## Keep out of the project

Do not add a second CLI, model-specific training infrastructure, batch alias,
`--models`, StudySpec/ModelSpec, certificates, declarations, manifests,
readiness protocols, or experiment/report Markdown files.
