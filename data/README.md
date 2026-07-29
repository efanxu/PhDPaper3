# Local data

Place the two protocol-named SDWPF parquet files here:

- `sdwpf_model_input_base.parquet`
- `sdwpf_eval_target.parquet`

The input parquet contributes only the configured historical feature columns.
The target parquet supplies `Patv_raw` and `valid_target_mask` to the shared
Trainer/Evaluator boundary. Raw data and generated parquet files are ignored
by Git. The current local files were prepared by the existing
`dataset/clean_sdwpf_data.py` script and are not runtime dependencies on the
old project.
