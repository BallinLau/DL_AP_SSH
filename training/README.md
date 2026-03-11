# training

Training orchestration for episodes and modules.

## episode.py
`Episode` encapsulates a single training episode:
- `generate_data` (legacy) and `run_episode` (current flow)
- SDF, Policy/Value, FC2 losses and batch builders

### Episode flow
- Episode 0:
  1) Sample-based SDF/FC1 training (`build_sdf_fc1_df`)
  2) Sample-based Policy/Value training (`build_policy_value_df`)
  3) FC2 training using firm-level aggregation and Policy/Value outputs
- Episode >= 1:
  - Use `SimulateTS` to generate firm-time series, then train SDF/FC1, Policy/Value, FC2

### Batch builders
- `_create_sdf_batches_from_macro_df`: macro cross-section for SDF/FC1
- `_create_firm_batches_from_df`: firm triplets for Policy/Value losses
- `_create_fc2_batches`: parent/children firm groups + K for FC2

## trainer.py
`Trainer` orchestrates multi-episode training, checkpoints, and callbacks.

## scheduler.py
Learning rate and loss-weight schedulers used by `Episode`.

## gradient_utils.py
Gradient clipping and NaN protection.
