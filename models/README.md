# models

Neural network modules for the asset pricing system.

## policy_value.py
`PolicyValueModel` outputs firm-level decision/value objects.

### Input
Firm-state (7D):
```
(b, z, ETA, i, x, Hatcf, LnKF)
```

### Output (PolicyValueOutput)
- `Q`, `bp0`, `bpI`, `P0`, `PI`, `bar_i`, `bar_z`, `P`, `Phat`, `bp`

### Update rule
`update_leverage(b_old, bp, eta)`:
```
b_new = eta * bp + (1 - eta) * b_old
```

## sdf_fc1.py
`SDFFC1Combined` combines SDF, FC1, and value model `W`.

Key API:
- `forward_fc1(x)`: predict `(Hatcf, LnKF)` given `(x_prev, x_curr, Hatcf_prev, LnKF_prev)`
- `forward_step(...)`: returns `(w_prev, w_curr, M, hatcf_curr, lnkf_curr)`

This is used by SDF loss and macro proxy prediction.

## fc2.py
`FC2Model` maps cross-sectional distribution features to macro proxies.

Input:
- `phi = [b_quantiles(100), z_quantiles(100), x, lnK]` (default 202 dims)

Output:
- `(hatc, lnk)`

## share_layer.py
Shared backbone + heads used by policy/value to enforce structure
(e.g., no-invest heads do not depend on investment cost `i`).
