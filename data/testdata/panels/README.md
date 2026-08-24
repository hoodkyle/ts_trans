# Synthetic monthly panel datasets

These deterministic fixtures contain 100 panels and 240 monthly observations
per panel from `2000-01` through `2019-12`. The generator is
`scripts/generate_panel_test_data.py` and uses seed `20260824`.

- `linear_panel.csv`: `x = b + m * t`; parameters are in `linear_parameters.csv`.
- `sinusoidal_panel.csv`: `x = A * sin(2πt / wavelength + phase)`; parameters are in `sinusoidal_parameters.csv`.
- `ar1_panel.csv`: the pre-sample state is `x_(-1) = alpha / (1 - rho)`, then
  `x_t = alpha + rho * x_(t-1) + epsilon_t`. Therefore the first observed
  value is `x_0 = alpha / (1 - rho) + epsilon_0`; parameters are in
  `ar1_parameters.csv`.

All files use the canonical `panel,date,value` long-form format at monthly
frequency. Each dataset has 24,000 observations and produces 22,800 windows
with `window_length=12`.
