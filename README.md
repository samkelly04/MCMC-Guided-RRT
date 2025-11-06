# RRT with Uniform & Metropolis-within-Gibbs Samplers
## Quickstart
- `uv venv && uv pip install -r requirements.txt` (or `pip`/`poetry`)
- Run tests: `pytest -q`
- Example run: `python experiments/run_experiment.py --config configs/baseline.yaml`

## Project vocabulary
- *Planning world*: inflated obstacles for path search.
- *Noisy world*: evaluation rollouts; Gaussian noise on obstacle size/position.
- *Robustness*: performance of planned path over 100 sims in noisy world.

## Common commands
- Format/lint: `ruff check --fix . && ruff format .`
- Generate a random map preview: `python -m src.env.preview --seed 7`
