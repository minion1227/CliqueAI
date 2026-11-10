# CliqueAI

CliqueAI - AI-Powered Maximum Clique Solver Network

## Maximum Clique Mechanism
Four-stage autonomous mechanism for distributed problem solving

### Problem Selection
Advanced AI algorithms curate complex graph problems from our distributed database. The system intelligently categorizes challenges by difficulty, graph structure, and computational requirements.

### Miner Selection
Smart allocation engine filters eligible miners and samples each miner with the same difficulty-adjusted probability, keeping problem distribution fair while scaling participation by challenge difficulty.

### Scoring
Dual-metric evaluation system assesses both solution optimality and algorithmic diversity. This approach rewards accuracy while encouraging innovative problem-solving methodologies.

### Weight Setting
Exponential moving average algorithms continuously adjust miner reputation scores. Historical performance data influences future problem allocation and reward distribution.

## Documentation
- [Mechanism](docs/mechanism.md) – Detailed explanation of our mechanism.
- [W&B Logging](docs/wandb_logging.md) – Data logging schema for monitoring, analysis, and debugging via Weights & Biases.

## Getting Started
### Miner
```
./start_miner.sh --wallet.name <coldkey-name> --wallet.hotkey <hotkey-name> --subtensor.network finney --netuid 83 --logging.info --axon.ip <your-miner-ip> --axon.port <your-miner-port>
```

#### Clique solver

The miner answers with a compiled local-search solver ([fast_solver](CliqueAI/clique_algorithms/fast_solver)). `start_miner.sh` builds it with `g++`/`clang++`, or with the pip `ziglang` toolchain when no system compiler exists. Each request:

- runs in its own subprocess until shortly before the validator's timeout, so overlapping requests solve in parallel;
- collects every distinct maximum clique it visits and returns a rarely-visited one, because identical answers share the diversity score;
- always returns a *maximal* clique (validators score non-maximal cliques 0).

Options: `--solver.threads` (0 = all cores, split across overlapping requests), `--solver.safety_margin` (seconds reserved for the network, default 1.0 plus 5% of the timeout), `--solver.pick rarest|random`, `--solver.schedule R,P,G` (local-search phase mix; the default random-only selection measured fastest on CliqueAI's random graphs).

This miner defaults to `--neuron.autoupdate 0`: auto-update pulls `origin/main` and restarts whenever your HEAD differs from it, which would restart-loop a modified miner. Watch upstream releases yourself. The miner refuses to start if the solver cannot be built or fails its self-test.

```
python -m pytest tests/test_fast_solver.py   # correctness against the validator's own checks
python scripts/bench_solver.py               # solution quality at real tier sizes and budgets
```

### Validator
```
./start_validator.sh --wallet.name <coldkey-name> --wallet.hotkey <hotkey-name> --subtensor.network finney --netuid 83 --logging.info --axon.ip <your-validator-ip> --axon.port <your-validator-port>
```
