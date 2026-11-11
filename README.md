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
- collects every distinct maximum clique it visits, plus maximal cliques one vertex smaller;
- always returns a *maximal* clique (validators score non-maximal cliques 0).

Which clique it returns: the reward is `optimality * (1 + difficulty) + diversity`, and diversity is split between miners returning the identical clique. When the search finds only a few maximum cliques (`--solver.near_max_cliques`, default 4), the rest of the field almost surely returns those same cliques, so the miner returns the least-visited maximal clique one vertex smaller instead: practically unique, and it loses little optimality when few miners reach the maximum. Otherwise it returns the least-visited maximum clique. Tune the threshold on real rounds with `scripts/replay_rounds.py` (below); `0` always returns a maximum clique.

Both protocols are served: `MaximumCliqueOfLambdaGraph` (current validators) and upstream v0.0.18's commit-reveal `MaximumCliqueSealedGraph` ([commit_reveal](CliqueAI/commit_reveal)), so the miner keeps answering when validators upgrade. Answers do not echo the graph back (smaller, faster responses). Only hotkeys with a validator permit may query the miner (`--no-blacklist.force_validator_permit` to accept any registered hotkey), as in v0.0.18.

Options: `--solver.threads` (0 = all cores, split across overlapping requests), `--solver.safety_margin` (seconds reserved for the network, default 1.0 plus 5% of the timeout), `--solver.near_max_cliques N`, `--solver.pick rarest|random`, `--solver.schedule R,P,G` (local-search phase mix; the default random-only selection measured fastest on CliqueAI's random graphs).

This miner defaults to `--neuron.autoupdate 0`: auto-update pulls `origin/main` and restarts whenever your HEAD differs from it, which would restart-loop a modified miner. Watch upstream releases yourself. The miner refuses to start if the solver cannot be built or fails its self-test. Each answer is logged as `Maximum clique found: size=... answer=max|near ... validator=uid... protocol=...`.

```
python -m pytest tests/                      # solver, policy, both protocols, blacklist, a real axon
python scripts/bench_solver.py               # solution quality at real tier sizes and budgets
python scripts/replay_rounds.py --rounds 150 --hotkey <hotkey> --save replay.jsonl
                                             # rescore real logged rounds (needs `pip install wandb`)
python scripts/replay_rounds.py --load replay.jsonl   # re-analyse; sweeps --solver.near_max_cliques
```

### Validator
```
./start_validator.sh --wallet.name <coldkey-name> --wallet.hotkey <hotkey-name> --subtensor.network finney --netuid 83 --logging.info --axon.ip <your-validator-ip> --axon.port <your-validator-port>
```
