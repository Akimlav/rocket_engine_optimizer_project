# Rocket Nozzle Optimizer

Parametrized rocket nozzle analytical solver and Rao optimum shape optimizer.
Part of the `rocket_engine_optimizet_project` stack — results feed directly into **LLM_based_DB**.

---

## Project Structure

```
rocket_nozzle/
├── nozzle_analysis.py        # 1D isentropic nozzle solver (physics engine)
├── moc.py                    # Rao optimum contour generator (Hermite + arc)
├── nozzle_optimizer.py       # Closed-loop shape optimizer (DE + Nelder-Mead)
├── post_processor/
│   └── process_nozzle.py     # Copies results -> LLM_based_DB/raw/
├── results/
│   └── <run_name>/
│       ├── log.nozzle_<run_name>       # Markdown + YAML frontmatter (LLM input)
│       ├── results.json                # Structured performance data
│       ├── contour.csv                 # Wall coordinates (optimizer only)
│       ├── optimization_history.csv    # Per-eval log (optimizer only)
│       ├── nozzle_plots.png            # Analysis plots
│       └── optimizer_plots.png         # Optimizer plots
└── README.md
```

---

## Dependencies

```bash
pip install numpy scipy matplotlib pandas
```

Python 3.11+. No GPU required.

---

## 1. Analytical Solver — `nozzle_analysis.py`

1D isentropic nozzle: straight-cone diverging section.

### Usage

```bash
# Default run
python nozzle_analysis.py

# Custom parameters
python nozzle_analysis.py \
  --name lox_h2 \
  --gamma 1.2 \
  --R 461.5 \
  --T0 3500 \
  --P0 7e6 \
  --Pe 101325 \
  --throat_radius 0.04 \
  --half_angle_deg 12
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--name` | nozzle_run | Run identifier (used for output folder) |
| `--gamma` | 1.4 | Specific heat ratio |
| `--R` | 287.0 | Gas constant [J/kg/K] |
| `--T0` | 3000 | Chamber stagnation temperature [K] |
| `--P0` | 5e6 | Chamber stagnation pressure [Pa] |
| `--Pe` | 101325 | Exit pressure [Pa] |
| `--throat_radius` | 0.05 | Throat radius [m] |
| `--half_angle_deg` | 15.0 | Diverging cone half-angle [deg] |
| `--n_points` | 200 | Profile discretization points |

### Assumptions

- 1D steady isentropic flow (no friction, no heat transfer, ideal gas)
- Flow choked at throat (M=1 enforced)
- Fully supersonic diverging section (no shocks)
- Equivalent OpenFOAM BCs: `inletTotalPressure` + `inletTotalTemperature` inlet, `fixedValue` pressure outlet, slip walls

---

## 2. Shape Optimizer — `nozzle_optimizer.py`

Optimizes the diverging wall contour using the **Rao parabolic (TIC) method**:
- Circular arc throat section (radius = 0.382·Rt, standard)
- Hermite cubic wall from inflection point to exit

Maximizes **vacuum Isp** including divergence loss correction:
`λ = 0.5 · (1 + cos(θ_exit))`

### Usage

```bash
python nozzle_optimizer.py --name rao_01 --n_eval 60

# Full custom run
python nozzle_optimizer.py \
  --name lox_h2_rao \
  --gamma 1.2 \
  --R 461.5 \
  --T0 3500 \
  --P0 7e6 \
  --Pe 101325 \
  --throat_radius 0.04 \
  --max_length_m 0.5 \
  --n_eval 200
```

### Parameters

| Flag | Default | Description |
|---|---|---|
| `--name` | rao_opt_01 | Run identifier (must be unique) |
| `--gamma` | 1.4 | Specific heat ratio |
| `--R` | 287.0 | Gas constant [J/kg/K] |
| `--T0` | 3000 | Chamber temperature [K] |
| `--P0` | 5e6 | Chamber pressure [Pa] |
| `--Pe` | 101325 | Exit pressure [Pa] |
| `--throat_radius` | 0.05 | Throat radius [m] |
| `--max_length_m` | 0.6 | Maximum allowed nozzle length [m] (constraint) |
| `--n_eval` | 60 | Optimization evaluation budget |

### Free variables optimized

| Variable | Bounds | Description |
|---|---|---|
| `theta_i` | 15–45 deg | Wall angle at inflection point |
| `theta_e` | 2–15 deg | Wall angle at nozzle exit |
| `n_lines` | 5–20 | Arc resolution (integer) |

### Strategy

1. `scipy.optimize.differential_evolution` — global search
2. `scipy.optimize.minimize` Nelder-Mead — local polish
3. Compare against straight 15° cone baseline

### Expected improvement

~1.5–2% Isp gain over straight cone (divergence loss reduction).
Typical result: `θ_e ≈ 2–5°`, `θ_i ≈ 25–40°`.

---

## 3. Feed Results to LLM_based_DB

The post-processor copies `log.nozzle_<name>` → `LLM_based_DB/raw/<name>_report.md`,
which `wiki_maintainer.py` picks up automatically.

```bash
# Single run
python post_processor/process_nozzle.py --run_dir results/rao_01

# All existing results
python post_processor/process_nozzle.py

# Watch mode — auto-feeds every new run
python post_processor/process_nozzle.py --watch
```

### Full pipeline (3 terminals)

```bash
# Terminal 1: LLM server
cd ../LLM_based_DB && bash llama_server.sh

# Terminal 2: Wiki maintainer
cd ../LLM_based_DB && python agents/wiki_maintainer.py

# Terminal 3: Run optimizer + feed
cd rocket_nozzle
python nozzle_optimizer.py --name rao_01 --n_eval 150
python post_processor/process_nozzle.py --run_dir results/rao_01

# Query the result
cd ../LLM_based_DB && python agents/query.py "What is the Isp of rao_01?"
```

---

## Output File Reference

### `log.nozzle_<name>` (LLM_based_DB input)

Markdown file with YAML frontmatter. Key frontmatter fields:

```yaml
solver: rocket_nozzle_optimizer   # or rocket_nozzle_analytical
run_name: rao_01
best_Isp_vacuum_s: 219.50
best_Cf_vac: 1.0423
best_theta_i_deg: 32.5
best_theta_e_deg: 3.1
cone_baseline_Isp_vac_s: 215.76
improvement_over_cone_pct: 1.73
```

### `results.json`

Structured performance data. All NaN/Inf replaced with `null` for valid JSON.

### `contour.csv`

Wall coordinates for CFD mesh generation:
```
x_m,r_m
0.00096,0.05000
0.00201,0.05043
...
```

### `optimization_history.csv`

One row per evaluation — useful for convergence analysis:
```
eval_id,theta_i,theta_e,n_lines,Isp_vac,length_m,Cf_vac
0,30.0,8.0,12,217.3,0.221,1.031
...
```

---

## Common Gas Properties

| Propellant | gamma | R [J/kg/K] | T0 typical [K] |
|---|---|---|---|
| Air | 1.40 | 287.0 | — |
| LOX/H2 | 1.20 | 461.5 | 3500 |
| LOX/RP-1 | 1.24 | 330.0 | 3600 |
| N2O4/UDMH | 1.25 | 320.0 | 3100 |
| Cold N2 (test) | 1.40 | 296.8 | 300 |
