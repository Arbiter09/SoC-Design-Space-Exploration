# AI-Driven SoC Design Space Exploration Platform

Automated platform that generates, simulates, synthesises, and ML-predicts
PPA (Power, Performance, Area) across 100+ RTL configurations of an SoC design.
A Graph Neural Network replaces expensive synthesis runs for early screening, and
a two-stage failure triage engine classifies RTL bugs automatically from
simulation artefacts.

---

## Architecture

```
YAML param space
      │
      ▼
generator.py  ──▶  configs/<id>/rtl/{pipeline,cache,soc_top}.v
      │
      ├──▶  synth_runner.py  ──▶  Yosys  ──▶  netlist.json  ──▶  synthesis_results (SQLite)
      │                                              │
      │                                     netlist_parser.py  ──▶  graphs/<id>.json
      │                                              │
      │                                       dataset.py (PyG)
      │                                              │
      │                                      gnn.py / train.py / predict.py
      │
      └──▶  sim_runner.py   ──▶  Verilator  ──▶  sim.vcd / sim.log  ──▶  simulation_results (SQLite)
                                                         │
                                              vcd_parser.py + classifier.py  ──▶  triage_report.json
```

---

## Tool Installation

### macOS (Homebrew)

```bash
brew install yosys verilator
```

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y yosys verilator build-essential g++
```

### From source (latest versions)

```bash
# Yosys
git clone https://github.com/YosysHQ/yosys
cd yosys && make -j$(nproc) && sudo make install

# Verilator
git clone https://github.com/verilator/verilator
cd verilator && autoconf && ./configure && make -j$(nproc) && sudo make install
```

---

## Python Setup

```bash
# Clone the repo
git clone https://github.com/Arbiter09/SoC-Design-Space-Exploration.git
cd SoC-Design-Space-Exploration

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r soc_dse/requirements.txt

# Install the package (editable)
pip install -e .
```

---

## Quickstart: 10-config smoke test

```bash
# 1. Generate 10 RTL configurations (random sample from the full 162-config grid)
python -m soc_dse.backend.generator --sample 10 --seed 42

# 2. Run synthesis + simulation in parallel (4 workers)
python -m soc_dse.backend.run_pipeline --sample 10 --seed 42 --workers 4

# 3. Parse netlists into graph JSON files
python -m soc_dse.backend.netlist_parser

# 4. Train the GNN (5 epochs for a sanity check)
python -m soc_dse.model.train --epochs 5

# 5. Triage any failed simulations
python -m soc_dse.triage.classifier

# 6. Generate visualisation plots → soc_dse/viz/output/
python -m soc_dse.viz.plot
```

---

## Full Design Space Sweep (162 configs)

```bash
# Full grid: 3×3×3×3×2 = 162 configs
python -m soc_dse.backend.run_pipeline --workers 8

# Parse all netlists
python -m soc_dse.backend.netlist_parser

# Train GNN for 100 epochs (target: mean R² ≥ 0.75)
python -m soc_dse.model.train --epochs 100 --batch-size 16

# Predict PPA for a specific config (with MC Dropout CI)
python -m soc_dse.model.predict soc_dse/graphs/<config_id>.json

# Generate all plots
python -m soc_dse.viz.plot
```

---

## Parameter Space

Edit `soc_dse/configs/param_space.yaml` to define the design space:

```yaml
pipeline_stages: [3, 5, 7]
cache_size_kb:   [16, 32, 64]
alu_units:       [1, 2, 4]
memory_banks:    [1, 2, 4]
bus_width:       [32, 64]
```

Each combination gets a deterministic 8-character config ID:
`sha256(json(sorted(params)))[:8]`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SOC_DB_PATH` | `soc_dse/dse.db` | SQLite database path |
| `SOC_CONFIGS_DIR` | `soc_dse/configs/` | Root for generated RTL configs |
| `SOC_TEMPLATES_DIR` | `soc_dse/templates/` | Jinja2 Verilog template directory |
| `SOC_GRAPHS_DIR` | `soc_dse/graphs/` | Output directory for graph JSON files |
| `SOC_TESTBENCH` | `soc_dse/testbench/testbench.cpp` | Verilator C++ testbench |
| `SOC_VIZ_DIR` | `soc_dse/viz/output/` | Plot output directory |
| `YOSYS_BIN` | `yosys` | Path to Yosys binary |
| `VERILATOR_BIN` | `verilator` | Path to Verilator binary |

---

## Project Structure

```
soc_dse/
├── configs/
│   ├── param_space.yaml           # Parameter space definition
│   └── <config_id>/
│       ├── params.yaml            # Config parameters
│       ├── rtl/                   # Generated Verilog (gitignored)
│       ├── netlist.json           # Yosys JSON netlist (gitignored)
│       ├── synth.ys               # Yosys script
│       ├── synth.log
│       ├── sim.vcd                # VCD waveform dump (gitignored)
│       ├── sim.log
│       └── triage_report.json
├── templates/
│   ├── pipeline.v.j2              # Parametric pipeline Verilog
│   ├── cache.v.j2                 # Parametric cache Verilog
│   └── soc_top.v.j2              # Top-level SoC integration
├── testbench/
│   └── testbench.cpp              # Verilator C++ testbench
├── graphs/
│   └── <config_id>.json           # Graph data for GNN (gitignored)
├── backend/
│   ├── db.py                      # SQLite schema + helpers
│   ├── generator.py               # RTL config generator (Jinja2)
│   ├── synth_runner.py            # Yosys automation
│   ├── sim_runner.py              # Verilator automation
│   ├── netlist_parser.py          # Yosys JSON → NetworkX → graph JSON
│   └── run_pipeline.py            # Parallel orchestrator
├── model/
│   ├── dataset.py                 # PyTorch Geometric dataset
│   ├── gnn.py                     # GraphSAGE architecture
│   ├── train.py                   # Training loop
│   ├── predict.py                 # Inference + MC Dropout CI
│   └── checkpoints/
│       └── best.pt                # Best model checkpoint (gitignored)
├── triage/
│   ├── vcd_parser.py              # Two-pass VCD parser
│   └── classifier.py              # Rule-based + TF-IDF/LR classifier
├── viz/
│   ├── plot.py                    # Pareto + scatter + GNN + heatmap
│   └── output/                    # Generated PNG + HTML plots
├── dse.db                         # SQLite database (gitignored)
└── requirements.txt
```

---

## GNN Model Details

- **Architecture**: 3-layer GraphSAGE encoder → global mean pool → 2-layer MLP
- **Node features** (19-dim): cell type one-hot (16) + fanin + fanout + logic depth
- **Targets**: area, delay, power_estimate (z-score normalised)
- **Training**: 70/15/15 split, Adam + ReduceLROnPlateau, MSE loss
- **Inference**: 30-sample MC Dropout for confidence intervals
- **Target accuracy**: mean R² ≥ 0.75 on test set

---

## Failure Triage

The triage engine classifies simulation failures into five categories:

| Label | Trigger |
|---|---|
| `pipeline_stall_deadlock` | stall / livelock / backpressure patterns |
| `cache_coherency_violation` | MESI / dirty / coherency patterns |
| `overflow_in_alu` | overflow / carry / wraparound patterns |
| `reset_sequencing_error` | rst_n / POR / initialization patterns |
| `bus_contention` | contention / arbitration / collision patterns |

Unknown failures fall back to a TF-IDF + Logistic Regression classifier
pre-trained on 80 synthetic labelled log snippets.

---

## CI/CD

GitHub Actions workflow (`.github/workflows/dse_pipeline.yml`):

1. Cache Yosys + Verilator apt packages (keyed on tool versions)
2. Install Python deps (pip cache keyed on `requirements.txt` hash)
3. Generate 10 RTL configs (smoke test)
4. Run Yosys synthesis
5. Parse netlists → graphs
6. Run Verilator simulation (best-effort)
7. **5-epoch GNN sanity check — asserts mean R² > 0.5**
8. Triage failed configs
9. Generate plots
10. Upload PNG/HTML plots and graph JSON as workflow artifacts

---

## License

MIT
