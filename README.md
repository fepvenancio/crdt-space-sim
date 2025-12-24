# CRDT Space Robotics Hub

> **Distributed autonomous coordination for space robots using Conflict-free Replicated Data Types**

[![Status](https://img.shields.io/badge/status-research%20prototype-yellow)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## The Problem

Space robots today rely on **centralized ground control**. With communication delays (0.5s to GEO, 1.3s to Moon, 20+ min to Mars) and unreliable links, this creates critical bottlenecks:

- Robots idle waiting for commands
- Single point of failure
- Expensive ground station time
- Can't operate during blackouts (eclipse, lunar far-side, solar conjunction)

## Our Solution

**CRDT-coordinated autonomous robot swarms** that:

- Operate independently with local decision-making
- Sync state when communications allow
- **Mathematically guaranteed** to converge without conflicts
- Continue working during communication blackouts

```
Traditional:                    Our Approach:

Ground <--> Robot              Robot <--> Robot
  |                               |         |
  v                               v         v
Ground <--> Robot              Robot <--> Robot

No ground dependency
Partition tolerant
Eventually consistent
```

---

## Live Simulations

### 1. Lunar Orbital Refueling Depot

```bash
python orbital_simulation.py
```

**Scenario**: 5 autonomous servicing robots coordinate to refuel client spacecraft at a Lunar Gateway-style depot.

- **3D visualization** with real orbital mechanics (two-body problem, RK45 integration)
- **Realistic physics**: 100km lunar orbit (~2 hour period), 50N thrusters, Tsiolkovsky fuel consumption
- **Communication blackouts**: Robots lose contact when behind the Moon (8% partition probability)
- **CRDT coordination**: First-write-wins task claiming prevents two robots servicing the same client

| Parameter | Value |
|-----------|-------|
| Orbital altitude | 100 km |
| Orbital period | ~7138 seconds (~2 hours) |
| Thrust | 50N (realistic for proximity ops) |
| Comm reliability | 80% |
| Partition events | ~8% probability per sync |

### 2. Lunar Ground Operations

```bash
python ground_simulation.py
```

**Scenario**: 5 rovers coordinate construction tasks at a lunar base (solar panels, regolith covering, resource transport).

- **2D top-down view** with procedural terrain (craters, hills)
- **Line-of-sight communication**: Hills block rover-to-base and rover-to-rover comms
- **Peer-to-peer sync**: Rovers out of base range can still sync with nearby rovers
- **CRDT task claiming**: Prevents multiple rovers driving to same construction site

| Parameter | Value |
|-----------|-------|
| Base comm range | 5 km |
| Rover speed | 5 m/s (18 km/h) |
| Terrain | 20x20 km with 15 craters, 10 hills |
| Comm reliability | 85% |

### 3. Statistical Benchmark (100 trials)

```bash
python benchmark_stats.py      # Full 100-trial statistical analysis
python -m src.simulation.engine  # Quick comparison
```

**Fair comparison** of CRDT vs centralized control with:
- Command buffering (5 commands per robot for centralized)
- Synchronized partition events (identical timing for both)
- Same message success/failure sequences
- **100 trials per scenario with statistical significance testing**

| Scenario | CRDT (steps) | Centralized | Difference | p-value | Winner |
|----------|--------------|-------------|------------|---------|--------|
| LEO | 164 ± 51 | 86 ± 11 | -90% | <0.001*** | Centralized |
| LEO + Eclipse | 158 ± 47 | 102 ± 16 | -55% | <0.001*** | Centralized |
| Lunar | 154 ± 41 | 156 ± 35 | **+1%** | 0.80 | **Tie** |
| **Mars** | **252 ± 63** | **1000 (timeout)** | **+75%** | **<0.001***** | **CRDT** |

*\* p<0.05, \*\* p<0.01, \*\*\* p<0.001 (statistically significant)*

**Key findings:**
1. **Mars is the killer app**: CRDT completes 75% faster (p<0.001)
2. **Lunar is a statistical tie**: No significant difference at 80% reliability
3. **LEO favors centralized**: Low latency + high reliability = ground control wins
4. **CRDT overhead**: ~100% duplicate work at Lunar distances (robots work on same tasks before syncing)

**Honest interpretation**: CRDT coordination only shows clear advantage at Mars+ distances where centralized control times out. At Lunar distances, the approaches are equivalent.

---

## CRDT Data Structures

| Structure | Type | Purpose |
|-----------|------|---------|
| Completed tasks | G-Set | Grows only, tasks never "uncomplete" |
| Task progress | G-Counter | Increments only, per robot |
| Robot positions | LWW-Register | Last-write-wins with timestamps |
| Task claims | First-Write-Wins | Prevents duplicate work |

**Properties guaranteed after any merge:**
- Completed tasks never decrease
- Progress never decreases
- Merge is idempotent (merging same state twice = no change)
- Merge is commutative (order doesn't matter)
- Merge is associative (grouping doesn't matter)

---

## Project Structure

```
crdt-space-sim/
├── orbital_simulation.py    # 3D orbital refueling demo
├── ground_simulation.py     # 2D ground operations demo
├── benchmark_stats.py       # 100-trial statistical analysis
├── README.md
├── CLAUDE.md                # Development guidelines
├── requirements.txt
│
├── src/
│   ├── crdt/
│   │   └── state.py         # CRDT implementations (core IP)
│   └── simulation/
│       └── engine.py        # Benchmark comparison engine
│
└── tests/
    └── test_crdt.py         # 22 property tests
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/fepvenancio/crdt-space-sim.git
cd crdt-space-sim

# Setup
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Run simulations
python orbital_simulation.py   # 3D orbital (interactive)
python ground_simulation.py    # 2D ground (interactive)

# Run statistical benchmark (100 trials, ~2 min)
python benchmark_stats.py

# Run tests
pytest tests/ -v
```

---

## Current Status

- [x] Core CRDT implementation (G-Set, G-Counter, LWW-Register, FWW-Map)
- [x] Fair benchmark comparison (centralized has command buffering)
- [x] **100-trial statistical analysis with p-values**
- [x] 3D orbital simulation with real physics
- [x] 2D ground simulation with terrain/LoS
- [x] Unit tests (22/22 passing)
- [ ] Safety supervisor
- [ ] ROS2 integration

## Known Limitations

This is a **proof of concept** with honest limitations:

**Benchmark limitations:**
- **Task allocation only**: Benchmark tests task coordination, not full robotics
- **Independent tasks**: No task dependencies or resource constraints
- **No physics**: No collision, fuel, or mass constraints in benchmark
- **CRDT overhead**: ~100% duplicate work at Lunar distances

**What the results actually show:**
- CRDT wins decisively only at Mars+ distances (100+ step latency)
- Lunar is a statistical tie (p=0.80, not significant)
- Centralized wins at LEO with good comms

The orbital/ground simulations add realistic physics but are visual demos, not rigorous benchmarks.

---

## Target Market

**On-Orbit Servicing (OOS)**: $4.4B by 2030
- Satellite life extension
- Debris removal
- Space station maintenance
- Lunar/Mars base operations

---

## Looking For

**Technical Cofounder** with:
- Robotics background (PhD or industry)
- Space industry knowledge preferred
- Interest in distributed systems

## Contact

- **Email**: filipeepv@gmail.com
- **LinkedIn**: [linkedin.com/in/fven](https://www.linkedin.com/in/fven/)

---

MIT License - See [LICENSE](LICENSE)

*Research prototype - not for production use*
