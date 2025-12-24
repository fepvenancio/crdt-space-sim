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

### 3. Validated Benchmark (100 trials, Physics-Based)

```bash
python benchmark_validated.py   # 100-trial analysis with real physics
python benchmark_stats.py       # Statistical analysis (simplified model)
```

**Physics-based benchmark** with validated parameters:
- **Speed-of-light latency**: Earth-Moon RTT = 2.56s, Earth-Mars RTT = 25-45 min
- **Real blackout durations**: Lunar far-side = 30-50 min, Mars conjunction = hours
- **Tsiolkovsky fuel model**: Isp=230s, 0.5 m/s cruise velocity, ~55g per burn
- **100 trials per scenario with statistical significance testing**

#### Mission Completion Time (hours)

| Scenario | CRDT | Centralized | Difference | Winner |
|----------|------|-------------|------------|--------|
| LEO | 0.54 ± 0.14 | 0.24 ± 0.03 | -122% | Centralized |
| Lunar | 0.56 ± 0.15 | 0.25 ± 0.03 | -120% | Centralized |
| **Mars Nominal** | **0.62 ± 0.17** | **8.00 (timeout)** | **+92%** | **CRDT** |
| **Mars Conjunction** | **0.65 ± 0.18** | **8.00 (timeout)** | **+92%** | **CRDT** |

#### Fuel Consumption (kg) - Physics-Based

| Scenario | CRDT | Centralized | Notes |
|----------|------|-------------|-------|
| LEO | 4.17 ± 1.38 | 1.11 ± 0.00 | Both complete; CRDT uses 4x more |
| Lunar | 4.13 ± 1.38 | 1.11 ± 0.00 | Both complete; CRDT uses 4x more |
| Mars Nominal | 4.09 ± 1.23 | 0.59 ± 0.02 | **Centralized fails to complete** |
| Mars Conjunction | 4.32 ± 1.13 | 0.58 ± 0.04 | **Centralized fails to complete** |

**Key findings:**
1. **Mars missions require CRDT**: Centralized **times out** - cannot complete with 25+ min RTT
2. **CRDT enables missions that would otherwise fail**: 10 tasks completed in ~0.6h vs never
3. **LEO/Lunar trade-off**: Centralized is faster AND uses less fuel (4x efficiency)
4. **Physics explanation**: CRDT robots do ~4x more maneuvers (multiple robots approach same task before sync)

**Why CRDT uses more fuel (physics-based):**
- Each approach = 2 burns (accelerate + decelerate) = ~110g fuel
- CRDT: 5 robots independently start approaches → more total maneuvers
- Centralized: Ground assigns uniquely → exactly 1 robot per task
- Wasted fuel from aborted approaches: only 0.3-2.6% (most fuel is productive)

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
├── benchmark_validated.py   # 100-trial benchmark with NASA parameters
├── benchmark_stats.py       # Statistical analysis (simplified model)
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

# Run validated benchmark (100 trials, real NASA parameters, ~3 min)
python benchmark_validated.py

# Run simplified benchmark
python benchmark_stats.py

# Run tests
pytest tests/ -v
```

---

## Current Status

- [x] Core CRDT implementation (G-Set, G-Counter, LWW-Register, FWW-Map)
- [x] Fair benchmark comparison (centralized has command buffering)
- [x] **Validated benchmark with real NASA parameters**
- [x] **Tsiolkovsky fuel consumption model**
- [x] 3D orbital simulation with real physics
- [x] 2D ground simulation with terrain/LoS
- [x] Unit tests (22/22 passing)
- [ ] Safety supervisor
- [ ] ROS2 integration

## Known Limitations

This is a **proof of concept** with honest limitations:

**Benchmark limitations:**
- **Task allocation only**: Tests coordination, not full robotics (no arm control, docking)
- **Independent tasks**: No task dependencies or resource constraints
- **Simplified dynamics**: Fixed-speed movement, no orbital mechanics in benchmark
- **No sensor noise**: Perfect position knowledge assumed

**What the physics-based results show:**
- **Mars missions require CRDT**: Centralized cannot complete (times out at 8h)
- **LEO/Lunar favor centralized**: 4x fuel efficiency, 2x faster
- **CRDT overhead is real**: Multiple robots start toward same task before sync
- **Wasted fuel is small**: Only 0.3-2.6% from aborted approaches

**Physics model validation:**
- Earth-Moon RTT: 2.56s (speed of light, verified)
- Earth-Mars RTT: 25-45 min (speed of light, verified)
- Fuel: Tsiolkovsky equation with Isp=230s (hydrazine monoprop)
- Burns: ~55g per acceleration/deceleration at 0.5 m/s cruise
- Station keeping: 0.01 m/s/hour (LEO typical)

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
