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

### 3. Validated Benchmark (100 trials, Real NASA Parameters)

```bash
python benchmark_validated.py   # 100-trial analysis with real physics
python benchmark_stats.py       # Statistical analysis (simplified model)
```

**Validated benchmark** using real NASA communication parameters:
- **Speed-of-light latency**: Earth-Moon RTT = 2.56s, Earth-Mars RTT = 25-45 min
- **Real blackout durations**: Lunar far-side = 30-50 min, Mars conjunction = hours
- **Tsiolkovsky fuel model**: Isp=250s, delta-v per task = 8 m/s
- **100 trials per scenario with statistical significance testing**

#### Mission Completion Time (hours)

| Scenario | CRDT | Centralized | Difference | Winner |
|----------|------|-------------|------------|--------|
| LEO | 0.54 ± 0.15 | 0.24 ± 0.03 | -119% | Centralized |
| Lunar | 0.52 ± 0.14 | 0.25 ± 0.03 | -104% | Centralized |
| **Mars Nominal** | **0.53 ± 0.16** | **8.00 (timeout)** | **+93%** | **CRDT** |
| **Mars Conjunction** | **0.53 ± 0.12** | **8.00 (timeout)** | **+93%** | **CRDT** |

#### Fuel Consumption (kg)

| Scenario | CRDT | Centralized | Difference |
|----------|------|-------------|------------|
| LEO | 27.7 ± 10.0 | 7.3 ± 0.0 | -277% (CRDT uses more) |
| Lunar | 25.4 ± 9.0 | 7.3 ± 0.0 | -245% (CRDT uses more) |
| Mars Nominal | 23.6 ± 10.2 | 4.0 ± 0.1 | -488% (CRDT uses more) |
| Mars Conjunction | 21.5 ± 6.9 | 3.9 ± 0.3 | -447% (CRDT uses more) |

**Key findings:**
1. **Mars is the killer app**: Centralized **times out at 8 hours** (25+ min RTT makes ground control impractical)
2. **Time vs Fuel trade-off**: CRDT completes faster at Mars but uses **3-5x more fuel** due to duplicate work
3. **LEO/Lunar favor centralized**: Low latency = ground control is more efficient
4. **Duplicate work overhead**: CRDT robots work on same tasks before syncing

**Honest interpretation**: CRDT coordination wins at Mars+ distances where centralized times out. However, the fuel cost is significant. For fuel-critical missions, hybrid approaches may be needed.

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
- **Task allocation only**: Benchmark tests task coordination, not full robotics
- **Independent tasks**: No task dependencies or resource constraints
- **Simplified physics**: No collision detection, attitude control, or thermal constraints
- **CRDT overhead**: 3-5x fuel consumption due to duplicate work

**What the validated results show:**
- CRDT wins at Mars+ distances where centralized **times out** (25+ min RTT)
- Centralized wins at LEO/Lunar (low latency = efficient ground control)
- **Trade-off**: CRDT is faster but uses significantly more fuel
- Fuel-critical missions may need hybrid approaches

**Real NASA parameters used:**
- Earth-Moon RTT: 2.56s (speed of light)
- Earth-Mars RTT: 25-45 min (opposition to conjunction)
- Blackouts: Lunar far-side 30-50 min, Mars conjunction hours
- Fuel model: Tsiolkovsky equation, Isp=250s, 8 m/s delta-v per task

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
