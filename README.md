# CRDT Space Robotics Hub

> **Distributed autonomous coordination for on-orbit servicing robots using Conflict-free Replicated Data Types**

[![Status](https://img.shields.io/badge/status-research%20prototype-yellow)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 🚀 The Problem

Space robots today rely on **centralized ground control**. With communication delays (0.5s to GEO, 20+ minutes to Mars) and unreliable links, this creates critical bottlenecks:

- Robots idle waiting for commands
- Single point of failure
- Expensive ground station time
- Can't operate during blackouts (eclipse, solar events)

## 💡 Our Solution

**CRDT-coordinated autonomous robot swarms** that:

- Operate independently with local decision-making
- Sync state when communications allow
- **Mathematically guaranteed** to converge without conflicts
- Continue working during communication blackouts

```
Traditional:                    Our Approach:
                               
Ground ◄──► Robot              Robot ◄──► Robot
  │                               │         │
  ▼                               ▼         ▼
Ground ◄──► Robot              Robot ◄──► Robot
  │                               
  ▼                            No ground dependency
Ground ◄──► Robot              Partition tolerant
                               Eventually consistent
```

## 📊 Proof of Concept Results

**Fair comparison** with centralized baseline using:
- Command buffering (5 commands per robot)
- Synchronized partition events (identical timing for both approaches)
- Same message success/failure sequences
- Identical task completion criteria

| Scenario | CRDT | Centralized | Winner | Notes |
|----------|------|-------------|--------|-------|
| LEO (95% reliable) | ~150 steps | ~90 steps | Centralized | Good comms favor ground control |
| LEO + Eclipse (8% partition) | ~170 steps | ~100 steps | Centralized | Buffering handles short blackouts |
| **Lunar** (80% reliable, 10 step latency) | **~120 steps** | **~150 steps** | **CRDT (+18%)** | Crossover point |
| **Mars** (70% reliable, 100 step latency) | **~210 steps** | **1000+ (timeout)** | **CRDT (+79%)** | Ground control breaks down |

**Key finding**: The crossover point is at **Lunar distances**. When reliability drops below ~80% and latency exceeds ~10 round-trip steps, CRDT coordination outperforms even well-buffered centralized control.

**Why centralized wins in LEO/LEO_Eclipse:**
- High reliability (95%) means commands almost always arrive
- Low latency (1 step) allows fast recovery after partitions
- 5-command buffer sustains work during short blackouts

**Why CRDT wins at Lunar+ distances:**
- Lower reliability (80%) causes more failed messages
- Higher latency (10+ steps) makes ground unable to reassign tasks quickly
- Robots continue autonomous work regardless of comms state
- ~26% duplicate work overhead is outweighed by zero idle time

This means CRDT coordination is valuable for:
- **Lunar operations** (Earth-Moon latency + far-side blackouts)
- **Mars missions** (20+ minute latency, solar conjunction)
- **Deep space missions** (hours of light-time delay)

*Tested with 5 robots, 10 tasks, fair command buffering for centralized baseline*

## 🎯 Target Market

**On-Orbit Servicing (OOS)**: $4.4B by 2030

- Satellite life extension
- Debris removal  
- Space station maintenance
- Constellation servicing

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORBITAL HUB                          │
│                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │ Robot 1 │  │ Robot 2 │  │ Robot n │   LIDAR        │
│  │  CRDT   │◄─┼─► CRDT  │◄─┼─► CRDT  │   + Vision    │
│  │  State  │  │  State  │  │  State  │                │
│  └─────────┘  └─────────┘  └─────────┘                │
│       │            │            │                      │
│       └────────────┴────────────┘                      │
│                    │                                   │
│         ┌─────────────────────┐                       │
│         │   CRDT Merge Layer  │                       │
│         │  • G-Set (tasks)    │                       │
│         │  • G-Counter (prog) │                       │
│         │  • LWW-Register     │                       │
│         └─────────────────────┘                       │
│                    │                                   │
│         ┌─────────────────────┐                       │
│         │  STOP-ALL BROADCAST │ ← Safety override     │
│         └─────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

## 🔬 CRDT Data Structures Used

| Structure | Type | Purpose |
|-----------|------|---------|
| Completed tasks | G-Set | Grows only, tasks never "uncomplete" |
| Task progress | G-Counter | Increments only, per robot |
| Robot positions | LWW-Register | Last-write-wins with timestamps |
| Task claims | First-write-wins | Prevents duplicate work |

## 📁 Project Structure

```
crdt-space-sim/
├── README.md                 # This file
├── CLAUDE.md                 # Development guidelines
├── PITCH.md                 # Cofounder pitch
├── requirements.txt         # Python dependencies
│
├── src/
│   ├── crdt/
│   │   ├── __init__.py
│   │   └── state.py         # CRDT implementations (core)
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── engine.py        # Fair comparison simulation
│   │
│   ├── safety/              # (placeholder)
│   └── visualization/       # (placeholder)
│
├── tests/
│   └── test_crdt.py         # CRDT property tests (22 tests)
│
├── output/
│   └── simulation_results.json
│
└── legacy/
    └── simulation.py        # Original prototype
```

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/fepvenancio/crdt-space-sim.git
cd crdt-space-sim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run simulation
python -m src.simulation.engine

# Run tests
pytest tests/ -v
```

## 🎯 Current Status

- [x] Core CRDT implementation (G-Set, G-Counter, LWW-Register, FWW-Map)
- [x] Fair simulation comparison (centralized has command buffering)
- [x] Latency and partition modeling
- [x] Unit tests (22/22 passing)
- [ ] Hardware failure modeling
- [ ] Safety supervisor
- [ ] ROS2 integration
- [ ] 3D visualization

## ⚠️ Known Limitations

This is a **proof of concept** with the following simplifications:

- **No physics**: No fuel consumption, collision detection, or mass constraints
- **Discrete time**: Simulation uses discrete steps, not continuous dynamics
- **Perfect sensing**: No sensor noise or localization error
- **Simple tasks**: Tasks have instant start and linear progress
- **No clock skew**: All robots have synchronized clocks

These limitations are documented to be honest with potential technical cofounders. The goal is to prove the CRDT coordination concept, not build a high-fidelity simulator.

## 🤝 Looking For

**Technical Cofounder** with:
- Robotics background (PhD or industry experience)
- Space industry knowledge preferred
- Interest in distributed systems
- Willingness to work for equity initially

**What I Bring**:
- Distributed systems expertise (Web3 background)
- Working proof of concept
- Vision for the product
- Business development

## 📬 Contact

- **Email**: filipeepv@gmail.com
- **LinkedIn**: [linkedin.com/in/fven](https://www.linkedin.com/in/fven/)

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

*This project is in research/prototype phase. Not intended for production use.*
