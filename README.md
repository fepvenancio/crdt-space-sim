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

**Fair comparison** with centralized baseline using command buffering:

| Scenario | CRDT | Centralized | Winner |
|----------|------|-------------|--------|
| LEO (95% reliable, 1 step latency) | 165 steps | 87 steps | Centralized |
| GEO (90% reliable, 3 step latency) | 153 steps | 106 steps | Centralized |
| Lunar (80% reliable, 10 step latency) | 146 steps | 172 steps | **CRDT** |
| Mars (70% reliable, 100 step latency) | 301 steps | 1000+ (timeout) | **CRDT** |

**Key finding**: CRDT coordination advantage emerges when latency and partitions dominate. In good comms conditions (LEO/GEO), centralized coordination is actually more efficient. This crossover point is the honest reality.

*Tested with 5 robots, 10 tasks, fair command buffering for centralized baseline*

![Results Chart](simulation_results.png)

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
├── CLAUDE.md                 # Claude Code agent instructions
├── ROADMAP.md               # Development roadmap
├── PITCH.md                 # Cofounder/investor pitch
├── requirements.txt         # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── crdt/
│   │   ├── __init__.py
│   │   ├── state.py         # CRDT implementations
│   │   ├── robot.py         # Robot with CRDT
│   │   └── merge.py         # Merge operations
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── engine.py        # Simulation runner
│   │   ├── centralized.py   # Baseline comparison
│   │   └── scenarios.py     # Test scenarios
│   │
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── supervisor.py    # Safety monitoring
│   │   └── geofence.py      # Keep-out zones
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── charts.py        # Result charts
│       └── realtime.py      # Live visualization
│
├── tests/
│   ├── test_crdt.py         # CRDT unit tests
│   ├── test_merge.py        # Merge property tests
│   └── test_safety.py       # Safety tests
│
├── docs/
│   ├── technical.md         # Technical deep-dive
│   ├── crdt_primer.md       # CRDT explanation
│   └── space_context.md     # Space industry context
│
├── output/
│   ├── simulation_results.json
│   └── simulation_results.png
│
└── legacy/
    ├── simulation.py        # Original monolithic simulation
    └── visualize.py         # Original visualization
```

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/[your-username]/crdt-space-sim.git
cd crdt-space-sim

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Run simulation
python -m src.simulation.engine

# Generate charts
python -m src.visualization.charts
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

- **Email**: [your-email]
- **LinkedIn**: [your-linkedin]
- **Twitter**: [your-twitter]

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

*This project is in research/prototype phase. Not intended for production use.*
