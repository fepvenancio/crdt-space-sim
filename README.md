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
| LEO (perfect comms) | 145 steps | 91 steps | Centralized |
| **LEO + eclipse blackouts** | **123 steps** | **275 steps** | **CRDT (55% faster)** |
| Lunar | 146 steps | 172 steps | **CRDT** |
| Mars | 301 steps | 1000+ (timeout) | **CRDT** |

**Key finding**: CRDT wins when blackouts/partitions occur—even in LEO. The ISS experiences ~45-minute eclipse periods every 90-minute orbit. During these blackouts, centralized control fails while CRDT robots keep working.

This means CRDT coordination is valuable for:
- **ISS/space station maintenance** (eclipse blackouts)
- **Lunar operations** (Earth-Moon latency + far-side blackouts)
- **Mars missions** (20+ minute latency, solar conjunction)

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
