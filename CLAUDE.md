# CLAUDE.md - Claude Code Agent Instructions

You are an expert Python developer specializing in simulations and 3D visualizations.

## Project Context

This is a **CRDT-based space robotics coordination system** proof of concept. The goal is to demonstrate that distributed CRDT coordination outperforms centralized ground control for multi-robot space operations.

**Current stage**: Research prototype seeking technical cofounder.

## Project Goals

1. **Prove technical thesis**: CRDT coordination > centralized control under unreliable comms
2. **Build credibility**: Clean code, tests, documentation for showing to potential cofounders
3. **Prepare for next stage**: NASA SBIR application, accelerator applications

## Tech Stack

- **Language**: Python 3.10+
- **Key Libraries**: 
  - `matplotlib` - visualization
  - `pytest` - testing
  - `pydantic` - data validation (add this)
  - Future: `rclpy` (ROS2), `fastapi` (API layer)
- **Style**: Type hints required, docstrings required

## Code Standards

### Always Follow

```python
# Type hints required
def merge(self, other: 'CRDTRobotState') -> None:
    """
    Merge another robot's state into this one.
    
    CRDT properties maintained:
    - Commutative: merge(A,B) == merge(B,A)
    - Associative: merge(merge(A,B),C) == merge(A,merge(B,C))
    - Idempotent: merge(A,A) == A
    
    Args:
        other: State from another robot to merge
    """
    pass
```

### File Organization

```
orbital_simulation.py  - 3D orbital refueling demo
ground_simulation.py   - 2D ground operations demo
src/crdt/              - CRDT implementations (core IP)
src/simulation/        - Benchmark comparison engine
tests/                 - CRDT property tests
```

### Naming Conventions

```python
# Classes: PascalCase
class CRDTRobotState:
    pass

# Functions/methods: snake_case
def merge_states(a: State, b: State) -> State:
    pass

# Constants: SCREAMING_SNAKE_CASE
MAX_ROBOTS = 100
SYNC_INTERVAL_MS = 100

# Private: leading underscore
def _internal_helper(self):
    pass
```

## Key Technical Concepts

### CRDT Types We Use

| Type | Name | Property | Use Case |
|------|------|----------|----------|
| G-Set | Grow-only Set | Add only | Completed tasks |
| G-Counter | Grow-only Counter | Increment only | Task progress |
| LWW-Register | Last-Write-Wins | Latest timestamp wins | Robot positions |
| FWW-Map | First-Write-Wins | First claim wins | Task assignments |

### Critical Invariants

```python
# These must ALWAYS be true after any merge operation:

# 1. Completed tasks never decrease
assert len(merged.completed_tasks) >= len(original.completed_tasks)

# 2. Progress never decreases
for task_id in original.task_progress:
    assert merged.get_task_progress(task_id) >= original.get_task_progress(task_id)

# 3. Merge is idempotent
state_copy = deepcopy(state)
state.merge(state)
assert state == state_copy
```

## Current Tasks Backlog

### Completed
- [x] Package structure with proper imports
- [x] CRDT property tests (22 tests, commutativity/associativity/idempotency)
- [x] 3D orbital simulation with real physics
- [x] 2D ground simulation with terrain/LoS
- [x] Type hints and docstrings
- [x] Fair benchmark comparison

### P0 - Next Up

1. **Implement SafetySupervisor**
   - Watchdog timer
   - Geofence checking
   - Collision prediction
   - E-stop broadcast

2. **Add hardware failure model**
   - SEU (single event upset) from radiation
   - Sensor noise
   - Additional comms latency

### P1 - Future

3. **ROS2 integration** - `rclpy` nodes for each robot
4. **Web visualization** - FastAPI + real-time dashboard
5. **Docker containerization**

## How To Help Me Build

When I ask you to work on this project:

### For New Features

1. Check this file for context
2. Follow the code standards above
3. Add tests for new code
4. Update docstrings
5. Keep CRDT invariants intact

### For Refactoring

1. Preserve all existing functionality
2. Run tests before and after
3. Keep the simulation runnable
4. Update imports as needed

### For Documentation

1. Target audience: potential technical cofounders
2. Assume robotics knowledge, explain CRDT concepts
3. Include code examples
4. Be concise but complete

### For Debugging

1. Check CRDT invariants first
2. Add logging if needed
3. Write a test that reproduces the bug
4. Fix and verify test passes

## Commands Reference

```bash
# Run 3D orbital simulation (interactive)
python orbital_simulation.py

# Run 2D ground simulation (interactive)
python ground_simulation.py

# Run benchmark comparison
python -m src.simulation.engine

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## Don't Do

- ❌ Don't break existing simulation functionality
- ❌ Don't remove type hints
- ❌ Don't merge without testing CRDT properties
- ❌ Don't use `print()` for logging - use `logging` module
- ❌ Don't hardcode values - use constants or config

## Do

- ✅ Write tests first when adding features
- ✅ Keep functions small and focused
- ✅ Document CRDT properties in docstrings
- ✅ Use descriptive variable names
- ✅ Add type hints to all new code

## Architecture Decisions

### Why Python (not Rust/C++)?

- Faster iteration for prototyping
- Easier for potential cofounders to understand
- Will port safety-critical code to Rust/C++ for production

### Why not ROS2 yet?

- Adds complexity for prototype stage
- Want to prove concept first
- Will add ROS2 integration after core is solid

### Why separate Safety module?

- Safety code must be independent of main logic
- Easier to audit
- Reflects real space systems architecture

## Files That Matter Most

```
src/crdt/state.py        - Core CRDT implementation (THE IP)
orbital_simulation.py    - 3D orbital demo with real physics
ground_simulation.py     - 2D ground demo with terrain
src/simulation/engine.py - Benchmark comparison
tests/test_crdt.py       - CRDT property tests (22 tests)
```

## Questions To Ask Me

If unclear about anything, ask:

1. "Should this preserve CRDT properties?"
2. "Is this safety-critical code?"
3. "Should I add tests for this?"
4. "What's the priority of this task?"

## Simulation Validity Requirements

### Fair Comparison Criteria (ALL IMPLEMENTED)
For the CRDT vs Centralized comparison to survive technical scrutiny:

1. **Synchronized partition events** - Both approaches experience IDENTICAL partition timing
2. **Same message success/failure sequence** - Pre-generated random outcomes used for both
3. **Same starting positions** - Pre-generated positions shared between runs
4. **Same completion criteria** - Both complete when actual work is done (not when knowledge syncs)
5. **Centralized has command buffering** - 5-command buffer per robot (not a strawman)
6. **Same sync interval** - Both use latency-based sync timing

### Communication Model Parameters
```python
@dataclass
class CommsModel:
    reliability: float      # P(message arrives intact) 0.0-1.0
    latency_steps: int      # Round-trip time in simulation steps
    partition_duration: int # Steps of zero connectivity
    sync_interval: int      # How often robots attempt sync

@dataclass
class CommsScenario:
    partition_probability: float    # Per-step chance of starting blackout
    partition_duration_range: tuple # (min, max) steps
```

### Space-Realistic Test Scenarios
| Scenario    | Reliability | Latency | Partition Prob | Duration | Notes |
|-------------|-------------|---------|----------------|----------|-------|
| LEO         | 0.95        | 1 step  | 1%             | 0-5 steps | Best case |
| LEO_Eclipse | 0.95        | 1 step  | 8%             | 15-40 steps | ISS eclipse |
| Lunar       | 0.80        | 10 steps | 3%            | 10-30 steps | Earth-Moon |
| Mars        | 0.70        | 100 steps | 5%           | 50-200 steps | The killer app |

### Known Limitations (Honestly Documented)
- [x] Command buffering for centralized (5 commands)
- [x] Synchronized random events between runs
- [x] Same completion criteria (actual work done)
- [ ] No physics constraints (fuel, collision, mass, thrust limits)
- [ ] Discrete time steps, not continuous dynamics
- [ ] No sensor noise or localization error
- [ ] Task model is simplified (instant start, linear progress)
- [ ] CRDT has ~26% duplicate work overhead during partitions

### What Would Convince a Technical Cofounder
- [x] Centralized baseline has command buffering (not strawman)
- [x] Synchronized partition events (provably same conditions)
- [x] Same message outcomes for both approaches
- [x] Same completion criteria (actual work, not knowledge)
- [x] Latency demonstrated as primary advantage (not just reliability)
- [x] Partition tolerance shown explicitly with duration analysis
- [x] Limitations documented honestly in README
- [ ] Edge cases handled (tie-breaking, clock skew)

### Current Fair Comparison Results
| Scenario    | CRDT Steps | Centralized Steps | Winner |
|-------------|------------|-------------------|--------|
| LEO         | ~150       | ~90               | Centralized (40% faster) |
| LEO_Eclipse | ~170       | ~100              | Centralized (41% faster) |
| Lunar       | ~120       | ~150              | **CRDT (+18%)** |
| Mars        | ~210       | 1000+ (timeout)   | **CRDT (+79%)** |

**Key insight**: Crossover point is at Lunar distances (~80% reliability, 10 step latency).
CRDT's ~26% duplicate work overhead is outweighed by zero idle time at Lunar+ distances.

## Success Criteria

The codebase is ready for cofounder conversations when:

- [x] All tests pass (22/22 passing)
- [x] Code is well-documented (docstrings, type hints)
- [x] Simulation runs with one command (`python -m src.simulation.engine`)
- [x] Results are reproducible (seeded RNG, pre-generated events)
- [x] Technical documentation explains the approach (README, PITCH, CLAUDE.md)
- [ ] Safety architecture is implemented (even if basic)
- [x] Simulation comparison is fair and defensible (synchronized events)
- [x] Known limitations are documented (README, CLAUDE.md)
