# CLAUDE.md - Claude Code Agent Instructions

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
src/crdt/       - CRDT implementations (core IP)
src/simulation/ - Simulation engine
src/safety/     - Safety-critical code (extra careful here)
src/visualization/ - Charts and displays
tests/          - Mirror src/ structure
docs/           - Technical documentation
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

### P0 - Critical (Do First)

1. **Refactor into proper package structure**
   - Move monolithic `simulation.py` into `src/` modules
   - Create proper `__init__.py` files
   - Ensure imports work: `from src.crdt import CRDTRobotState`

2. **Add unit tests for CRDT properties**
   - Test commutativity: `merge(A,B) == merge(B,A)`
   - Test associativity
   - Test idempotency
   - Use `pytest` and property-based testing with `hypothesis`

3. **Add hardware failure model**
   ```python
   class HardwareFailureModel:
       seu_rate: float = 0.001      # Single event upset (radiation)
       sensor_noise: float = 0.5    # Position uncertainty meters
       comms_latency: float = 0.0   # Additional delay
   ```

4. **Implement SafetySupervisor**
   - Watchdog timer
   - Geofence checking
   - Collision prediction
   - E-stop broadcast

### P1 - Important (This Week)

5. **Add type hints everywhere**
6. **Add docstrings to all public functions**
7. **Create technical documentation**
8. **Improve visualization** - add 3D plot option

### P2 - Nice to Have

9. **Add FastAPI endpoint** for running simulations
10. **Add real-time web visualization**
11. **Containerize with Docker**

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
# Run simulation
python -m src.simulation.engine

# Run specific scenario
python -m src.simulation.engine --reliability 0.5 --robots 10

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Type checking
mypy src/

# Lint
ruff check src/

# Generate charts
python -m src.visualization.charts
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
src/crdt/state.py      - Core CRDT implementation (THE IP)
src/simulation/engine.py - Main simulation loop
src/safety/supervisor.py - Safety monitoring
tests/test_crdt.py     - CRDT property tests
```

## Questions To Ask Me

If unclear about anything, ask:

1. "Should this preserve CRDT properties?"
2. "Is this safety-critical code?"
3. "Should I add tests for this?"
4. "What's the priority of this task?"

## Simulation Validity Requirements

### Fair Comparison Criteria
For the CRDT vs Centralized comparison to survive technical scrutiny:

1. **Same communication budget** - Both approaches use roughly equal total messages
2. **Equivalent local autonomy** - Centralized robots have buffered commands (3-5 step lookahead)
3. **Same operational model** - If CRDT robots work during partition, centralized executes stored sequences
4. **Latency as primary variable** - Space comms latency is the real constraint, not just reliability

### Communication Model Parameters
```python
@dataclass
class CommsModel:
    reliability: float      # P(message arrives intact) 0.0-1.0
    latency_steps: int      # Round-trip time in simulation steps
    partition_duration: int # Steps of zero connectivity
    sync_interval: int      # How often robots attempt sync
```

### Space-Realistic Test Scenarios
| Scenario | Reliability | Latency | Partition | Notes |
|----------|-------------|---------|-----------|-------|
| LEO      | 0.95        | 1 step  | 0-5 steps | Best case |
| GEO      | 0.90        | 3 steps | 5-15 steps | Typical comsat |
| Lunar    | 0.80        | 10 steps | 10-30 steps | Earth-Moon |
| Mars     | 0.70        | 100 steps | 50+ steps | The killer app |

### Known Limitations (Be Honest About These)
- [x] Current centralized baseline is naive (no command buffering) - **FIXED: Now has 5-command buffer**
- [ ] No physics constraints (fuel, collision, mass, thrust limits)
- [ ] Discrete time steps, not continuous dynamics
- [ ] No sensor noise or localization error
- [ ] Task model is simplified (instant start, linear progress)

### What Would Convince a Technical Cofounder
- [x] Centralized baseline has command buffering (not strawman)
- [x] Same comms budget for both approaches (same sync_interval)
- [x] Latency demonstrated as primary advantage (not just reliability)
- [x] Partition tolerance shown explicitly with duration analysis
- [ ] Limitations documented honestly in README
- [ ] Edge cases handled (tie-breaking, clock skew)

### Current Fair Comparison Results
| Scenario | CRDT Steps | Centralized Steps | Winner |
|----------|------------|-------------------|--------|
| LEO      | ~165       | ~87               | Centralized (89% faster) |
| GEO      | ~153       | ~106              | Centralized (45% faster) |
| Lunar    | ~146       | ~172              | CRDT (15% faster) |
| Mars     | ~301       | 1000+ (timeout)   | CRDT (70% faster) |

**Key insight**: CRDT advantage emerges when latency and partitions dominate.
Centralized wins in good conditions - this is honest and defensible.

## Success Criteria

The codebase is ready for cofounder conversations when:

- [x] All tests pass (22/22 passing)
- [ ] Code is well-documented
- [x] Simulation runs with one command
- [x] Results are reproducible
- [ ] Technical documentation explains the approach
- [ ] Safety architecture is implemented (even if basic)
- [x] Simulation comparison is fair and defensible
- [ ] Known limitations are documented
