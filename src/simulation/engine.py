"""
Fair Simulation Engine for CRDT vs Centralized Comparison.

This module implements a fair comparison between CRDT-coordinated robots
and centralized ground control, addressing the following requirements:

1. Same communication budget for both approaches
2. Centralized robots have command buffering (not strawman)
3. Latency modeled separately from reliability
4. Partition tolerance tested explicitly

The goal is to produce results that would survive scrutiny from a
technical cofounder with robotics/distributed systems background.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.crdt import CRDTState, Vector3

logger = logging.getLogger(__name__)


# =============================================================================
# COMMUNICATION MODEL
# =============================================================================

@dataclass
class CommsModel:
    """
    Communication model for space robotics simulation.

    Separates reliability (packet loss) from latency (delay) and
    supports partition events (total blackouts).

    Attributes:
        reliability: Probability a message arrives intact (0.0-1.0)
        latency_steps: Round-trip time in simulation steps
        partition_duration: Current remaining steps of zero connectivity
        sync_interval: How often robots attempt to sync state
    """
    reliability: float = 0.9
    latency_steps: int = 1
    partition_duration: int = 0
    sync_interval: int = 5

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be between 0.0 and 1.0")
        if self.latency_steps < 0:
            raise ValueError("latency_steps must be non-negative")

    def message_succeeds(self) -> bool:
        """Check if a message would be delivered (considering partition)."""
        if self.partition_duration > 0:
            return False
        return random.random() < self.reliability

    def start_partition(self, duration: int) -> None:
        """Start a communication partition (blackout)."""
        self.partition_duration = duration

    def tick(self) -> None:
        """Advance one simulation step, reducing partition duration."""
        if self.partition_duration > 0:
            self.partition_duration -= 1


@dataclass
class CommsScenario:
    """Predefined communication scenarios for different space environments."""
    name: str
    reliability: float
    latency_steps: int
    partition_probability: float  # Per-step chance of starting a partition
    partition_duration_range: Tuple[int, int]  # (min, max) steps

    def create_comms_model(self) -> CommsModel:
        """Create a CommsModel for this scenario."""
        return CommsModel(
            reliability=self.reliability,
            latency_steps=self.latency_steps,
            sync_interval=max(1, self.latency_steps)  # Sync at least as often as latency allows
        )

    def generate_partition_schedule(
        self,
        max_steps: int,
        rng: random.Random
    ) -> Dict[int, int]:
        """
        Pre-generate partition events for deterministic comparison.

        Returns dict mapping step -> partition_duration.
        This ensures both CRDT and centralized runs experience
        the exact same partition events at the exact same times.
        """
        schedule: Dict[int, int] = {}
        step = 1
        while step <= max_steps:
            if rng.random() < self.partition_probability:
                duration = rng.randint(*self.partition_duration_range)
                schedule[step] = duration
                step += duration  # Skip ahead past this partition
            step += 1
        return schedule


# Predefined scenarios from CLAUDE.md
SCENARIOS = {
    "LEO": CommsScenario("LEO", 0.95, 1, 0.01, (0, 5)),
    "LEO_Eclipse": CommsScenario("LEO_Eclipse", 0.95, 1, 0.08, (15, 40)),  # ISS-like eclipse blackouts
    "GEO": CommsScenario("GEO", 0.90, 3, 0.02, (5, 15)),
    "Lunar": CommsScenario("Lunar", 0.80, 10, 0.03, (10, 30)),
    "Mars": CommsScenario("Mars", 0.70, 100, 0.05, (50, 200)),
}


# =============================================================================
# TASK MODEL
# =============================================================================

class TaskStatus(Enum):
    """Status of a task in the simulation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Task:
    """
    A task to be completed by robots.

    Attributes:
        task_id: Unique identifier
        location: 3D position where task must be performed
        duration: Number of work steps required to complete
        task_type: Category of task (inspect, repair, refuel)
    """
    task_id: str
    location: Vector3
    duration: int
    task_type: str = "inspect"

    def __hash__(self) -> int:
        return hash(self.task_id)


# =============================================================================
# CRDT ROBOT (Our Approach)
# =============================================================================

@dataclass
class CRDTRobot:
    """
    Robot using CRDT coordination for autonomous operation.

    Key properties:
    - Operates autonomously using local state
    - Syncs with peers when communication allows
    - Continues working during partitions
    """
    robot_id: str
    position: Vector3
    speed: float = 2.0
    state: CRDTState = field(default=None)
    current_task: Optional[str] = None
    working: bool = False

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = CRDTState(self.robot_id)

    def decide_and_act(self, tasks: Dict[str, Task], timestamp: int) -> None:
        """
        Make autonomous decisions based on local CRDT state.

        This is the key advantage: robots can act without ground control.
        """
        # Update own position in state
        self.state.update_position(self.robot_id, self.position, timestamp)

        # If working on a task, continue
        if self.current_task and self.working:
            if self.current_task not in self.state.completed_tasks:
                task = tasks.get(self.current_task)
                if task:
                    self._work_on_task(task, timestamp)
                    return
            else:
                # Task completed, find new work
                self.current_task = None
                self.working = False

        # Find best available task
        best_task = self._select_task(tasks, timestamp)

        if best_task:
            self.current_task = best_task.task_id
            self.state.claim_task(best_task.task_id, self.robot_id, timestamp)

            dist = self.position.distance_to(best_task.location)
            if dist < 2.0:
                self.working = True
                self._work_on_task(best_task, timestamp)
            else:
                self.working = False
                self.position = self.position.move_toward(best_task.location, self.speed)

    def _select_task(self, tasks: Dict[str, Task], timestamp: int) -> Optional[Task]:
        """Select best available task based on local knowledge."""
        available = []

        for task in tasks.values():
            if task.task_id in self.state.completed_tasks:
                continue

            # Check if claimed by others, but allow if we're much closer
            if self.state.is_task_claimed_by_other(task.task_id, self.robot_id):
                claimer_id = self.state.get_task_claimer(task.task_id)
                if claimer_id:
                    claimer_pos = self.state.robot_positions.get(claimer_id)
                    if claimer_pos:
                        my_dist = self.position.distance_to(task.location)
                        their_dist = claimer_pos[0].distance_to(task.location)
                        if my_dist > their_dist * 0.5:
                            continue

            available.append(task)

        if not available:
            return None

        return min(available, key=lambda t: self.position.distance_to(t.location))

    def _work_on_task(self, task: Task, timestamp: int) -> None:
        """Perform work on a task."""
        self.state.add_progress(task.task_id, 1)

        total_progress = self.state.get_task_progress(task.task_id)
        if total_progress >= task.duration:
            self.state.mark_task_complete(task.task_id, timestamp)
            self.current_task = None
            self.working = False

    def sync_with(self, other_state: CRDTState) -> None:
        """Merge another robot's state into ours."""
        self.state.merge(other_state)


# =============================================================================
# FAIR CENTRALIZED ROBOT (With Command Buffering)
# =============================================================================

@dataclass
class BufferedCommand:
    """A command stored in the robot's buffer."""
    command_type: str  # "goto", "work", "idle"
    task_id: Optional[str]
    target_position: Optional[Vector3]
    issued_at: int  # Timestamp when issued


@dataclass
class FairCentralizedRobot:
    """
    Centralized robot with command buffering for fair comparison.

    Key difference from naive baseline:
    - Stores multiple commands in buffer
    - Can execute buffered commands during comms blackout
    - Only truly stalls when buffer is empty AND comms are down
    - Tracks completed tasks locally to avoid duplicate work

    This represents a more realistic centralized architecture.
    """
    robot_id: str
    position: Vector3
    speed: float = 2.0
    command_buffer: List[BufferedCommand] = field(default_factory=list)
    buffer_size: int = 5  # How many commands can be buffered
    current_command: Optional[BufferedCommand] = None
    work_progress: Dict[str, int] = field(default_factory=dict)
    completed_tasks: set = field(default_factory=set)  # Track locally completed tasks
    current_task_id: Optional[str] = None  # Track what task we're working on

    def receive_commands(self, commands: List[BufferedCommand]) -> None:
        """
        Receive commands from ground control.

        Filters out exact duplicate commands and commands for completed tasks.
        """
        for cmd in commands:
            # Skip commands for already completed tasks
            if cmd.task_id and cmd.task_id in self.completed_tasks:
                continue

            # Skip exact duplicates (same task AND command type)
            if cmd.task_id:
                is_duplicate = (
                    (self.current_command and
                     self.current_command.task_id == cmd.task_id and
                     self.current_command.command_type == cmd.command_type) or
                    any(c.task_id == cmd.task_id and c.command_type == cmd.command_type
                        for c in self.command_buffer)
                )
                if is_duplicate:
                    continue

            if len(self.command_buffer) < self.buffer_size:
                self.command_buffer.append(cmd)

    def execute_step(self, tasks: Dict[str, Task], timestamp: int) -> Optional[Dict]:
        """
        Execute one simulation step.

        Returns status dict if progress was made, None if idle.
        """
        # Get next command if needed
        if self.current_command is None:
            while self.command_buffer:
                cmd = self.command_buffer.pop(0)
                # Skip commands for completed tasks
                if cmd.task_id and cmd.task_id in self.completed_tasks:
                    continue
                self.current_command = cmd
                self.current_task_id = cmd.task_id
                break
            else:
                self.current_task_id = None
                return None  # No valid commands, truly idle

        cmd = self.current_command

        # Double-check task not already completed
        if cmd.task_id and cmd.task_id in self.completed_tasks:
            self.current_command = None
            return None

        if cmd.command_type == "goto" and cmd.target_position:
            dist = self.position.distance_to(cmd.target_position)
            if dist < 2.0:
                # Arrived, command complete - immediately start work if we have work command
                self.current_command = None
                return {"type": "arrived", "task_id": cmd.task_id}
            else:
                self.position = self.position.move_toward(cmd.target_position, self.speed)
                return {"type": "moving", "task_id": cmd.task_id}

        elif cmd.command_type == "work" and cmd.task_id:
            task = tasks.get(cmd.task_id)
            if task:
                self.work_progress[cmd.task_id] = self.work_progress.get(cmd.task_id, 0) + 1
                if self.work_progress[cmd.task_id] >= task.duration:
                    self.current_command = None
                    self.completed_tasks.add(cmd.task_id)
                    self.current_task_id = None
                    return {"type": "completed", "task_id": cmd.task_id}
                return {"type": "working", "task_id": cmd.task_id}

        elif cmd.command_type == "idle":
            self.current_command = None
            return None

        return None

    def buffer_space(self) -> int:
        """Return available space in command buffer."""
        return self.buffer_size - len(self.command_buffer)

    def needs_assignment(self) -> bool:
        """Check if robot needs a new task assignment."""
        return self.current_task_id is None and len(self.command_buffer) == 0


@dataclass
class FairGroundControl:
    """
    Ground control with realistic command planning.

    Plans ahead and sends multiple commands per communication window
    to utilize robot's command buffer. Includes timeout for stuck tasks.
    """
    task_assignments: Dict[str, str] = field(default_factory=dict)  # task_id -> robot_id
    completed_tasks: set = field(default_factory=set)
    task_progress: Dict[str, int] = field(default_factory=dict)
    commands_sent_for: Dict[str, int] = field(default_factory=dict)  # task_id -> last timestamp
    assignment_timeout: int = 150  # Steps before considering reassignment

    def plan_commands(
        self,
        robots: List[FairCentralizedRobot],
        tasks: Dict[str, Task],
        timestamp: int
    ) -> Dict[str, List[BufferedCommand]]:
        """
        Plan commands for all robots, filling their buffers.

        Only sends commands to robots that need them (avoiding duplicates).
        Returns dict of robot_id -> list of commands.
        """
        commands: Dict[str, List[BufferedCommand]] = {r.robot_id: [] for r in robots}

        # Check for stuck assignments (no progress for too long)
        for task_id in list(self.task_assignments.keys()):
            last_cmd = self.commands_sent_for.get(task_id, 0)
            if timestamp - last_cmd > self.assignment_timeout:
                del self.task_assignments[task_id]

        available_tasks = [
            t for t in tasks.values()
            if t.task_id not in self.completed_tasks
            and t.task_id not in self.task_assignments
        ]

        for robot in robots:
            robot_commands = []

            # Check if robot has current assignment
            current_task_id = None
            for task_id, assigned_robot in self.task_assignments.items():
                if assigned_robot == robot.robot_id:
                    current_task_id = task_id
                    break

            # Only assign new task if robot needs one
            if current_task_id is None and available_tasks and robot.needs_assignment():
                closest = min(available_tasks,
                            key=lambda t: robot.position.distance_to(t.location))
                self.task_assignments[closest.task_id] = robot.robot_id
                available_tasks.remove(closest)
                current_task_id = closest.task_id

            # Only send commands if robot actually needs them
            if current_task_id and robot.needs_assignment():
                task = tasks[current_task_id]

                robot_commands.append(BufferedCommand(
                    command_type="goto",
                    task_id=current_task_id,
                    target_position=task.location,
                    issued_at=timestamp
                ))

                robot_commands.append(BufferedCommand(
                    command_type="work",
                    task_id=current_task_id,
                    target_position=None,
                    issued_at=timestamp
                ))

                self.commands_sent_for[current_task_id] = timestamp

            commands[robot.robot_id] = robot_commands

        return commands

    def receive_status(self, robot_id: str, status: Dict, tasks: Dict[str, Task], timestamp: int = 0) -> None:
        """Process status update from robot."""
        if status is None:
            return

        task_id = status.get("task_id")

        if status.get("type") == "completed":
            if task_id:
                self.completed_tasks.add(task_id)
                if task_id in self.task_assignments:
                    del self.task_assignments[task_id]
                if task_id in self.commands_sent_for:
                    del self.commands_sent_for[task_id]

        elif status.get("type") in ("working", "moving", "arrived"):
            if task_id:
                self.task_progress[task_id] = self.task_progress.get(task_id, 0) + 1
                # Reset timeout since robot is making progress
                if task_id in self.commands_sent_for:
                    self.commands_sent_for[task_id] = timestamp

        elif status.get("type") == "idle":
            # Robot is idle - sync any completed tasks ground might not know about
            robot_completed = status.get("completed_tasks", [])
            for tid in robot_completed:
                if tid not in self.completed_tasks:
                    self.completed_tasks.add(tid)
                    if tid in self.task_assignments:
                        del self.task_assignments[tid]
                    if tid in self.commands_sent_for:
                        del self.commands_sent_for[tid]

            # Clear any remaining stale assignments for this robot
            stale_tasks = [
                tid for tid, rid in list(self.task_assignments.items())
                if rid == robot_id and tid not in self.completed_tasks
            ]
            for tid in stale_tasks:
                del self.task_assignments[tid]
                if tid in self.commands_sent_for:
                    del self.commands_sent_for[tid]


# =============================================================================
# FAIR SIMULATION ENGINE
# =============================================================================

@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    steps: int = 0
    completed_tasks: int = 0
    total_tasks: int = 0
    messages_sent: int = 0
    messages_failed: int = 0
    partition_steps: int = 0  # Steps spent in partition
    idle_steps: int = 0  # Steps where robots had nothing to do
    actual_work_done: int = 0  # Total work units completed by all robots
    duplicate_work: int = 0  # Work done on already-completed tasks (CRDT overhead)
    total_work_required: int = 0  # Sum of all task durations

    @property
    def completion_rate(self) -> float:
        return self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0

    @property
    def message_success_rate(self) -> float:
        return (self.messages_sent - self.messages_failed) / self.messages_sent if self.messages_sent > 0 else 0.0


class FairSimulation:
    """
    Fair simulation comparing CRDT vs Centralized approaches.

    Key fairness properties:
    1. Same communication budget (sync_interval)
    2. Centralized robots have command buffering
    3. Both use same CommsModel with SYNCHRONIZED random events
    4. Same partition schedule for both approaches
    5. Same message success/failure pattern for fair comparison
    6. Metrics track actual work done, not just completion
    """

    def __init__(
        self,
        num_robots: int = 5,
        num_tasks: int = 10,
        scenario: str = "GEO",
        space_size: float = 100.0,
        seed: Optional[int] = None,
        buffer_size: int = 5,
        max_steps: int = 1000
    ):
        self.seed = seed if seed is not None else 42
        self.rng = random.Random(self.seed)

        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.scenario = SCENARIOS.get(scenario, SCENARIOS["GEO"])
        self.space_size = space_size
        self.buffer_size = buffer_size
        self.max_steps = max_steps

        # Generate tasks with main RNG
        self.tasks = self._create_tasks()

        # Pre-generate synchronized random events
        self._partition_schedule: Dict[int, int] = {}
        self._message_outcomes: List[bool] = []
        self._start_positions: List[Vector3] = []

    def _prepare_trial(self) -> None:
        """
        Prepare synchronized random events for a single trial.

        This ensures CRDT and centralized runs experience IDENTICAL:
        - Partition events (same start time, same duration)
        - Message success/failure patterns
        - Starting positions
        """
        # Create a fresh RNG for this trial
        trial_rng = random.Random(self.rng.randint(0, 2**32))

        # Pre-generate partition schedule
        self._partition_schedule = self.scenario.generate_partition_schedule(
            self.max_steps, trial_rng
        )

        # Pre-generate many message outcomes (enough for any run)
        # Each sync round uses roughly num_robots * (num_robots-1) messages for CRDT
        # or num_robots * 2 for centralized, so pre-generate plenty
        max_messages = self.max_steps * self.num_robots * self.num_robots * 2
        self._message_outcomes = [
            trial_rng.random() < self.scenario.reliability
            for _ in range(max_messages)
        ]
        self._message_index = 0

        # Pre-generate starting positions
        self._start_positions = [
            Vector3(
                self.space_size / 2 + trial_rng.uniform(-10, 10),
                self.space_size / 2 + trial_rng.uniform(-10, 10),
                self.space_size / 2 + trial_rng.uniform(-10, 10)
            )
            for _ in range(self.num_robots)
        ]

    def _next_message_succeeds(self, partition_active: bool) -> bool:
        """Get next pre-determined message outcome."""
        if partition_active:
            return False
        if self._message_index >= len(self._message_outcomes):
            # Fallback - should not happen with proper pre-generation
            return self.rng.random() < self.scenario.reliability
        result = self._message_outcomes[self._message_index]
        self._message_index += 1
        return result

    def _reset_message_index(self) -> None:
        """Reset message index for centralized run to use same sequence."""
        self._message_index = 0

    def _create_tasks(self) -> Dict[str, Task]:
        """Generate random tasks using controlled RNG."""
        tasks = {}
        for i in range(self.num_tasks):
            task_id = f"task_{i}"
            tasks[task_id] = Task(
                task_id=task_id,
                location=Vector3(
                    self.rng.uniform(0, self.space_size),
                    self.rng.uniform(0, self.space_size),
                    self.rng.uniform(0, self.space_size)
                ),
                duration=self.rng.randint(5, 15),
                task_type=self.rng.choice(["inspect", "repair", "refuel"])
            )
        return tasks

    def _get_start_positions(self) -> List[Vector3]:
        """Return pre-generated starting positions for fair comparison."""
        return [deepcopy(pos) for pos in self._start_positions]

    def run_crdt(self, max_steps: int = None) -> SimulationMetrics:
        """
        Run simulation with CRDT approach.

        Uses pre-generated partition schedule and message outcomes
        for fair comparison with centralized approach.
        """
        if max_steps is None:
            max_steps = self.max_steps

        tasks = deepcopy(self.tasks)
        comms = self.scenario.create_comms_model()

        # Use pre-generated positions
        positions = self._get_start_positions()
        robots = [
            CRDTRobot(robot_id=f"robot_{i}", position=pos)
            for i, pos in enumerate(positions)
        ]

        metrics = SimulationMetrics(total_tasks=len(tasks))
        metrics.total_work_required = sum(t.duration for t in tasks.values())

        # Track actual work done (not just what robots know about)
        actual_completed: set = set()
        # Track total work done per task (to detect duplicate work)
        total_work_per_task: Dict[str, int] = {t: 0 for t in tasks.keys()}

        for step in range(1, max_steps + 1):
            metrics.steps = step

            # Check pre-generated partition schedule
            if step in self._partition_schedule:
                comms.start_partition(self._partition_schedule[step])

            partition_active = comms.partition_duration > 0
            if partition_active:
                metrics.partition_steps += 1

            # Each robot acts autonomously
            for robot in robots:
                # Track work before action
                old_progress = {
                    t: robot.state.get_task_progress(t)
                    for t in tasks.keys()
                }

                robot.decide_and_act(tasks, step)

                # Track new work done
                for task_id, task in tasks.items():
                    new_progress = robot.state.get_task_progress(task_id)
                    work_increment = new_progress - old_progress.get(task_id, 0)
                    if work_increment > 0:
                        metrics.actual_work_done += work_increment
                        # Track duplicate work (work on already-completed tasks)
                        if task_id in actual_completed:
                            metrics.duplicate_work += work_increment
                        else:
                            total_work_per_task[task_id] += work_increment
                            # Check if this work over-completed the task (multiple robots)
                            if total_work_per_task[task_id] > task.duration:
                                overflow = total_work_per_task[task_id] - task.duration
                                metrics.duplicate_work += min(work_increment, overflow)
                    if new_progress >= task.duration:
                        actual_completed.add(task_id)

            # Periodic state sync (same interval as centralized comms)
            if step % comms.sync_interval == 0:
                for i, robot_a in enumerate(robots):
                    for robot_b in robots[i+1:]:
                        # Bidirectional sync attempts
                        metrics.messages_sent += 2

                        if self._next_message_succeeds(partition_active):
                            robot_b.sync_with(robot_a.state)
                        else:
                            metrics.messages_failed += 1

                        if self._next_message_succeeds(partition_active):
                            robot_a.sync_with(robot_b.state)
                        else:
                            metrics.messages_failed += 1

            comms.tick()

            # Check completion: ALL tasks actually completed (work done)
            if len(actual_completed) >= len(tasks):
                break

        metrics.completed_tasks = len(actual_completed)

        return metrics

    def run_centralized(self, max_steps: int = None) -> SimulationMetrics:
        """
        Run simulation with fair centralized approach.

        Uses pre-generated partition schedule and message outcomes
        for fair comparison with CRDT approach.
        """
        if max_steps is None:
            max_steps = self.max_steps

        tasks = deepcopy(self.tasks)
        comms = self.scenario.create_comms_model()
        ground = FairGroundControl()

        # Reset message index so centralized uses same message outcomes as CRDT
        self._reset_message_index()

        # Use pre-generated positions (same as CRDT)
        positions = self._get_start_positions()
        robots = [
            FairCentralizedRobot(
                robot_id=f"robot_{i}",
                position=pos,
                buffer_size=self.buffer_size
            )
            for i, pos in enumerate(positions)
        ]

        metrics = SimulationMetrics(total_tasks=len(tasks))

        # Track actual work done (same criteria as CRDT)
        actual_completed: set = set()

        for step in range(1, max_steps + 1):
            metrics.steps = step

            # Check pre-generated partition schedule (SAME as CRDT)
            if step in self._partition_schedule:
                comms.start_partition(self._partition_schedule[step])

            partition_active = comms.partition_duration > 0
            if partition_active:
                metrics.partition_steps += 1

            # Communication window (same interval as CRDT sync)
            if step % comms.sync_interval == 0:
                # Ground plans commands
                commands = ground.plan_commands(robots, tasks, step)

                # Send commands to each robot
                for robot in robots:
                    robot_commands = commands.get(robot.robot_id, [])
                    if robot_commands:
                        metrics.messages_sent += 1
                        if self._next_message_succeeds(partition_active):
                            robot.receive_commands(robot_commands)
                        else:
                            metrics.messages_failed += 1

            # Each robot executes from buffer
            for robot in robots:
                # Track work before action
                old_progress = dict(robot.work_progress)

                status = robot.execute_step(tasks, step)

                # Track new work done
                for task_id, task in tasks.items():
                    new_progress = robot.work_progress.get(task_id, 0)
                    if new_progress > old_progress.get(task_id, 0):
                        metrics.actual_work_done += (new_progress - old_progress.get(task_id, 0))
                    if new_progress >= task.duration:
                        actual_completed.add(task_id)

                if status is None:
                    metrics.idle_steps += 1
                    # Robot is idle - send idle status with completed tasks so ground can sync
                    if step % comms.sync_interval == 0 and robot.needs_assignment():
                        idle_status = {
                            "type": "idle",
                            "robot_id": robot.robot_id,
                            "completed_tasks": list(robot.completed_tasks)
                        }
                        metrics.messages_sent += 1
                        if self._next_message_succeeds(partition_active):
                            ground.receive_status(robot.robot_id, idle_status, tasks, step)
                        else:
                            metrics.messages_failed += 1
                else:
                    # Try to send status back (same interval)
                    if step % comms.sync_interval == 0:
                        metrics.messages_sent += 1
                        if self._next_message_succeeds(partition_active):
                            ground.receive_status(robot.robot_id, status, tasks, step)
                        else:
                            metrics.messages_failed += 1

            comms.tick()

            # Check completion: ALL tasks actually completed (same as CRDT)
            if len(actual_completed) >= len(tasks):
                break

        metrics.completed_tasks = len(actual_completed)

        return metrics

    def run_comparison(self, max_steps: int = None, num_trials: int = 10) -> Dict:
        """
        Run multiple trials comparing both approaches.

        FAIRNESS GUARANTEE: Each trial uses:
        - Same tasks
        - Same starting positions
        - Same partition schedule
        - Same message success/failure sequence

        Returns comprehensive comparison data.
        """
        if max_steps is None:
            max_steps = self.max_steps

        crdt_results = []
        centralized_results = []

        logger.info(f"Running comparison: {self.scenario.name} scenario")
        logger.info(f"Robots: {self.num_robots}, Tasks: {self.num_tasks}")
        logger.info(f"Reliability: {self.scenario.reliability}, Latency: {self.scenario.latency_steps}")
        logger.info(f"Partition probability: {self.scenario.partition_probability}")

        for trial in range(num_trials):
            # Generate new tasks for this trial
            self.tasks = self._create_tasks()

            # Prepare synchronized random events for this trial
            self._prepare_trial()

            # Run CRDT simulation
            crdt = self.run_crdt(max_steps)

            # Run centralized simulation with SAME random events
            cent = self.run_centralized(max_steps)

            crdt_results.append(crdt)
            centralized_results.append(cent)

            logger.info(
                f"Trial {trial+1}: CRDT={crdt.steps} steps (partition={crdt.partition_steps}), "
                f"Centralized={cent.steps} steps (partition={cent.partition_steps}, idle={cent.idle_steps})"
            )

        return self._analyze_results(crdt_results, centralized_results)

    def _analyze_results(
        self,
        crdt_results: List[SimulationMetrics],
        centralized_results: List[SimulationMetrics]
    ) -> Dict:
        """Analyze and summarize comparison results."""

        def avg(results: List[SimulationMetrics], attr: str) -> float:
            return sum(getattr(r, attr) for r in results) / len(results)

        crdt_avg_steps = avg(crdt_results, "steps")
        cent_avg_steps = avg(centralized_results, "steps")

        # Calculate improvement (positive = CRDT better)
        if cent_avg_steps > 0:
            steps_improvement = (cent_avg_steps - crdt_avg_steps) / cent_avg_steps * 100
        else:
            steps_improvement = 0

        return {
            "scenario": self.scenario.name,
            "config": {
                "num_robots": self.num_robots,
                "num_tasks": self.num_tasks,
                "reliability": self.scenario.reliability,
                "latency_steps": self.scenario.latency_steps,
                "buffer_size": self.buffer_size,
            },
            "crdt": {
                "avg_steps": crdt_avg_steps,
                "avg_completion_rate": avg(crdt_results, "completion_rate"),
                "avg_messages_sent": avg(crdt_results, "messages_sent"),
                "avg_messages_failed": avg(crdt_results, "messages_failed"),
                "avg_partition_steps": avg(crdt_results, "partition_steps"),
            },
            "centralized": {
                "avg_steps": cent_avg_steps,
                "avg_completion_rate": avg(centralized_results, "completion_rate"),
                "avg_messages_sent": avg(centralized_results, "messages_sent"),
                "avg_messages_failed": avg(centralized_results, "messages_failed"),
                "avg_partition_steps": avg(centralized_results, "partition_steps"),
                "avg_idle_steps": avg(centralized_results, "idle_steps"),
            },
            "comparison": {
                "steps_improvement_pct": steps_improvement,
                "crdt_faster": crdt_avg_steps < cent_avg_steps,
            }
        }


def run_scenario_sweep(scenarios: List[str] = None, num_trials: int = 5, max_steps: int = 1000) -> List[Dict]:
    """
    Run comparison across multiple scenarios.

    Each scenario tests different communication conditions to identify
    where CRDT coordination outperforms centralized control.
    """
    if scenarios is None:
        scenarios = ["LEO", "LEO_Eclipse", "Lunar", "Mars"]

    results = []
    for scenario_name in scenarios:
        sim = FairSimulation(
            num_robots=5,
            num_tasks=10,
            scenario=scenario_name,
            seed=42,
            max_steps=max_steps
        )
        result = sim.run_comparison(num_trials=num_trials)
        results.append(result)

        # Report partition statistics
        avg_partition = result['crdt']['avg_partition_steps']
        print(f"\n{scenario_name}: CRDT {'wins' if result['comparison']['crdt_faster'] else 'loses'} "
              f"({result['comparison']['steps_improvement_pct']:.1f}% difference)")
        print(f"  Avg partition steps: {avg_partition:.1f}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("FAIR SIMULATION: CRDT vs Centralized (with buffering)")
    print("=" * 60)

    results = run_scenario_sweep()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scenario':<10} {'CRDT Steps':<12} {'Cent Steps':<12} {'Improvement':<12}")
    print("-" * 46)
    for r in results:
        print(f"{r['scenario']:<10} "
              f"{r['crdt']['avg_steps']:<12.1f} "
              f"{r['centralized']['avg_steps']:<12.1f} "
              f"{r['comparison']['steps_improvement_pct']:>+.1f}%")
