#!/usr/bin/env python3
"""
Lunar Ground Operations Simulation with CRDT Coordination.

This simulation demonstrates CRDT-based coordination for autonomous rovers
performing construction tasks at a lunar base. Rovers must handle
communication partitions caused by terrain occlusion and distance.

Scenario: Lunar Base Construction
- 5 autonomous rovers/trucks
- 1 central base (Starship-style habitat)
- Construction tasks: solar panels, regolith covering, resource transport
- Communication partitions from terrain and distance

Physics:
- Lunar gravity: 1.62 m/s² (1/6 Earth)
- Rover dynamics: simplified ground vehicle model
- Terrain: procedurally generated with craters and hills

CRDT Coordination:
- Task claiming prevents duplicate work
- Progress tracking survives partitions
- State merges when rovers return to base range

Usage:
    python ground_simulation.py

Author: CRDT Space Robotics Team
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D

# Import CRDT implementation
from src.crdt import CRDTState, Vector3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

LUNAR_GRAVITY = 1.62e-3      # km/s² (1.62 m/s²)
LUNAR_RADIUS = 1737.4        # km

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Base parameters
BASE_POSITION = np.array([0.0, 0.0])  # Base at origin (km)
BASE_COMM_RANGE = 5.0                  # Direct comm range (km)
RELAY_BOOST_RANGE = 2.0                # Range boost when rovers relay

# Rover parameters
ROVER_MAX_SPEED = 0.005      # km/s (5 m/s = 18 km/h)
ROVER_ACCELERATION = 0.001   # km/s²
ROVER_TURN_RATE = 0.5        # rad/s
NUM_ROVERS = 5

# Task parameters
TASK_WORK_RATE = 10.0        # Work units per second when at task
TASK_COMPLETION = 200.0      # Work units to complete task

# Terrain parameters
TERRAIN_SIZE = 20.0          # km x km area
NUM_CRATERS = 15             # Number of craters
NUM_HILLS = 10               # Number of hills

# Communication parameters
RELIABILITY = 0.85           # Base message success rate
SYNC_INTERVAL = 30.0         # Seconds between sync attempts
PARTITION_PROBABILITY = 0.02 # Per-sync chance of partition
PARTITION_DURATION_RANGE = (60.0, 180.0)  # Partition duration (seconds)

# Simulation parameters
SIMULATION_DT = 1.0          # Time step (seconds)
MAX_SIMULATION_TIME = 2 * 3600  # 2 hours

# Visualization
ANIMATION_INTERVAL = 50      # ms between frames
TRAIL_LENGTH = 100           # Points in rover trail


# =============================================================================
# TERRAIN MODEL
# =============================================================================

@dataclass
class Crater:
    """A crater that blocks line-of-sight communication."""
    center: np.ndarray  # (x, y) position
    radius: float       # km
    depth: float        # km (for visualization)


@dataclass
class Hill:
    """A hill/ridge that blocks communication."""
    center: np.ndarray  # (x, y) position
    radius: float       # km
    height: float       # km


class LunarTerrain:
    """
    Procedurally generated lunar terrain with craters and hills.

    Used for:
    - Line-of-sight communication blocking
    - Visual representation
    - Path planning obstacles
    """

    def __init__(self, size: float, num_craters: int, num_hills: int,
                 rng: random.Random):
        self.size = size
        self.craters: List[Crater] = []
        self.hills: List[Hill] = []
        self.rng = rng

        self._generate_terrain(num_craters, num_hills)

    def _generate_terrain(self, num_craters: int, num_hills: int) -> None:
        """Generate random craters and hills, avoiding the base area."""
        half = self.size / 2

        # Generate craters (avoid center 2km where base is)
        for _ in range(num_craters):
            for attempt in range(10):
                x = self.rng.uniform(-half, half)
                y = self.rng.uniform(-half, half)
                dist_from_base = math.sqrt(x**2 + y**2)

                if dist_from_base > 2.0:  # Not too close to base
                    radius = self.rng.uniform(0.1, 0.5)  # 100-500m
                    depth = radius * 0.3
                    self.craters.append(Crater(
                        center=np.array([x, y]),
                        radius=radius,
                        depth=depth
                    ))
                    break

        # Generate hills
        for _ in range(num_hills):
            for attempt in range(10):
                x = self.rng.uniform(-half, half)
                y = self.rng.uniform(-half, half)
                dist_from_base = math.sqrt(x**2 + y**2)

                if dist_from_base > 1.5:
                    radius = self.rng.uniform(0.2, 0.8)
                    height = self.rng.uniform(0.05, 0.2)  # 50-200m
                    self.hills.append(Hill(
                        center=np.array([x, y]),
                        radius=radius,
                        height=height
                    ))
                    break

    def blocks_line_of_sight(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """
        Check if terrain blocks line-of-sight between two positions.

        Uses simple ray-circle intersection for hills.
        """
        for hill in self.hills:
            if self._ray_intersects_circle(pos1, pos2, hill.center, hill.radius):
                return True
        return False

    def _ray_intersects_circle(self, p1: np.ndarray, p2: np.ndarray,
                                center: np.ndarray, radius: float) -> bool:
        """Check if line segment p1-p2 intersects circle."""
        d = p2 - p1
        f = p1 - center

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return False

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)

        # Check if intersection is within segment
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)

    def is_in_crater(self, pos: np.ndarray) -> bool:
        """Check if position is inside a crater (impassable)."""
        for crater in self.craters:
            dist = np.linalg.norm(pos - crater.center)
            if dist < crater.radius * 0.8:  # Inner 80% is impassable
                return True
        return False

    def get_elevation(self, pos: np.ndarray) -> float:
        """Get terrain elevation at position (for visualization)."""
        elevation = 0.0

        for hill in self.hills:
            dist = np.linalg.norm(pos - hill.center)
            if dist < hill.radius:
                # Gaussian-like hill profile
                elevation += hill.height * math.exp(-2 * (dist/hill.radius)**2)

        for crater in self.craters:
            dist = np.linalg.norm(pos - crater.center)
            if dist < crater.radius:
                # Bowl-shaped crater
                elevation -= crater.depth * (1 - (dist/crater.radius)**2)

        return elevation


# =============================================================================
# CONSTRUCTION TASKS
# =============================================================================

@dataclass
class ConstructionTask:
    """
    A construction task at the lunar base.

    Types:
    - solar_panel: Install solar panel array
    - regolith: Cover structure with regolith for radiation protection
    - transport: Transport resources from one location to another
    """
    task_id: str
    task_type: str
    position: np.ndarray        # Location (km from base)
    work_required: float        # Total work units needed
    work_completed: float = 0.0 # Work done so far
    assigned_rover: Optional[str] = None
    is_complete: bool = False

    def add_work(self, amount: float) -> None:
        """Add work progress to the task."""
        self.work_completed += amount
        if self.work_completed >= self.work_required:
            self.is_complete = True


# =============================================================================
# LUNAR ROVER
# =============================================================================

class LunarRover:
    """
    Autonomous lunar rover with CRDT coordination.

    Capabilities:
    - Ground navigation with obstacle avoidance
    - Construction task execution
    - Peer-to-peer CRDT synchronization
    - Relay communication for other rovers
    """

    def __init__(self, rover_id: str, start_position: np.ndarray):
        self.rover_id = rover_id

        # Position and motion (2D ground plane)
        self.position = start_position.copy()
        self.velocity = np.zeros(2)
        self.heading = 0.0  # radians, 0 = +X direction

        # CRDT state
        self.crdt_state = CRDTState(rover_id)

        # Task management
        self.current_task: Optional[str] = None
        self.is_working: bool = False

        # Communication
        self.in_base_range: bool = True
        self.last_sync_time: float = 0.0

        # Trajectory for visualization
        self.trajectory: List[np.ndarray] = [self.position.copy()]

    def update(self, dt: float, target: Optional[np.ndarray],
               terrain: LunarTerrain) -> None:
        """
        Update rover position and state.

        Args:
            dt: Time step in seconds
            target: Target position to move toward (or None to stop)
            terrain: Terrain for obstacle avoidance
        """
        if target is None:
            # Brake to stop
            speed = np.linalg.norm(self.velocity)
            if speed > 0.0001:
                brake_accel = min(ROVER_ACCELERATION, speed / dt)
                self.velocity -= brake_accel * dt * self.velocity / speed
            else:
                self.velocity = np.zeros(2)
        else:
            # Move toward target
            to_target = target - self.position
            distance = np.linalg.norm(to_target)

            if distance > 0.01:  # More than 10m away
                # Desired direction
                desired_dir = to_target / distance

                # Simple obstacle avoidance: check if path is clear
                if terrain.is_in_crater(self.position + desired_dir * 0.1):
                    # Turn to avoid crater
                    perp = np.array([-desired_dir[1], desired_dir[0]])
                    desired_dir = perp  # Turn 90 degrees

                # Accelerate toward target
                current_speed = np.linalg.norm(self.velocity)

                # Slow down when approaching target
                target_speed = min(ROVER_MAX_SPEED, distance / 10)

                if current_speed < target_speed:
                    # Accelerate
                    self.velocity += ROVER_ACCELERATION * dt * desired_dir
                else:
                    # Coast or brake
                    self.velocity = target_speed * desired_dir

                # Cap speed
                speed = np.linalg.norm(self.velocity)
                if speed > ROVER_MAX_SPEED:
                    self.velocity = ROVER_MAX_SPEED * self.velocity / speed

        # Update position
        new_position = self.position + self.velocity * dt

        # Check terrain collision
        if not terrain.is_in_crater(new_position):
            self.position = new_position
        else:
            # Stop at crater edge
            self.velocity = np.zeros(2)

        # Keep within terrain bounds
        half = TERRAIN_SIZE / 2
        self.position = np.clip(self.position, -half, half)

        # Update heading from velocity
        if np.linalg.norm(self.velocity) > 0.0001:
            self.heading = math.atan2(self.velocity[1], self.velocity[0])

        # Record trajectory
        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > TRAIL_LENGTH:
            self.trajectory.pop(0)

    def work_on_task(self, task: ConstructionTask, dt: float) -> None:
        """Perform work on a construction task."""
        if not self.is_working:
            return

        # Check if at task location
        dist = np.linalg.norm(self.position - task.position)
        if dist < 0.05:  # Within 50m
            work_done = TASK_WORK_RATE * dt
            task.add_work(work_done)

            # Update CRDT progress
            self.crdt_state.add_progress(task.task_id, int(work_done))

    def sync_with(self, other_state: CRDTState) -> None:
        """Merge another rover's CRDT state."""
        self.crdt_state.merge(other_state)


# =============================================================================
# COMMUNICATION MODEL
# =============================================================================

class GroundCommsModel:
    """
    Communication model for lunar ground operations.

    Features:
    - Base station with limited range
    - Line-of-sight blocking by terrain
    - Rover-to-rover relay capability
    - Random partition events
    """

    def __init__(self, terrain: LunarTerrain, rng: random.Random):
        self.terrain = terrain
        self.rng = rng
        self.partition_end_time = 0.0
        self.reliability = RELIABILITY

    def can_communicate(self, rover: LunarRover,
                        current_time: float) -> bool:
        """Check if rover can communicate with base."""
        # Check for partition event
        if current_time < self.partition_end_time:
            return False

        # Check distance to base
        dist_to_base = np.linalg.norm(rover.position - BASE_POSITION)
        if dist_to_base > BASE_COMM_RANGE:
            return False

        # Check line-of-sight
        if self.terrain.blocks_line_of_sight(rover.position, BASE_POSITION):
            return False

        # Random reliability
        return self.rng.random() < self.reliability

    def can_rovers_communicate(self, rover1: LunarRover, rover2: LunarRover,
                                current_time: float) -> bool:
        """Check if two rovers can communicate directly."""
        if current_time < self.partition_end_time:
            return False

        dist = np.linalg.norm(rover1.position - rover2.position)
        if dist > BASE_COMM_RANGE:  # Same range as base
            return False

        if self.terrain.blocks_line_of_sight(rover1.position, rover2.position):
            return False

        return self.rng.random() < self.reliability

    def maybe_start_partition(self, current_time: float) -> None:
        """Randomly start a communication partition."""
        if self.rng.random() < PARTITION_PROBABILITY:
            duration = self.rng.uniform(*PARTITION_DURATION_RANGE)
            self.partition_end_time = current_time + duration
            logger.info(f"[{current_time:.0f}s] PARTITION started - duration {duration:.0f}s")

    def is_partitioned(self, current_time: float) -> bool:
        """Check if currently in partition."""
        return current_time < self.partition_end_time


# =============================================================================
# GROUND SIMULATION
# =============================================================================

@dataclass
class GroundSimulationState:
    """Snapshot of simulation state."""
    time: float
    rovers: List[Dict]
    tasks: List[Dict]
    completed_count: int
    partitioned: bool


class GroundSimulation:
    """
    Main simulation for lunar ground operations.

    Coordinates rovers performing construction tasks with CRDT state
    synchronization through the lunar base.
    """

    def __init__(self, num_rovers: int = NUM_ROVERS, num_tasks: int = 5,
                 seed: int = 42):
        self.rng = random.Random(seed)
        np.random.seed(seed)

        # Initialize terrain
        self.terrain = LunarTerrain(
            TERRAIN_SIZE, NUM_CRATERS, NUM_HILLS, self.rng
        )

        # Initialize communication
        self.comms = GroundCommsModel(self.terrain, self.rng)

        # Initialize rovers
        self.rovers = self._init_rovers(num_rovers)

        # Initialize tasks
        self.tasks = self._init_tasks(num_tasks)

        # Simulation state
        self.current_time = 0.0
        self.max_time = MAX_SIMULATION_TIME

        logger.info(f"Initialized ground simulation:")
        logger.info(f"  Rovers: {num_rovers}")
        logger.info(f"  Tasks: {num_tasks}")
        logger.info(f"  Terrain: {len(self.terrain.craters)} craters, "
                   f"{len(self.terrain.hills)} hills")

    def _init_rovers(self, n: int) -> List[LunarRover]:
        """Create rovers positioned around the base."""
        rovers = []

        for i in range(n):
            # Position rovers in a circle around base
            angle = 2 * math.pi * i / n
            radius = 0.5  # 500m from base
            pos = np.array([
                radius * math.cos(angle),
                radius * math.sin(angle)
            ])

            rover = LunarRover(f"rover_{i}", pos)
            rovers.append(rover)

            logger.info(f"  Rover {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}) km")

        return rovers

    def _init_tasks(self, n: int) -> Dict[str, ConstructionTask]:
        """Create construction tasks at various locations."""
        tasks = {}

        task_types = ["solar_panel", "regolith", "transport"]

        for i in range(n):
            # Position tasks around the area
            angle = 2 * math.pi * i / n + 0.3  # Offset from rover positions
            radius = self.rng.uniform(1.5, 4.0)  # 1.5-4 km from base

            # Avoid placing in craters
            for attempt in range(10):
                pos = np.array([
                    radius * math.cos(angle),
                    radius * math.sin(angle)
                ])
                if not self.terrain.is_in_crater(pos):
                    break
                angle += 0.2

            task_type = task_types[i % len(task_types)]
            task_id = f"task_{i}_{task_type}"

            tasks[task_id] = ConstructionTask(
                task_id=task_id,
                task_type=task_type,
                position=pos,
                work_required=TASK_COMPLETION
            )

            logger.info(f"  Task {i}: {task_type} at ({pos[0]:.2f}, {pos[1]:.2f}) km")

        return tasks

    def step(self, dt: float) -> GroundSimulationState:
        """
        Advance simulation by dt seconds.

        Steps:
        1. Update rover positions
        2. Check communication and sync CRDT
        3. Assign tasks to idle rovers
        4. Update task progress
        5. Check completion
        """
        timestamp = int(self.current_time)

        # Maybe start partition
        if timestamp % int(SYNC_INTERVAL) == 0:
            self.comms.maybe_start_partition(self.current_time)

        # Update each rover
        for rover in self.rovers:
            # Determine target position
            target = None

            if rover.current_task:
                task = self.tasks.get(rover.current_task)
                if task and not task.is_complete:
                    target = task.position
                    rover.is_working = True
                else:
                    rover.current_task = None
                    rover.is_working = False
            else:
                # No task - try to get one
                rover.is_working = False
                new_task = self._select_task(rover, timestamp)
                if new_task:
                    rover.current_task = new_task
                    target = self.tasks[new_task].position

            # Update rover position
            rover.update(dt, target, self.terrain)

            # Update CRDT position
            pos_3d = Vector3(rover.position[0], 0, rover.position[1])
            rover.crdt_state.update_position(rover.rover_id, pos_3d, timestamp)

            # Work on task if at location
            if rover.current_task and rover.is_working:
                task = self.tasks.get(rover.current_task)
                if task:
                    rover.work_on_task(task, dt)

                    # Check completion
                    if task.is_complete:
                        rover.crdt_state.mark_task_complete(
                            rover.current_task, timestamp
                        )
                        logger.info(f"[{self.current_time:.0f}s] {rover.rover_id} "
                                   f"completed {rover.current_task}")
                        rover.current_task = None
                        rover.is_working = False

        # Synchronize CRDT states
        self._process_syncs(timestamp)

        # Advance time
        self.current_time += dt

        return self._get_state()

    def _select_task(self, rover: LunarRover, timestamp: int) -> Optional[str]:
        """Select an unclaimed, incomplete task for the rover."""
        available = []

        for task_id, task in self.tasks.items():
            if task.is_complete:
                continue
            if task_id in rover.crdt_state.completed_tasks:
                continue
            if rover.crdt_state.is_task_claimed_by_other(task_id, rover.rover_id):
                continue

            # Calculate distance
            dist = np.linalg.norm(rover.position - task.position)
            available.append((dist, task_id))

        if not available:
            return None

        # Pick closest task
        available.sort()
        task_id = available[0][1]

        # Claim it
        if rover.crdt_state.claim_task(task_id, rover.rover_id, timestamp):
            logger.info(f"[{self.current_time:.0f}s] {rover.rover_id} claimed {task_id}")
            return task_id

        return None

    def _process_syncs(self, timestamp: int) -> None:
        """Process CRDT synchronization between rovers and base."""
        # Check which rovers can communicate
        for rover in self.rovers:
            rover.in_base_range = self.comms.can_communicate(
                rover, self.current_time
            )

        # Rovers in base range sync with each other (via base)
        in_range = [r for r in self.rovers if r.in_base_range]
        for i, rover_a in enumerate(in_range):
            for rover_b in in_range[i+1:]:
                rover_a.sync_with(rover_b.crdt_state)
                rover_b.sync_with(rover_a.crdt_state)

        # Direct rover-to-rover sync for those out of base range
        out_of_range = [r for r in self.rovers if not r.in_base_range]
        for i, rover_a in enumerate(out_of_range):
            for rover_b in out_of_range[i+1:]:
                if self.comms.can_rovers_communicate(
                    rover_a, rover_b, self.current_time
                ):
                    rover_a.sync_with(rover_b.crdt_state)
                    rover_b.sync_with(rover_a.crdt_state)

    def _get_state(self) -> GroundSimulationState:
        """Get current simulation state snapshot."""
        completed = sum(1 for t in self.tasks.values() if t.is_complete)

        return GroundSimulationState(
            time=self.current_time,
            rovers=[{
                'id': r.rover_id,
                'position': r.position.tolist(),
                'task': r.current_task,
                'in_range': r.in_base_range
            } for r in self.rovers],
            tasks=[{
                'id': t.task_id,
                'type': t.task_type,
                'position': t.position.tolist(),
                'progress': t.work_completed / t.work_required,
                'complete': t.is_complete
            } for t in self.tasks.values()],
            completed_count=completed,
            partitioned=self.comms.is_partitioned(self.current_time)
        )


# =============================================================================
# 2D VISUALIZATION
# =============================================================================

class GroundAnimator:
    """2D top-down visualization of ground operations."""

    def __init__(self, simulation: GroundSimulation):
        self.sim = simulation
        self.fig, self.ax = plt.subplots(figsize=(12, 10))

        # Colors
        self.colors = {
            'base': '#FFD700',      # Gold
            'rover_ok': '#00FF00',  # Green (in range)
            'rover_out': '#FF4444', # Red (out of range)
            'task_pending': '#00FFFF',  # Cyan
            'task_active': '#FFFF00',   # Yellow
            'task_done': '#888888',     # Gray
            'crater': '#333333',
            'hill': '#666666',
            'trail': '#4444FF'
        }

    def setup_scene(self) -> None:
        """Initialize the visualization."""
        self.ax.set_xlim(-TERRAIN_SIZE/2 - 1, TERRAIN_SIZE/2 + 1)
        self.ax.set_ylim(-TERRAIN_SIZE/2 - 1, TERRAIN_SIZE/2 + 1)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_title('Lunar Base Ground Operations - CRDT Coordination')
        self.ax.grid(True, alpha=0.3)

        # Draw terrain features
        for crater in self.sim.terrain.craters:
            circle = Circle(crater.center, crater.radius,
                          color=self.colors['crater'], alpha=0.5)
            self.ax.add_patch(circle)

        for hill in self.sim.terrain.hills:
            circle = Circle(hill.center, hill.radius,
                          color=self.colors['hill'], alpha=0.3)
            self.ax.add_patch(circle)

        # Draw base
        base_marker = Circle(BASE_POSITION, 0.3,
                           color=self.colors['base'], zorder=10)
        self.ax.add_patch(base_marker)
        self.ax.annotate('BASE', BASE_POSITION,
                        textcoords='offset points', xytext=(10, 10),
                        fontweight='bold')

        # Draw communication range
        comm_circle = Circle(BASE_POSITION, BASE_COMM_RANGE,
                           fill=False, linestyle='--',
                           color=self.colors['base'], alpha=0.5)
        self.ax.add_patch(comm_circle)

        # Initialize rover markers
        self.rover_markers = []
        self.rover_trails = []
        for i, rover in enumerate(self.sim.rovers):
            marker, = self.ax.plot([], [], 'o', markersize=10,
                                  label=rover.rover_id)
            trail, = self.ax.plot([], [], '-', alpha=0.4, linewidth=1)
            self.rover_markers.append(marker)
            self.rover_trails.append(trail)

        # Initialize task markers
        self.task_markers = {}
        for task_id, task in self.sim.tasks.items():
            marker, = self.ax.plot(task.position[0], task.position[1],
                                  's', markersize=12,
                                  color=self.colors['task_pending'])
            self.task_markers[task_id] = marker

        # Status text
        self.status_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # Legend
        self.ax.legend(loc='upper right')

    def update(self, frame: int) -> List:
        """Update animation frame."""
        # Step simulation
        state = self.sim.step(SIMULATION_DT)

        # Update rover positions and colors
        for i, rover in enumerate(self.sim.rovers):
            # Position
            self.rover_markers[i].set_data([rover.position[0]],
                                           [rover.position[1]])

            # Color based on communication status
            color = (self.colors['rover_ok'] if rover.in_base_range
                    else self.colors['rover_out'])
            self.rover_markers[i].set_color(color)

            # Trail
            if len(rover.trajectory) > 1:
                trail = np.array(rover.trajectory)
                self.rover_trails[i].set_data(trail[:, 0], trail[:, 1])
                self.rover_trails[i].set_color(color)

        # Update task markers
        for task_id, task in self.sim.tasks.items():
            if task.is_complete:
                color = self.colors['task_done']
            elif task.assigned_rover:
                color = self.colors['task_active']
            else:
                color = self.colors['task_pending']
            self.task_markers[task_id].set_color(color)

        # Update status
        completed = sum(1 for t in self.sim.tasks.values() if t.is_complete)
        partition_status = "PARTITIONED" if state.partitioned else "Connected"
        in_range = sum(1 for r in self.sim.rovers if r.in_base_range)

        status = (
            f"Time: {self.sim.current_time:.0f}s ({self.sim.current_time/60:.1f} min)\n"
            f"Tasks: {completed}/{len(self.sim.tasks)} complete\n"
            f"Rovers in range: {in_range}/{len(self.sim.rovers)}\n"
            f"Comm: {partition_status}"
        )
        self.status_text.set_text(status)

        return self.rover_markers + self.rover_trails + list(self.task_markers.values())

    def run(self) -> None:
        """Run the animation."""
        self.setup_scene()

        frames = int(self.sim.max_time / SIMULATION_DT)
        ani = FuncAnimation(
            self.fig, self.update, frames=frames,
            interval=ANIMATION_INTERVAL, blit=False
        )

        plt.tight_layout()
        plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the lunar ground operations simulation."""
    print("=" * 60)
    print("CRDT Ground Operations Simulation - Lunar Base")
    print("=" * 60)
    print(f"Rovers: {NUM_ROVERS}")
    print(f"Base comm range: {BASE_COMM_RANGE} km")
    print(f"Terrain: {TERRAIN_SIZE}x{TERRAIN_SIZE} km")
    print("=" * 60)

    sim = GroundSimulation(num_rovers=NUM_ROVERS, num_tasks=5, seed=42)
    animator = GroundAnimator(sim)
    animator.run()


if __name__ == "__main__":
    main()
