#!/usr/bin/env python3
"""
Lunar Orbital Refueling Depot Simulation

Demonstrates CRDT-based coordination for autonomous space robotics.
5 servicing robots coordinate to refuel client spacecraft in lunar orbit,
handling communication blackouts when passing behind the Moon.

Features:
- Two-body orbital mechanics with numerical integration
- Low-thrust continuous propulsion with Tsiolkovsky fuel consumption
- CRDT task claiming prevents duplicate servicing
- Realistic partition events (robots behind Moon lose comms)
- Interactive 3D matplotlib animation

Run: python orbital_simulation.py
"""

from __future__ import annotations

import sys
import math
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Add project root to path for imports
sys.path.insert(0, '.')
from src.crdt import CRDTState, Vector3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

MU_MOON = 4902.8          # Moon gravitational parameter (km³/s²)
MOON_RADIUS = 1737.4      # Moon radius (km)
ORBIT_ALTITUDE = 100.0    # Orbital altitude above Moon surface (km)
ORBIT_RADIUS = MOON_RADIUS + ORBIT_ALTITUDE  # Total orbital radius (km)

# Orbital period for circular orbit: T = 2π * sqrt(a³/μ)
ORBITAL_PERIOD = 2 * np.pi * np.sqrt(ORBIT_RADIUS**3 / MU_MOON)  # ~7138 seconds

# Propulsion parameters
ISP = 300.0               # Specific impulse (seconds) - bipropellant
G0 = 9.81e-3              # Standard gravity (km/s²)
DRY_MASS = 50.0           # Robot dry mass without fuel (kg)
INITIAL_FUEL = 500.0      # Initial fuel mass (kg) - generous for demo
MAX_THRUST = 1.0          # Maximum thrust (kN) - high thrust for orbital maneuvers
MIN_ALTITUDE = 30.0       # Minimum safe altitude above Moon (km)
MAX_ALTITUDE = 1000.0     # Maximum altitude for operations (km)

# Communication parameters (Lunar scenario)
RELIABILITY = 0.80        # Message success probability
SYNC_INTERVAL = 100.0     # Seconds between sync attempts
PARTITION_PROBABILITY = 0.03  # Per-sync chance of partition starting
PARTITION_DURATION_RANGE = (100.0, 400.0)  # Partition duration (seconds)

# Simulation parameters
SIMULATION_DT = 10.0      # Time step for animation (seconds)
MAX_SIMULATION_TIME = 4 * 3600  # Maximum simulation time (4 hours)
RENDEZVOUS_DISTANCE = 10.0  # Distance to dock (km) - within LIDAR/vision range
REFUELING_DURATION = 200.0  # Seconds to complete refueling

# Visualization parameters
TRAIL_LENGTH = 100        # Number of points in orbital trail
ANIMATION_INTERVAL = 50   # Milliseconds between frames


# =============================================================================
# ORBITAL MECHANICS
# =============================================================================

@dataclass
class OrbitalElements:
    """
    Classical Keplerian orbital elements.

    These six parameters uniquely define an orbit around a central body.
    For this simulation, we use circular orbits (e=0) for simplicity.
    """
    semi_major_axis: float    # a - orbit size (km)
    eccentricity: float       # e - orbit shape (0 = circular)
    inclination: float        # i - tilt from equator (radians)
    raan: float               # Ω - right ascension of ascending node (radians)
    arg_periapsis: float      # ω - argument of periapsis (radians)
    true_anomaly: float       # ν - position in orbit (radians)

    def to_cartesian(self, mu: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert orbital elements to Cartesian position and velocity vectors.

        Uses the standard algorithm from Vallado's "Fundamentals of Astrodynamics".

        Args:
            mu: Gravitational parameter of central body (km³/s²)

        Returns:
            Tuple of (position, velocity) as numpy arrays in km and km/s
        """
        a = self.semi_major_axis
        e = self.eccentricity
        i = self.inclination
        raan = self.raan
        w = self.arg_periapsis
        nu = self.true_anomaly

        # Semi-latus rectum
        p = a * (1 - e**2) if e < 1 else a * (e**2 - 1)

        # Position magnitude
        r_mag = p / (1 + e * np.cos(nu))

        # Position in orbital plane (perifocal frame)
        r_pf = np.array([
            r_mag * np.cos(nu),
            r_mag * np.sin(nu),
            0.0
        ])

        # Velocity in orbital plane
        v_pf = np.sqrt(mu / p) * np.array([
            -np.sin(nu),
            e + np.cos(nu),
            0.0
        ])

        # Rotation matrices
        R3_raan = self._rotation_z(-raan)
        R1_i = self._rotation_x(-i)
        R3_w = self._rotation_z(-w)

        # Combined rotation: perifocal to inertial
        Q = R3_raan @ R1_i @ R3_w

        # Transform to inertial frame
        r_eci = Q @ r_pf
        v_eci = Q @ v_pf

        return r_eci, v_eci

    @staticmethod
    def _rotation_x(angle: float) -> np.ndarray:
        """Rotation matrix about X-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        """Rotation matrix about Z-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])


def orbital_dynamics(
    t: float,
    state: np.ndarray,
    mu: float,
    thrust: np.ndarray
) -> np.ndarray:
    """
    Two-body orbital dynamics with thrust perturbation.

    This is the ODE function for scipy.integrate.solve_ivp.
    Implements the equation of motion:

        r̈ = -μ r / |r|³ + thrust/mass

    With mass flow rate from rocket equation:

        ṁ = -|thrust| / (Isp * g0)

    Args:
        t: Current time (seconds)
        state: State vector [x, y, z, vx, vy, vz, fuel_mass]
        mu: Gravitational parameter (km³/s²)
        thrust: Thrust vector (kN)

    Returns:
        Derivative of state vector
    """
    r = state[:3]  # Position (km)
    v = state[3:6]  # Velocity (km/s)
    m_fuel = state[6]  # Fuel mass (kg)

    # Total mass
    m_total = DRY_MASS + max(0, m_fuel)

    # Distance from center
    r_norm = np.linalg.norm(r)

    # Prevent division by zero (collision with Moon) - clamp silently
    if r_norm < MOON_RADIUS:
        r_norm = MOON_RADIUS

    # Gravitational acceleration: a = -μr/|r|³
    a_grav = -mu * r / (r_norm ** 3)

    # Thrust acceleration (only if we have fuel)
    thrust_mag = np.linalg.norm(thrust)
    if m_fuel > 0 and thrust_mag > 0:
        a_thrust = thrust / m_total
        # Mass flow rate: ṁ = -|F| / (Isp * g0)
        m_dot = -thrust_mag / (ISP * G0)
    else:
        a_thrust = np.zeros(3)
        m_dot = 0.0

    # Total acceleration
    a_total = a_grav + a_thrust

    # Return derivatives: [dr/dt, dv/dt, dm/dt]
    return np.concatenate([v, a_total, [m_dot]])


# =============================================================================
# CLIENT SPACECRAFT (Tasks to service)
# =============================================================================

@dataclass
class ClientSpacecraft:
    """
    A spacecraft in lunar orbit that needs refueling.

    These are the "tasks" that servicing robots must complete.
    Each client has its own orbit and position that evolves over time.
    """
    spacecraft_id: str
    orbital_elements: OrbitalElements
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fuel_needed: float = REFUELING_DURATION  # Seconds of docking required
    fuel_received: float = 0.0  # Progress toward completion
    is_complete: bool = False
    servicing_robot: Optional[str] = None  # Robot currently docking

    def __post_init__(self):
        """Initialize position and velocity from orbital elements."""
        self.position, self.velocity = self.orbital_elements.to_cartesian(MU_MOON)

    def propagate(self, dt: float) -> None:
        """
        Propagate client spacecraft orbit forward by dt seconds.

        Uses the same orbital dynamics as robots, but with no thrust.
        """
        state0 = np.concatenate([self.position, self.velocity, [0.0]])

        sol = solve_ivp(
            orbital_dynamics,
            (0, dt),
            state0,
            args=(MU_MOON, np.zeros(3)),
            method='RK45',
            max_step=dt/2
        )

        self.position = sol.y[:3, -1]
        self.velocity = sol.y[3:6, -1]

    def receive_fuel(self, amount: float) -> None:
        """
        Receive fuel from a servicing robot.

        Args:
            amount: Seconds worth of refueling progress
        """
        self.fuel_received += amount
        if self.fuel_received >= self.fuel_needed:
            self.is_complete = True


# =============================================================================
# ORBITAL ROBOT (Servicing robot with CRDT)
# =============================================================================

class OrbitalRobot:
    """
    Autonomous servicing robot with orbital dynamics and CRDT coordination.

    Each robot maintains:
    - Orbital state (position, velocity, fuel)
    - CRDT state for distributed coordination
    - Task assignment and progress tracking
    - Trajectory history for visualization
    """

    def __init__(self, robot_id: str, orbital_elements: OrbitalElements):
        """
        Initialize a robot in the given orbit.

        Args:
            robot_id: Unique identifier for this robot
            orbital_elements: Initial orbital elements
        """
        self.robot_id = robot_id

        # Orbital state
        pos, vel = orbital_elements.to_cartesian(MU_MOON)
        self.position = pos
        self.velocity = vel
        self.fuel_mass = INITIAL_FUEL

        # CRDT state for coordination
        self.crdt_state = CRDTState(robot_id)

        # Task management
        self.current_task: Optional[str] = None
        self.is_docking: bool = False

        # Control
        self.thrust_command = np.zeros(3)

        # History for visualization
        self.trajectory: List[np.ndarray] = [self.position.copy()]

        # Metrics
        self.total_delta_v = 0.0

    def propagate(self, dt: float, target_pos: Optional[np.ndarray] = None,
                  target_vel: Optional[np.ndarray] = None) -> None:
        """
        Propagate orbital dynamics using real physics.

        Uses proper orbital mechanics:
        - Full gravitational dynamics via solve_ivp (two-body problem)
        - Phasing maneuvers: thrust prograde/retrograde to change orbital period
        - Proximity operations: Hill-Clohessy-Wiltshire relative motion

        Physics:
        - To catch target ahead: thrust retrograde (lower orbit → faster angular rate)
        - To let target catch up: thrust prograde (higher orbit → slower angular rate)
        - Tsiolkovsky equation for fuel consumption: Δv = Isp·g₀·ln(m₀/m₁)

        Args:
            dt: Time step in seconds
            target_pos: Optional target position for rendezvous
            target_vel: Optional target velocity for matching
        """
        # Calculate thrust vector
        thrust = np.zeros(3)

        if target_pos is not None and self.fuel_mass > 0:
            rel_pos = target_pos - self.position
            distance = np.linalg.norm(rel_pos)

            if distance < RENDEZVOUS_DISTANCE * 1.5:  # Lock position at 15km
                # DOCKING: Match position and velocity with target
                self.is_docking = True
                self.position = target_pos.copy()
                if target_vel is not None:
                    self.velocity = target_vel.copy()
                self.trajectory.append(self.position.copy())
                if len(self.trajectory) > TRAIL_LENGTH:
                    self.trajectory.pop(0)
                return

            self.is_docking = False

            # Calculate phase angle in X-Z plane (polar orbit)
            robot_phase = np.arctan2(self.position[2], self.position[0])
            target_phase = np.arctan2(target_pos[2], target_pos[0])

            # Phase difference: positive = target is ahead
            phase_diff = target_phase - robot_phase
            # Normalize to [-π, π]
            while phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            while phase_diff < -np.pi:
                phase_diff += 2 * np.pi

            # Velocity direction (prograde is perpendicular to radial, in orbit direction)
            r_vec = self.position / np.linalg.norm(self.position)
            # For polar orbit in X-Z plane, prograde is perpendicular
            prograde = np.array([-r_vec[2], 0, r_vec[0]])
            prograde = prograde / np.linalg.norm(prograde)

            if distance < 50:
                # PROXIMITY OPERATIONS: Direct thrust toward target
                # At close range, use direct approach (approximates CW equations)
                rel_vel = target_vel - self.velocity if target_vel is not None else -self.velocity
                rel_speed = np.linalg.norm(rel_vel)

                # Proportional navigation: thrust to close distance and match velocity
                approach_dir = rel_pos / distance
                closing_speed = -np.dot(rel_vel, approach_dir)  # Positive = closing

                # If closing too fast, brake; if too slow, accelerate
                desired_closing = min(0.1, distance / 100)  # km/s, slower when closer
                speed_error = desired_closing - closing_speed

                # Thrust toward target + velocity correction
                thrust_mag = min(MAX_THRUST, 0.5)  # Gentle thrust for prox ops
                thrust = thrust_mag * (0.7 * approach_dir + 0.3 * rel_vel / max(rel_speed, 0.01))
                thrust = thrust / np.linalg.norm(thrust) * thrust_mag

            else:
                # VELOCITY MATCHING RENDEZVOUS
                # Two-step approach:
                # 1. Match target's velocity (reduces relative drift to zero)
                # 2. Apply small corrections toward target

                if target_vel is not None:
                    vel_error = self.velocity - target_vel
                else:
                    vel_error = np.zeros(3)
                vel_error_mag = np.linalg.norm(vel_error)

                # Position error unit vector
                approach_dir = rel_pos / distance

                # Priority 1: Match velocity (drift nulling)
                # Priority 2: Close distance

                if vel_error_mag > 0.01:  # More than 10 m/s velocity mismatch
                    # Thrust to match target velocity
                    thrust_mag = min(MAX_THRUST * 0.15, vel_error_mag * 20)
                    thrust = -thrust_mag * vel_error / vel_error_mag
                else:
                    # Velocity matched - close distance with proportional control
                    # Thrust scales with distance: stronger far, gentler close
                    thrust_scale = min(0.15, distance / 500)  # 0.15 at 75km+
                    thrust_mag = MAX_THRUST * thrust_scale
                    thrust = thrust_mag * approach_dir

                    # Reduce thrust as we get very close to avoid overshoot
                    if distance < 15:
                        thrust = approach_dir * MAX_THRUST * 0.03

        # Store thrust command for visualization
        self.thrust_command = thrust

        # Integrate orbital dynamics with thrust
        state0 = np.concatenate([self.position, self.velocity, [self.fuel_mass]])

        sol = solve_ivp(
            orbital_dynamics,
            (0, dt),
            state0,
            args=(MU_MOON, thrust),
            method='RK45',
            max_step=dt / 2
        )

        # Update state from integration
        self.position = sol.y[:3, -1]
        self.velocity = sol.y[3:6, -1]
        new_fuel = sol.y[6, -1]

        # Track delta-v
        delta_v = ISP * G0 * np.log(max(self.fuel_mass + DRY_MASS, DRY_MASS + 0.01) /
                                     max(new_fuel + DRY_MASS, DRY_MASS + 0.01))
        self.total_delta_v += delta_v
        self.fuel_mass = max(0, new_fuel)

        # Record trajectory
        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > TRAIL_LENGTH:
            self.trajectory.pop(0)

    def compute_thrust(
        self,
        clients: Dict[str, ClientSpacecraft],
        timestamp: int
    ) -> None:
        """
        Decide thrust direction based on CRDT state and task locations.

        Uses simplified direct intercept guidance for demonstration.
        Real orbital rendezvous would use Hill's equations, but this
        simplified version focuses on demonstrating CRDT coordination.

        Args:
            clients: Dictionary of client spacecraft
            timestamp: Current simulation timestamp
        """
        # No fuel = no thrust
        if self.fuel_mass <= 0:
            self.thrust_command = np.zeros(3)
            return

        # Find or select a task
        if self.current_task is None:
            self.current_task = self._select_task(clients, timestamp)

        if self.current_task is None:
            self.thrust_command = np.zeros(3)
            return

        # Get target spacecraft
        target = clients.get(self.current_task)
        if target is None or target.is_complete:
            self.current_task = None
            self.thrust_command = np.zeros(3)
            return

        # Compute relative position and velocity
        rel_pos = target.position - self.position
        rel_vel = target.velocity - self.velocity
        distance = np.linalg.norm(rel_pos)

        # Check if close enough to dock
        if distance < RENDEZVOUS_DISTANCE:
            self.is_docking = True
            # Match velocity for station-keeping
            rel_vel_mag = np.linalg.norm(rel_vel)
            if rel_vel_mag > 0.001:  # More than 1 m/s relative velocity
                self.thrust_command = -MAX_THRUST * rel_vel / rel_vel_mag
            else:
                self.thrust_command = np.zeros(3)
            return

        self.is_docking = False

        # Simplified guidance: apply small constant acceleration toward target
        # This is not realistic orbital mechanics but demonstrates CRDT coordination
        if distance > 0:
            # Direct thrust toward target - small magnitude for stability
            thrust_dir = rel_pos / distance

            # Scale thrust based on distance - more gentle approach
            thrust_scale = min(1.0, distance / 500.0)  # Full thrust beyond 500 km

            self.thrust_command = thrust_scale * MAX_THRUST * thrust_dir
        else:
            self.thrust_command = np.zeros(3)

    def _select_task(
        self,
        clients: Dict[str, ClientSpacecraft],
        timestamp: int
    ) -> Optional[str]:
        """
        Select the best available task based on CRDT state.

        Considers:
        - Tasks not yet completed (not in completed_tasks)
        - Tasks not claimed by others (FWW check)
        - Distance (closest task preferred)

        Args:
            clients: Dictionary of client spacecraft
            timestamp: Current simulation timestamp

        Returns:
            Selected task ID or None if no suitable task found
        """
        best_task = None
        best_score = float('inf')

        for client_id, client in clients.items():
            # Skip completed tasks
            if client.is_complete or client_id in self.crdt_state.completed_tasks:
                continue

            # Skip tasks claimed by others
            if self.crdt_state.is_task_claimed_by_other(client_id, self.robot_id):
                continue

            # Calculate score (lower is better) - use angular separation in orbit
            # This is more relevant for orbital rendezvous than linear distance
            distance = np.linalg.norm(client.position - self.position)

            # For co-orbital targets, phasing maneuvers are cheap
            # Score primarily by distance, with small velocity penalty
            rel_vel = np.linalg.norm(client.velocity - self.velocity)
            score = distance + 10 * rel_vel

            if score < best_score:
                best_score = score
                best_task = client_id

        # Claim the task if we found one
        if best_task:
            self.crdt_state.claim_task(best_task, self.robot_id, timestamp)

        return best_task

    def sync_with(self, other_state: CRDTState) -> None:
        """
        Merge another robot's CRDT state into ours.

        This is the core CRDT operation that enables distributed coordination.
        After merge, both robots have consistent knowledge of:
        - Which tasks are completed
        - Which tasks are claimed by whom
        - Progress on each task

        Args:
            other_state: CRDT state from another robot
        """
        self.crdt_state.merge(other_state)

        # If our current task was claimed by someone else first, release it
        if self.current_task:
            if self.crdt_state.is_task_claimed_by_other(self.current_task, self.robot_id):
                self.current_task = None
                self.is_docking = False

    def update_crdt_position(self, timestamp: int) -> None:
        """Update our position in CRDT state."""
        pos_v3 = Vector3(self.position[0], self.position[1], self.position[2])
        self.crdt_state.update_position(self.robot_id, pos_v3, timestamp)


# =============================================================================
# COMMUNICATION MODEL
# =============================================================================

class LunarCommsModel:
    """
    Communication model for Lunar scenario.

    Models:
    - Message reliability (80% success rate)
    - Random partition events (when robots are "behind the Moon")
    - Partition duration (100-400 seconds)
    """

    def __init__(self, rng: random.Random):
        """
        Initialize communication model.

        Args:
            rng: Random number generator for reproducibility
        """
        self.rng = rng
        self.reliability = RELIABILITY
        self.sync_interval = SYNC_INTERVAL
        self.partition_end_time = 0.0
        self.partition_active = False

        # Track partition events for visualization
        self.partition_history: List[Tuple[float, float]] = []

    def is_partitioned(self, t: float) -> bool:
        """Check if currently in a communication partition."""
        self.partition_active = t < self.partition_end_time
        return self.partition_active

    def maybe_start_partition(self, t: float) -> bool:
        """
        Possibly start a new partition event.

        Called at each sync interval. Has PARTITION_PROBABILITY chance
        of starting a partition.

        Args:
            t: Current simulation time

        Returns:
            True if a partition was started
        """
        if t < self.partition_end_time:
            return False  # Already in partition

        if self.rng.random() < PARTITION_PROBABILITY:
            duration = self.rng.uniform(*PARTITION_DURATION_RANGE)
            self.partition_end_time = t + duration
            self.partition_history.append((t, duration))
            logger.info(f"[{t:.0f}s] PARTITION started - duration {duration:.0f}s")
            return True
        return False

    def message_succeeds(self) -> bool:
        """Check if a message would be delivered successfully."""
        if self.partition_active:
            return False
        return self.rng.random() < self.reliability


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

@dataclass
class SimulationState:
    """Snapshot of simulation state for visualization."""
    time: float
    robot_positions: Dict[str, np.ndarray]
    robot_fuels: Dict[str, float]
    robot_tasks: Dict[str, Optional[str]]
    robot_docking: Dict[str, bool]
    client_positions: Dict[str, np.ndarray]
    client_progress: Dict[str, float]
    client_complete: Dict[str, bool]
    partition_active: bool
    completed_tasks: int
    total_tasks: int


class OrbitalSimulation:
    """
    Main simulation orchestrating orbital mechanics and CRDT coordination.

    Manages:
    - Multiple servicing robots
    - Multiple client spacecraft
    - Communication model with partitions
    - Time stepping and state updates
    """

    def __init__(
        self,
        num_robots: int = 5,
        num_clients: int = 5,
        seed: int = 42
    ):
        """
        Initialize simulation.

        Args:
            num_robots: Number of servicing robots
            num_clients: Number of client spacecraft to service
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.num_robots = num_robots
        self.num_clients = num_clients

        # Create robots in phased orbits
        self.robots = self._init_robots(num_robots)

        # Create client spacecraft in slightly different orbits
        self.clients = self._init_clients(num_clients)

        # Communication model
        self.comms = LunarCommsModel(self.rng)

        # Time management
        self.current_time = 0.0
        self.max_time = MAX_SIMULATION_TIME
        self.last_sync_time = 0.0

        # Metrics
        self.sync_attempts = 0
        self.sync_successes = 0
        self.partition_steps = 0

        logger.info(f"Initialized simulation with {num_robots} robots and {num_clients} clients")
        logger.info(f"Orbital period: {ORBITAL_PERIOD:.0f}s ({ORBITAL_PERIOD/60:.1f} min)")

    def _init_robots(self, n: int) -> List[OrbitalRobot]:
        """
        Create n robots in phased lunar orbits.

        All robots are in the same circular polar orbit, just phased
        at different positions around the orbit.
        """
        robots = []

        # Common orbital parameters - polar orbit for visibility
        inclination = np.radians(90)  # Polar orbit
        raan = 0

        for i in range(n):
            # Distribute robots evenly around orbit
            true_anomaly = np.radians(i * 360 / n)

            elements = OrbitalElements(
                semi_major_axis=ORBIT_RADIUS,
                eccentricity=0.0,  # Circular orbit
                inclination=inclination,
                raan=raan,
                arg_periapsis=0,
                true_anomaly=true_anomaly
            )

            robot = OrbitalRobot(f"robot_{i}", elements)
            robots.append(robot)

            logger.info(f"  Robot {i}: phase={np.degrees(true_anomaly):.0f}°")

        return robots

    def _init_clients(self, n: int) -> Dict[str, ClientSpacecraft]:
        """
        Create n client spacecraft in the same orbit as robots.

        Clients are at the same altitude but different phases,
        making rendezvous a phasing problem (realistic for co-orbital ops).
        """
        clients = {}

        # Same orbital parameters as robots
        inclination = np.radians(90)  # Polar orbit
        raan = 0

        for i in range(n):
            # Phase clients very close to robot positions
            # Offset by just 3 degrees (~100km) for achievable rendezvous
            true_anomaly = np.radians(3 + i * 72)

            elements = OrbitalElements(
                semi_major_axis=ORBIT_RADIUS,
                eccentricity=0.0,  # Circular orbit
                inclination=inclination,
                raan=raan,
                arg_periapsis=0,
                true_anomaly=true_anomaly
            )

            client_id = f"client_{i}"
            clients[client_id] = ClientSpacecraft(
                spacecraft_id=client_id,
                orbital_elements=elements,
                fuel_needed=REFUELING_DURATION
            )

            logger.info(f"  Client {i}: phase={np.degrees(true_anomaly):.0f}°")

        return clients

    def step(self, dt: float) -> SimulationState:
        """
        Advance simulation by dt seconds.

        Steps:
        1. Propagate all orbits (robots and clients)
        2. Update CRDT positions
        3. Process sync events if at sync interval
        4. Update task progress for docking robots
        5. Robots make thrust decisions
        6. Return current state for visualization

        Args:
            dt: Time step in seconds

        Returns:
            Current simulation state
        """
        timestamp = int(self.current_time)

        # 1. Propagate all orbits
        for robot in self.robots:
            # Get target position and velocity if robot has a task
            target_pos = None
            target_vel = None
            if robot.current_task and robot.current_task in self.clients:
                client = self.clients[robot.current_task]
                target_pos = client.position.copy()
                target_vel = client.velocity.copy()

            robot.propagate(dt, target_pos, target_vel)

        for client in self.clients.values():
            client.propagate(dt)

        # 2. Update CRDT positions
        for robot in self.robots:
            robot.update_crdt_position(timestamp)

        # 3. Process sync events
        if self.current_time - self.last_sync_time >= self.comms.sync_interval:
            self._process_syncs(timestamp)
            self.last_sync_time = self.current_time

        # Check partition status
        if self.comms.is_partitioned(self.current_time):
            self.partition_steps += 1

        # 4. Update task progress for docking robots
        self._update_docking_progress(dt, timestamp)

        # 5. Robots make thrust decisions
        for robot in self.robots:
            robot.compute_thrust(self.clients, timestamp)

        # Advance time
        self.current_time += dt

        # Return current state
        return self._get_state()

    def _process_syncs(self, timestamp: int) -> None:
        """
        Attempt peer-to-peer CRDT synchronization between robots.

        Each pair of robots attempts bidirectional sync.
        Success depends on partition status and message reliability.
        """
        # Check for partition events
        self.comms.maybe_start_partition(self.current_time)

        # Pairwise sync attempts
        for i, robot_a in enumerate(self.robots):
            for robot_b in self.robots[i+1:]:
                self.sync_attempts += 2  # Bidirectional

                # Check if communication succeeds
                if not self.comms.is_partitioned(self.current_time):
                    if self.comms.message_succeeds():
                        robot_a.sync_with(robot_b.crdt_state)
                        self.sync_successes += 1

                    if self.comms.message_succeeds():
                        robot_b.sync_with(robot_a.crdt_state)
                        self.sync_successes += 1

    def _update_docking_progress(self, dt: float, timestamp: int) -> None:
        """
        Update refueling progress for robots that are docking.

        A robot docking with a client spacecraft adds progress over time.
        When complete, the task is marked as done in CRDT state.
        """
        for robot in self.robots:
            if not robot.is_docking or robot.current_task is None:
                continue

            client = self.clients.get(robot.current_task)
            if client is None or client.is_complete:
                robot.is_docking = False
                robot.current_task = None
                continue

            # Check distance
            distance = np.linalg.norm(client.position - robot.position)
            if distance > RENDEZVOUS_DISTANCE * 2:
                robot.is_docking = False
                continue

            # Add refueling progress
            client.servicing_robot = robot.robot_id
            client.receive_fuel(dt)

            # Update CRDT progress
            robot.crdt_state.add_progress(robot.current_task, int(dt))

            # Check completion
            if client.is_complete:
                robot.crdt_state.mark_task_complete(robot.current_task, timestamp)
                logger.info(f"[{self.current_time:.0f}s] {robot.robot_id} completed "
                           f"{robot.current_task}")
                robot.current_task = None
                robot.is_docking = False

    def _get_state(self) -> SimulationState:
        """Get current simulation state snapshot."""
        completed = sum(1 for c in self.clients.values() if c.is_complete)

        return SimulationState(
            time=self.current_time,
            robot_positions={r.robot_id: r.position.copy() for r in self.robots},
            robot_fuels={r.robot_id: r.fuel_mass for r in self.robots},
            robot_tasks={r.robot_id: r.current_task for r in self.robots},
            robot_docking={r.robot_id: r.is_docking for r in self.robots},
            client_positions={c.spacecraft_id: c.position.copy()
                            for c in self.clients.values()},
            client_progress={c.spacecraft_id: c.fuel_received / c.fuel_needed
                           for c in self.clients.values()},
            client_complete={c.spacecraft_id: c.is_complete
                           for c in self.clients.values()},
            partition_active=self.comms.partition_active,
            completed_tasks=completed,
            total_tasks=len(self.clients)
        )

    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(c.is_complete for c in self.clients.values())

    def get_metrics(self) -> Dict:
        """Get simulation metrics."""
        return {
            "time_elapsed": self.current_time,
            "tasks_completed": sum(1 for c in self.clients.values() if c.is_complete),
            "total_tasks": len(self.clients),
            "sync_attempts": self.sync_attempts,
            "sync_successes": self.sync_successes,
            "partition_time": self.partition_steps * SIMULATION_DT,
            "total_delta_v": sum(r.total_delta_v for r in self.robots),
            "avg_fuel_remaining": np.mean([r.fuel_mass for r in self.robots])
        }


# =============================================================================
# 3D VISUALIZATION
# =============================================================================

class OrbitalAnimator:
    """
    3D matplotlib animation for orbital simulation.

    Displays:
    - Moon as wireframe sphere
    - Robots as colored markers (green=connected, red=partitioned)
    - Client spacecraft as diamond markers
    - Orbital trails
    - Status information
    """

    def __init__(self, simulation: OrbitalSimulation):
        """
        Initialize animator.

        Args:
            simulation: The simulation to animate
        """
        self.sim = simulation

        # Create figure with dark background
        self.fig = plt.figure(figsize=(16, 10), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='black')

        # Storage for plot elements
        self.robot_plots = []
        self.robot_trails = []
        self.client_plots = []
        self.docking_lines = []
        self.moon_surface = None
        self.status_text = None
        self.info_text = None

        # Animation state
        self.frame_count = 0
        self.running = True

    def setup_scene(self) -> None:
        """Initialize all visual elements."""
        # Draw Moon as wireframe sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = MOON_RADIUS * np.outer(np.cos(u), np.sin(v))
        y = MOON_RADIUS * np.outer(np.sin(u), np.sin(v))
        z = MOON_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

        self.moon_surface = self.ax.plot_wireframe(
            x, y, z, color='gray', alpha=0.2, linewidth=0.5
        )

        # Initialize robot markers and trails
        colors = plt.cm.Set1(np.linspace(0, 1, self.sim.num_robots))

        for i, robot in enumerate(self.sim.robots):
            # Robot marker
            plot, = self.ax.plot(
                [robot.position[0]],
                [robot.position[1]],
                [robot.position[2]],
                'o', markersize=10, color=colors[i],
                label=f'Robot {i}'
            )
            self.robot_plots.append(plot)

            # Orbital trail
            trail, = self.ax.plot([], [], [], '-', alpha=0.4,
                                  color=colors[i], linewidth=1)
            self.robot_trails.append(trail)

        # Initialize client spacecraft markers
        for client_id, client in self.sim.clients.items():
            plot, = self.ax.plot(
                [client.position[0]],
                [client.position[1]],
                [client.position[2]],
                'D', markersize=12, color='cyan',
                markeredgecolor='white', markeredgewidth=1
            )
            self.client_plots.append(plot)

        # Docking lines (robot to client when docking)
        for _ in self.sim.robots:
            line, = self.ax.plot([], [], [], '-', color='white',
                                linewidth=2, alpha=0.8)
            self.docking_lines.append(line)

        # Configure axes
        limit = ORBIT_RADIUS * 1.3
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)

        # Style axes
        self.ax.set_xlabel('X (km)', color='white')
        self.ax.set_ylabel('Y (km)', color='white')
        self.ax.set_zlabel('Z (km)', color='white')
        self.ax.tick_params(colors='white')

        # Title
        self.ax.set_title('Lunar Orbital Refueling Depot - CRDT Coordination Demo',
                         color='white', fontsize=14, pad=20)

        # Status text (top left)
        self.status_text = self.ax.text2D(
            0.02, 0.98, '', transform=self.ax.transAxes,
            fontsize=11, color='white', family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8)
        )

        # Info text (bottom left)
        self.info_text = self.ax.text2D(
            0.02, 0.02, '', transform=self.ax.transAxes,
            fontsize=9, color='lightgray', family='monospace',
            verticalalignment='bottom'
        )

        # Add legend
        self.ax.legend(loc='upper right', fontsize=8,
                      facecolor='black', edgecolor='white',
                      labelcolor='white')

        plt.tight_layout()

    def update(self, frame: int) -> List:
        """
        Animation update function.

        Called by FuncAnimation for each frame.
        """
        if not self.running:
            return []

        # Step simulation
        state = self.sim.step(SIMULATION_DT)
        self.frame_count += 1

        # Check completion
        if self.sim.is_complete():
            logger.info(f"\nAll tasks completed at t={state.time:.0f}s!")
            self.running = False

        if state.time >= self.sim.max_time:
            logger.info(f"\nSimulation timeout at t={state.time:.0f}s")
            self.running = False

        # Update robot positions and colors
        for i, robot in enumerate(self.sim.robots):
            pos = robot.position

            # Update position
            self.robot_plots[i].set_data_3d([pos[0]], [pos[1]], [pos[2]])

            # Color based on state
            if state.partition_active:
                color = 'red'
            elif robot.fuel_mass < INITIAL_FUEL * 0.25:
                color = 'orange'  # Low fuel warning
            else:
                color = 'lime'

            self.robot_plots[i].set_color(color)

            # Update trail
            if len(robot.trajectory) > 1:
                trail = np.array(robot.trajectory)
                self.robot_trails[i].set_data_3d(
                    trail[:, 0], trail[:, 1], trail[:, 2]
                )

            # Update docking line
            if robot.is_docking and robot.current_task:
                client = self.sim.clients.get(robot.current_task)
                if client:
                    self.docking_lines[i].set_data_3d(
                        [pos[0], client.position[0]],
                        [pos[1], client.position[1]],
                        [pos[2], client.position[2]]
                    )
                else:
                    self.docking_lines[i].set_data_3d([], [], [])
            else:
                self.docking_lines[i].set_data_3d([], [], [])

        # Update client spacecraft
        for i, (client_id, client) in enumerate(self.sim.clients.items()):
            pos = client.position
            self.client_plots[i].set_data_3d([pos[0]], [pos[1]], [pos[2]])

            # Color based on state
            if client.is_complete:
                color = 'gray'
            elif client.servicing_robot:
                color = 'yellow'
            else:
                color = 'cyan'

            self.client_plots[i].set_color(color)

        # Update status text
        partition_status = "PARTITION" if state.partition_active else "CONNECTED"
        status_color = "red" if state.partition_active else "green"

        avg_fuel = np.mean([r.fuel_mass for r in self.sim.robots])

        self.status_text.set_text(
            f"Time: {state.time:.0f}s ({state.time/60:.1f} min)\n"
            f"Tasks: {state.completed_tasks}/{state.total_tasks} completed\n"
            f"Comms: {partition_status}\n"
            f"Avg Fuel: {avg_fuel:.1f} kg"
        )

        # Update info text
        docking_robots = [r.robot_id for r in self.sim.robots if r.is_docking]
        self.info_text.set_text(
            f"Frame: {self.frame_count} | "
            f"Orbital period: {ORBITAL_PERIOD/60:.0f} min | "
            f"Docking: {len(docking_robots)}"
        )

        # Slowly rotate view
        self.ax.view_init(elev=20, azim=self.frame_count * 0.2)

        return (self.robot_plots + self.robot_trails +
                self.client_plots + self.docking_lines)

    def run(self) -> None:
        """Run the animation."""
        self.setup_scene()

        # Calculate number of frames for max simulation time
        total_frames = int(self.sim.max_time / SIMULATION_DT)

        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=total_frames,
            interval=ANIMATION_INTERVAL,
            blit=False,
            repeat=False
        )

        plt.show()

        # Print final metrics
        self._print_final_metrics()

    def _print_final_metrics(self) -> None:
        """Print final simulation metrics."""
        metrics = self.sim.get_metrics()

        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE - Final Metrics")
        print("=" * 60)
        print(f"  Time elapsed:      {metrics['time_elapsed']:.0f}s "
              f"({metrics['time_elapsed']/60:.1f} min)")
        print(f"  Tasks completed:   {metrics['tasks_completed']}/{metrics['total_tasks']}")
        print(f"  Sync attempts:     {metrics['sync_attempts']}")
        print(f"  Sync successes:    {metrics['sync_successes']} "
              f"({100*metrics['sync_successes']/max(1,metrics['sync_attempts']):.1f}%)")
        print(f"  Partition time:    {metrics['partition_time']:.0f}s "
              f"({100*metrics['partition_time']/max(1,metrics['time_elapsed']):.1f}%)")
        print(f"  Total delta-v:     {metrics['total_delta_v']*1000:.1f} m/s")
        print(f"  Avg fuel remain:   {metrics['avg_fuel_remaining']:.1f} kg")
        print("=" * 60)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the orbital simulation."""
    print("=" * 60)
    print("  LUNAR ORBITAL REFUELING DEPOT SIMULATION")
    print("  CRDT-Based Autonomous Coordination Demo")
    print("=" * 60)
    print()
    print("Mission Parameters:")
    print(f"  - Servicing Robots: 5")
    print(f"  - Client Spacecraft: 5")
    print(f"  - Orbital Altitude: {ORBIT_ALTITUDE:.0f} km")
    print(f"  - Orbital Period: {ORBITAL_PERIOD/60:.1f} minutes")
    print(f"  - Comm Reliability: {RELIABILITY*100:.0f}%")
    print(f"  - Partition Probability: {PARTITION_PROBABILITY*100:.0f}% per sync")
    print()
    print("CRDT Coordination Features:")
    print("  - Task claiming (First-Write-Wins)")
    print("  - Position tracking (Last-Write-Wins)")
    print("  - Progress tracking (G-Counter)")
    print("  - Completed tasks (G-Set)")
    print()
    print("Controls:")
    print("  - Rotate: Click and drag")
    print("  - Zoom: Scroll wheel")
    print("  - Close window to exit")
    print()
    print("=" * 60)
    print()

    # Create and run simulation
    sim = OrbitalSimulation(num_robots=5, num_clients=5, seed=42)
    animator = OrbitalAnimator(sim)
    animator.run()


if __name__ == "__main__":
    main()
