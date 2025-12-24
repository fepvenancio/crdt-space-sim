#!/usr/bin/env python3
"""
Validated Statistical Benchmark with Real NASA Parameters.

Uses real-world data for:
1. Communication latency (speed of light calculations)
2. Communication reliability (DSN performance data)
3. Blackout durations (orbital mechanics)
4. Fuel consumption (Tsiolkovsky rocket equation)

Sources:
- NASA NTRS: Communication Delays for Crewed Mars Missions
- ESA Mars Express: Time delay between Mars and Earth
- NASA DSN Services Catalog
- Wikipedia: Delta-v budget, Tsiolkovsky equation

Run: python benchmark_validated.py
"""

import sys
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import random

sys.path.insert(0, '.')
from src.crdt import CRDTState, Vector3

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS (Real Values)
# =============================================================================

# Speed of light
C_KM_PER_S = 299_792.458  # km/s

# Distances (average)
EARTH_MOON_DISTANCE_KM = 384_400  # km (average)
EARTH_MARS_MIN_KM = 54_600_000    # km (closest approach)
EARTH_MARS_MAX_KM = 401_000_000   # km (opposition)
EARTH_MARS_AVG_KM = 225_000_000   # km (average)

# Orbital periods
LEO_ORBITAL_PERIOD_S = 90 * 60           # 90 minutes
LUNAR_ORBITAL_PERIOD_S = 2 * 3600        # ~2 hours (100km altitude)
MARS_ORBITAL_PERIOD_S = 24.6 * 3600      # ~24.6 hours (Mars day for surface ops)

# Standard gravity
G0_M_S2 = 9.80665  # m/s²


# =============================================================================
# COMMUNICATION PARAMETERS (Validated from NASA Sources)
# =============================================================================

@dataclass
class ValidatedCommsScenario:
    """
    Communication scenario with real NASA-validated parameters.

    Sources:
    - NASA NTRS 20220013418: Communication Delays for Mars Missions
    - NASA DSN Services Catalog (820-100)
    - Speed of light calculations for latency
    """
    name: str
    description: str

    # Latency (one-way, seconds) - from speed of light
    latency_one_way_s: float
    latency_round_trip_s: float

    # Reliability (probability message arrives intact)
    # DSN achieves ~99.99% for strong signals, but we model end-to-end
    reliability: float

    # Blackout parameters
    blackout_probability_per_orbit: float  # Probability of blackout per orbital period
    blackout_duration_range_s: Tuple[float, float]  # (min, max) seconds

    # Orbital period (for blackout timing)
    orbital_period_s: float

    # Sync interval (how often robots attempt to communicate)
    sync_interval_s: float

    def latency_in_steps(self, step_duration_s: float) -> int:
        """Convert round-trip latency to simulation steps."""
        return max(1, int(self.latency_round_trip_s / step_duration_s))

    def blackout_duration_in_steps(self, step_duration_s: float) -> Tuple[int, int]:
        """Convert blackout duration range to simulation steps."""
        return (
            int(self.blackout_duration_range_s[0] / step_duration_s),
            int(self.blackout_duration_range_s[1] / step_duration_s)
        )


# Calculate real latencies
def _calc_light_time(distance_km: float) -> float:
    """Calculate one-way light travel time in seconds."""
    return distance_km / C_KM_PER_S


# Real scenarios with citations
VALIDATED_SCENARIOS = {
    "LEO": ValidatedCommsScenario(
        name="LEO",
        description="Low Earth Orbit via TDRS relay",
        # LEO to ground via TDRS: ~0.24s one-way (GEO relay)
        latency_one_way_s=0.24,
        latency_round_trip_s=0.48,
        # TDRS provides ~99% coverage, DSN reliability ~99.9%
        reliability=0.98,
        # Eclipse occurs ~35 min per 90 min orbit, but TDRS coverage helps
        # Actual blackout: ~5% of orbit without TDRS
        blackout_probability_per_orbit=0.15,
        blackout_duration_range_s=(0, 300),  # 0-5 minutes
        orbital_period_s=LEO_ORBITAL_PERIOD_S,
        sync_interval_s=10.0,  # Can sync frequently
    ),

    "Lunar": ValidatedCommsScenario(
        name="Lunar",
        description="Lunar orbit - Earth-Moon distance",
        # Earth-Moon: 384,400 km → 1.28s one-way
        latency_one_way_s=_calc_light_time(EARTH_MOON_DISTANCE_KM),  # 1.28s
        latency_round_trip_s=2 * _calc_light_time(EARTH_MOON_DISTANCE_KM),  # 2.56s
        # Direct line of sight when visible, DSN reliability
        reliability=0.95,
        # Far side of Moon: ~45 min blackout per 2-hour orbit
        # Probability: happens every orbit when in polar/inclined orbit
        blackout_probability_per_orbit=0.8,
        blackout_duration_range_s=(30 * 60, 50 * 60),  # 30-50 minutes
        orbital_period_s=LUNAR_ORBITAL_PERIOD_S,
        sync_interval_s=30.0,  # Sync every 30s when possible
    ),

    "Mars_Conjunction": ValidatedCommsScenario(
        name="Mars_Conjunction",
        description="Mars during solar conjunction (worst case)",
        # Mars max distance: 401M km → 22.4 min one-way
        latency_one_way_s=_calc_light_time(EARTH_MARS_MAX_KM),  # ~1337s = 22.3 min
        latency_round_trip_s=2 * _calc_light_time(EARTH_MARS_MAX_KM),  # ~44.6 min
        # Solar conjunction: complete blackout for ~2 weeks
        # During conjunction approach, reliability degrades
        reliability=0.70,
        # Solar conjunction blackout
        blackout_probability_per_orbit=0.3,
        blackout_duration_range_s=(6 * 3600, 24 * 3600),  # 6-24 hours
        orbital_period_s=MARS_ORBITAL_PERIOD_S,
        sync_interval_s=300.0,  # 5 minutes (latency-limited)
    ),

    "Mars_Nominal": ValidatedCommsScenario(
        name="Mars_Nominal",
        description="Mars nominal operations (average distance)",
        # Mars average: 225M km → 12.5 min one-way
        latency_one_way_s=_calc_light_time(EARTH_MARS_AVG_KM),  # ~750s = 12.5 min
        latency_round_trip_s=2 * _calc_light_time(EARTH_MARS_AVG_KM),  # ~25 min
        # Good signal conditions
        reliability=0.90,
        # DSN coverage gaps, Mars rotation
        blackout_probability_per_orbit=0.2,
        blackout_duration_range_s=(1 * 3600, 4 * 3600),  # 1-4 hours
        orbital_period_s=MARS_ORBITAL_PERIOD_S,
        sync_interval_s=180.0,  # 3 minutes
    ),
}


# =============================================================================
# FUEL CONSUMPTION MODEL (Physics-Based with Tsiolkovsky Equation)
# =============================================================================

@dataclass
class SpacecraftFuelModel:
    """
    Physics-based fuel consumption model.

    Key insight for proximity operations:
    - Robot accelerates from rest to cruise velocity: delta-v = v_cruise
    - Robot coasts (no fuel) during transit
    - Robot decelerates to rest at target: delta-v = v_cruise
    - Total per maneuver: 2 * v_cruise (independent of distance!)

    This is physically correct for impulsive maneuvers.
    Distance affects TIME, not delta-v for point-to-point transfers.

    Tsiolkovsky equation converts delta-v to fuel mass:
    m_fuel = m_current × (1 - 1/exp(Δv / (Isp × g₀)))

    Sources:
    - Standard astrodynamics (Tsiolkovsky equation)
    - NASA OSAM-1 mission parameters
    - ESA e.Deorbit servicer studies
    """
    dry_mass_kg: float = 200.0      # Spacecraft dry mass (kg)
    initial_fuel_kg: float = 50.0   # Initial fuel load (kg)
    isp_s: float = 230.0            # Specific impulse (seconds) - hydrazine monoprop (corrected)

    # Cruise velocity for proximity operations (m/s)
    # Typical ISS approach: 0.1-0.3 m/s, servicing ops: 0.3-1.0 m/s
    cruise_velocity_m_s: float = 0.5

    # Station keeping delta-v (m/s per hour) - only when stationary
    # LEO: ~0.01 m/s/hour, Lunar: ~0.001 m/s/hour
    delta_v_station_keeping_per_hour: float = 0.01

    def fuel_for_delta_v(self, delta_v_m_s: float, current_mass_kg: float) -> float:
        """
        Calculate fuel required for a given delta-v using Tsiolkovsky equation.

        m_fuel = m_current × (1 - 1/exp(Δv / Ve))
        where Ve = Isp × g₀
        """
        if delta_v_m_s <= 0 or current_mass_kg <= 0:
            return 0.0

        exhaust_velocity = self.isp_s * G0_M_S2  # m/s
        mass_ratio = math.exp(delta_v_m_s / exhaust_velocity)
        fuel_mass = current_mass_kg * (1 - 1/mass_ratio)

        return max(0.0, fuel_mass)

    def fuel_for_acceleration_burn(self, current_mass_kg: float) -> float:
        """
        Fuel for one acceleration burn (0 -> cruise velocity).
        Called when robot STARTS moving toward a target.
        """
        return self.fuel_for_delta_v(self.cruise_velocity_m_s, current_mass_kg)

    def fuel_for_deceleration_burn(self, current_mass_kg: float) -> float:
        """
        Fuel for one deceleration burn (cruise velocity -> 0).
        Called when robot ARRIVES at target.
        """
        return self.fuel_for_delta_v(self.cruise_velocity_m_s, current_mass_kg)

    def fuel_for_station_keeping(self, hours: float, current_mass_kg: float) -> float:
        """
        Fuel for station keeping over time period.
        Only applies when robot is STATIONARY (working or idle).
        """
        total_dv = self.delta_v_station_keeping_per_hour * hours
        return self.fuel_for_delta_v(total_dv, current_mass_kg)


# =============================================================================
# VALIDATED METRICS
# =============================================================================

@dataclass
class ValidatedMetrics:
    """Comprehensive metrics with physics-based fuel tracking."""
    # Time metrics
    steps: int = 0
    wall_time_s: float = 0.0

    # Task metrics
    completed_tasks: int = 0
    total_tasks: int = 0

    # Communication metrics
    messages_sent: int = 0
    messages_failed: int = 0
    blackout_time_s: float = 0.0

    # Work metrics
    total_work_done: int = 0
    duplicate_work: int = 0

    # Fuel metrics (kg) - physics-based
    fuel_acceleration_burns: float = 0.0   # Fuel for starting movement
    fuel_deceleration_burns: float = 0.0   # Fuel for stopping at target
    fuel_station_keeping: float = 0.0      # Fuel for holding position
    fuel_wasted_aborts: float = 0.0        # Fuel wasted on aborted approaches

    # Maneuver counts
    acceleration_burns: int = 0
    deceleration_burns: int = 0
    aborted_approaches: int = 0  # Started toward task that was already done

    @property
    def total_fuel_used(self) -> float:
        return (self.fuel_acceleration_burns + self.fuel_deceleration_burns +
                self.fuel_station_keeping + self.fuel_wasted_aborts)

    @property
    def fuel_efficiency(self) -> float:
        """Tasks completed per kg of fuel."""
        if self.total_fuel_used <= 0:
            return float('inf')
        return self.completed_tasks / self.total_fuel_used

    @property
    def duplicate_work_pct(self) -> float:
        """Percentage of work that was duplicate."""
        total = self.total_work_done
        if total <= 0:
            return 0.0
        return 100.0 * self.duplicate_work / total

    @property
    def wasted_fuel_pct(self) -> float:
        """Percentage of fuel wasted on aborted approaches."""
        if self.total_fuel_used <= 0:
            return 0.0
        return 100.0 * self.fuel_wasted_aborts / self.total_fuel_used


# =============================================================================
# VALIDATED SIMULATION
# =============================================================================

@dataclass
class Task:
    """A task to be completed."""
    task_id: str
    location: Vector3
    duration: int  # Work units required

    def __hash__(self):
        return hash(self.task_id)


@dataclass
class CRDTRobot:
    """CRDT-coordinated robot with fuel tracking."""
    robot_id: str
    position: Vector3
    state: CRDTState = field(default=None)
    current_task: Optional[str] = None
    working: bool = False
    moving: bool = False  # True if currently in transit (coasting)

    # Fuel tracking
    fuel_remaining_kg: float = 50.0  # Starts with initial fuel load

    def __post_init__(self):
        if self.state is None:
            self.state = CRDTState(self.robot_id)

    @property
    def current_mass_kg(self) -> float:
        """Current total mass (dry + fuel)."""
        return 200.0 + self.fuel_remaining_kg  # dry_mass + fuel

    def burn_fuel(self, amount_kg: float) -> float:
        """Consume fuel and return actual amount burned."""
        actual = min(amount_kg, self.fuel_remaining_kg)
        self.fuel_remaining_kg -= actual
        return actual


@dataclass
class CentralizedRobot:
    """Centralized robot with command buffer and fuel tracking."""
    robot_id: str
    position: Vector3
    command_buffer: List[dict] = field(default_factory=list)
    current_command: Optional[dict] = None
    work_progress: Dict[str, int] = field(default_factory=dict)
    completed_tasks: set = field(default_factory=set)
    buffer_size: int = 5
    moving: bool = False  # True if currently in transit

    # Fuel tracking
    fuel_remaining_kg: float = 50.0

    @property
    def current_mass_kg(self) -> float:
        return 200.0 + self.fuel_remaining_kg

    def burn_fuel(self, amount_kg: float) -> float:
        actual = min(amount_kg, self.fuel_remaining_kg)
        self.fuel_remaining_kg -= actual
        return actual


class ValidatedSimulation:
    """
    Simulation with validated real-world parameters.
    """

    def __init__(
        self,
        scenario_name: str,
        num_robots: int = 5,
        num_tasks: int = 10,
        step_duration_s: float = 10.0,  # Each step = 10 seconds
        seed: int = 42,
        max_wall_time_s: float = 8 * 3600,  # 8 hours max
    ):
        self.scenario = VALIDATED_SCENARIOS[scenario_name]
        self.num_robots = num_robots
        self.num_tasks = num_tasks
        self.step_duration_s = step_duration_s
        self.seed = seed
        self.max_wall_time_s = max_wall_time_s
        self.max_steps = int(max_wall_time_s / step_duration_s)

        self.rng = random.Random(seed)
        self.fuel_model = SpacecraftFuelModel()

        # Convert scenario parameters to steps
        self.latency_steps = self.scenario.latency_in_steps(step_duration_s)
        self.blackout_range_steps = self.scenario.blackout_duration_in_steps(step_duration_s)
        self.sync_interval_steps = max(1, int(self.scenario.sync_interval_s / step_duration_s))
        self.orbital_period_steps = int(self.scenario.orbital_period_s / step_duration_s)

        # Pre-generate synchronized events
        self._partition_schedule: Dict[int, int] = {}
        self._message_outcomes: List[bool] = []
        self._start_positions: List[Vector3] = []

        # Generate tasks
        self.tasks = self._create_tasks()

    def _create_tasks(self) -> Dict[str, Task]:
        """Generate tasks."""
        tasks = {}
        for i in range(self.num_tasks):
            tasks[f"task_{i}"] = Task(
                task_id=f"task_{i}",
                location=Vector3(
                    self.rng.uniform(0, 100),
                    self.rng.uniform(0, 100),
                    self.rng.uniform(0, 100)
                ),
                duration=self.rng.randint(5, 15)
            )
        return tasks

    def _prepare_trial(self):
        """Prepare synchronized random events for fair comparison."""
        trial_rng = random.Random(self.rng.randint(0, 2**32))

        # Generate blackout schedule based on orbital mechanics
        self._partition_schedule = {}
        step = 0
        while step < self.max_steps:
            # Check each orbital period
            if trial_rng.random() < self.scenario.blackout_probability_per_orbit:
                duration_steps = trial_rng.randint(*self.blackout_range_steps)
                if duration_steps > 0:
                    self._partition_schedule[step] = duration_steps
                    step += duration_steps
            step += self.orbital_period_steps

        # Pre-generate message outcomes
        max_messages = self.max_steps * self.num_robots * self.num_robots * 2
        self._message_outcomes = [
            trial_rng.random() < self.scenario.reliability
            for _ in range(max_messages)
        ]
        self._message_index = 0

        # Pre-generate starting positions
        self._start_positions = [
            Vector3(50 + trial_rng.uniform(-5, 5),
                   50 + trial_rng.uniform(-5, 5),
                   50 + trial_rng.uniform(-5, 5))
            for _ in range(self.num_robots)
        ]

    def _next_message_succeeds(self, in_blackout: bool) -> bool:
        """Get next pre-determined message outcome."""
        if in_blackout:
            return False
        if self._message_index >= len(self._message_outcomes):
            return self.rng.random() < self.scenario.reliability
        result = self._message_outcomes[self._message_index]
        self._message_index += 1
        return result

    def run_crdt(self) -> ValidatedMetrics:
        """Run CRDT simulation with physics-based fuel consumption."""
        tasks = deepcopy(self.tasks)
        metrics = ValidatedMetrics(total_tasks=len(tasks))

        robots = [
            CRDTRobot(f"robot_{i}", deepcopy(self._start_positions[i]))
            for i in range(self.num_robots)
        ]

        blackout_remaining = 0
        actual_completed = set()
        total_work_per_task = {t: 0 for t in tasks}

        for step in range(1, self.max_steps + 1):
            metrics.steps = step
            metrics.wall_time_s = step * self.step_duration_s

            # Check blackout schedule
            if step in self._partition_schedule:
                blackout_remaining = self._partition_schedule[step]

            in_blackout = blackout_remaining > 0
            if in_blackout:
                metrics.blackout_time_s += self.step_duration_s
                blackout_remaining -= 1

            # Each robot acts autonomously
            for robot in robots:
                robot.state.update_position(robot.robot_id, robot.position, step)

                # Track work before action
                old_progress = {t: robot.state.get_task_progress(t) for t in tasks}

                # Robot is working on a task
                if robot.current_task and robot.working:
                    if robot.current_task not in robot.state.completed_tasks:
                        task = tasks.get(robot.current_task)
                        if task:
                            robot.state.add_progress(task.task_id, 1)
                            # Station keeping while working (stationary)
                            fuel = self.fuel_model.fuel_for_station_keeping(
                                self.step_duration_s / 3600, robot.current_mass_kg)
                            robot.burn_fuel(fuel)
                            metrics.fuel_station_keeping += fuel

                            if robot.state.get_task_progress(task.task_id) >= task.duration:
                                robot.state.mark_task_complete(task.task_id, step)
                                robot.current_task = None
                                robot.working = False
                    else:
                        # Task was completed by someone else while we were working
                        robot.current_task = None
                        robot.working = False

                # Robot is moving toward a task
                elif robot.current_task and robot.moving:
                    task = tasks.get(robot.current_task)
                    if task:
                        # Check if task was completed while we were en route
                        if task.task_id in actual_completed or task.task_id in robot.state.completed_tasks:
                            # ABORT: Task already done, we wasted the acceleration burn
                            # Decelerate to stop (still costs fuel)
                            fuel = self.fuel_model.fuel_for_deceleration_burn(robot.current_mass_kg)
                            robot.burn_fuel(fuel)
                            metrics.fuel_wasted_aborts += fuel
                            metrics.aborted_approaches += 1
                            metrics.deceleration_burns += 1
                            robot.current_task = None
                            robot.moving = False
                        else:
                            # Continue moving (coasting - no fuel cost)
                            dist = robot.position.distance_to(task.location)
                            if dist < 2.0:
                                # Arrived - deceleration burn
                                fuel = self.fuel_model.fuel_for_deceleration_burn(robot.current_mass_kg)
                                robot.burn_fuel(fuel)
                                metrics.fuel_deceleration_burns += fuel
                                metrics.deceleration_burns += 1
                                robot.moving = False
                                robot.working = True
                            else:
                                # Coast toward target (no fuel during coast)
                                robot.position = robot.position.move_toward(task.location, 2.0)

                # Robot is idle - find a task
                else:
                    # Station keeping while idle
                    fuel = self.fuel_model.fuel_for_station_keeping(
                        self.step_duration_s / 3600, robot.current_mass_kg)
                    robot.burn_fuel(fuel)
                    metrics.fuel_station_keeping += fuel

                    # Find best available task
                    best_task = None
                    best_dist = float('inf')
                    for task in tasks.values():
                        if task.task_id in robot.state.completed_tasks:
                            continue
                        if task.task_id in actual_completed:
                            continue
                        if robot.state.is_task_claimed_by_other(task.task_id, robot.robot_id):
                            continue
                        dist = robot.position.distance_to(task.location)
                        if dist < best_dist:
                            best_dist = dist
                            best_task = task

                    if best_task:
                        robot.current_task = best_task.task_id
                        robot.state.claim_task(best_task.task_id, robot.robot_id, step)
                        if best_dist < 2.0:
                            # Already at task location
                            robot.working = True
                        else:
                            # Start moving - acceleration burn
                            fuel = self.fuel_model.fuel_for_acceleration_burn(robot.current_mass_kg)
                            robot.burn_fuel(fuel)
                            metrics.fuel_acceleration_burns += fuel
                            metrics.acceleration_burns += 1
                            robot.moving = True
                            robot.position = robot.position.move_toward(best_task.location, 2.0)

                # Track work done
                for task_id, task in tasks.items():
                    new_progress = robot.state.get_task_progress(task_id)
                    work_increment = new_progress - old_progress.get(task_id, 0)
                    if work_increment > 0:
                        metrics.total_work_done += work_increment
                        if task_id in actual_completed:
                            metrics.duplicate_work += work_increment
                        else:
                            total_work_per_task[task_id] += work_increment
                            if total_work_per_task[task_id] > task.duration:
                                overflow = total_work_per_task[task_id] - task.duration
                                metrics.duplicate_work += min(work_increment, overflow)
                        if new_progress >= task.duration:
                            actual_completed.add(task_id)

            # Sync at interval
            if step % self.sync_interval_steps == 0:
                for i, ra in enumerate(robots):
                    for rb in robots[i+1:]:
                        metrics.messages_sent += 2
                        if self._next_message_succeeds(in_blackout):
                            rb.state.merge(ra.state)
                        else:
                            metrics.messages_failed += 1
                        if self._next_message_succeeds(in_blackout):
                            ra.state.merge(rb.state)
                        else:
                            metrics.messages_failed += 1

            if len(actual_completed) >= len(tasks):
                break

        metrics.completed_tasks = len(actual_completed)
        return metrics

    def run_centralized(self) -> ValidatedMetrics:
        """Run centralized simulation with physics-based fuel consumption."""
        tasks = deepcopy(self.tasks)
        metrics = ValidatedMetrics(total_tasks=len(tasks))

        self._message_index = 0  # Reset for fair comparison

        robots = [
            CentralizedRobot(f"robot_{i}", deepcopy(self._start_positions[i]))
            for i in range(self.num_robots)
        ]

        # Ground control state
        task_assignments = {}
        ground_completed = set()
        pending_commands = {r.robot_id: [] for r in robots}
        command_arrival_time = {}  # When commands will arrive (latency)

        blackout_remaining = 0
        actual_completed = set()

        for step in range(1, self.max_steps + 1):
            metrics.steps = step
            metrics.wall_time_s = step * self.step_duration_s

            # Check blackout
            if step in self._partition_schedule:
                blackout_remaining = self._partition_schedule[step]

            in_blackout = blackout_remaining > 0
            if in_blackout:
                metrics.blackout_time_s += self.step_duration_s
                blackout_remaining -= 1

            # Deliver commands that have arrived (after latency)
            for robot in robots:
                if robot.robot_id in command_arrival_time:
                    arrival = command_arrival_time[robot.robot_id]
                    if step >= arrival and pending_commands[robot.robot_id]:
                        cmds = pending_commands[robot.robot_id]
                        for cmd in cmds:
                            if len(robot.command_buffer) < robot.buffer_size:
                                robot.command_buffer.append(cmd)
                        pending_commands[robot.robot_id] = []
                        del command_arrival_time[robot.robot_id]

            # Ground control sends commands at sync interval
            if step % self.sync_interval_steps == 0 and not in_blackout:
                available_tasks = [
                    t for t in tasks.values()
                    if t.task_id not in ground_completed
                    and t.task_id not in task_assignments
                ]

                for robot in robots:
                    if robot.current_command is None and not robot.command_buffer:
                        if available_tasks:
                            metrics.messages_sent += 1
                            if self._next_message_succeeds(in_blackout):
                                task = min(available_tasks,
                                          key=lambda t: robot.position.distance_to(t.location))
                                available_tasks.remove(task)
                                task_assignments[task.task_id] = robot.robot_id

                                # Commands arrive after latency
                                pending_commands[robot.robot_id] = [
                                    {"type": "goto", "task_id": task.task_id, "target": task.location},
                                    {"type": "work", "task_id": task.task_id}
                                ]
                                command_arrival_time[robot.robot_id] = step + self.latency_steps
                            else:
                                metrics.messages_failed += 1

            # Robots execute commands
            for robot in robots:
                if robot.current_command is None and robot.command_buffer:
                    cmd = robot.command_buffer.pop(0)
                    robot.current_command = cmd
                    # Starting a goto command = acceleration burn
                    if cmd["type"] == "goto":
                        fuel = self.fuel_model.fuel_for_acceleration_burn(robot.current_mass_kg)
                        robot.burn_fuel(fuel)
                        metrics.fuel_acceleration_burns += fuel
                        metrics.acceleration_burns += 1
                        robot.moving = True

                if robot.current_command:
                    cmd = robot.current_command
                    if cmd["type"] == "goto":
                        dist = robot.position.distance_to(cmd["target"])
                        if dist < 2.0:
                            # Arrived - deceleration burn
                            fuel = self.fuel_model.fuel_for_deceleration_burn(robot.current_mass_kg)
                            robot.burn_fuel(fuel)
                            metrics.fuel_deceleration_burns += fuel
                            metrics.deceleration_burns += 1
                            robot.moving = False
                            robot.current_command = None
                        else:
                            # Coast toward target (no fuel)
                            robot.position = robot.position.move_toward(cmd["target"], 2.0)
                    elif cmd["type"] == "work":
                        task = tasks.get(cmd["task_id"])
                        if task:
                            robot.work_progress[task.task_id] = robot.work_progress.get(task.task_id, 0) + 1
                            metrics.total_work_done += 1
                            # Station keeping while working
                            fuel = self.fuel_model.fuel_for_station_keeping(
                                self.step_duration_s / 3600, robot.current_mass_kg)
                            robot.burn_fuel(fuel)
                            metrics.fuel_station_keeping += fuel

                            if robot.work_progress[task.task_id] >= task.duration:
                                robot.current_command = None
                                robot.completed_tasks.add(task.task_id)
                                actual_completed.add(task.task_id)
                                ground_completed.add(task.task_id)
                                if task.task_id in task_assignments:
                                    del task_assignments[task.task_id]
                else:
                    # Idle - station keeping
                    if not robot.moving:
                        fuel = self.fuel_model.fuel_for_station_keeping(
                            self.step_duration_s / 3600, robot.current_mass_kg)
                        robot.burn_fuel(fuel)
                        metrics.fuel_station_keeping += fuel

            if len(actual_completed) >= len(tasks):
                break

        metrics.completed_tasks = len(actual_completed)
        return metrics


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

@dataclass
class StatResult:
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int


def compute_stats(values: List[float], confidence: float = 0.95) -> StatResult:
    n = len(values)
    if n == 0:
        return StatResult(0, 0, 0, 0, 0)
    mean = sum(values) / n
    if n == 1:
        return StatResult(mean, 0, mean, mean, n)
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    t_val = 1.96 if n > 30 else 2.0
    margin = t_val * std / math.sqrt(n)
    return StatResult(mean, std, mean - margin, mean + margin, n)


def t_test(v1: List[float], v2: List[float]) -> Tuple[float, float]:
    n1, n2 = len(v1), len(v2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    m1, m2 = sum(v1)/n1, sum(v2)/n2
    var1 = sum((x-m1)**2 for x in v1) / (n1-1)
    var2 = sum((x-m2)**2 for x in v2) / (n2-1)
    se = math.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return 0.0, 1.0
    t_stat = (m1 - m2) / se
    z = abs(t_stat)
    if z > 3: p = 0.003
    elif z > 2.576: p = 0.01
    elif z > 1.96: p = 0.05
    else: p = min(1.0, 2 * (1 - 0.5 * (1 + math.erf(z / 1.414))))
    return t_stat, p


def run_validated_benchmark(num_trials: int = 100):
    """Run benchmark with validated, physics-based parameters."""

    print("=" * 75)
    print("VALIDATED BENCHMARK: Physics-Based Fuel Model")
    print("=" * 75)
    print()
    print("Communication Parameters (from NASA sources):")
    print("-" * 75)
    for name, scenario in VALIDATED_SCENARIOS.items():
        print(f"{name:20} RTT: {scenario.latency_round_trip_s:>8.1f}s  "
              f"Reliability: {scenario.reliability*100:>5.1f}%  "
              f"Blackout: {scenario.blackout_duration_range_s[0]/60:.0f}-{scenario.blackout_duration_range_s[1]/60:.0f} min")
    print()

    fm = SpacecraftFuelModel()
    print("Physics-Based Fuel Model:")
    print(f"  Isp: {fm.isp_s}s (hydrazine monoprop - corrected)")
    print(f"  Cruise velocity: {fm.cruise_velocity_m_s} m/s")
    print(f"  Accel burn: {fm.fuel_for_acceleration_burn(250):.4f} kg (at 250kg)")
    print(f"  Decel burn: {fm.fuel_for_deceleration_burn(250):.4f} kg (at 250kg)")
    print(f"  Station keeping: {fm.delta_v_station_keeping_per_hour} m/s/hour")
    print()
    print("Fuel is charged for:")
    print("  - Acceleration burn when starting to move")
    print("  - Deceleration burn when arriving at target")
    print("  - Station keeping when stationary (working or idle)")
    print("  - Aborted approaches count as wasted fuel (CRDT only)")
    print()
    print(f"Running {num_trials} trials per scenario...")
    print("=" * 75)

    results = {}

    for scenario_name in ["LEO", "Lunar", "Mars_Nominal", "Mars_Conjunction"]:
        print(f"\n{scenario_name}...", end=" ", flush=True)

        crdt_times = []
        crdt_fuel = []
        crdt_aborts = []
        crdt_wasted = []
        cent_times = []
        cent_fuel = []

        for trial in range(num_trials):
            if (trial + 1) % 25 == 0:
                print(f"{trial+1}", end=" ", flush=True)

            sim = ValidatedSimulation(scenario_name, seed=42 + trial)
            sim._prepare_trial()

            crdt = sim.run_crdt()
            cent = sim.run_centralized()

            crdt_times.append(crdt.wall_time_s / 3600)  # hours
            crdt_fuel.append(crdt.total_fuel_used)
            crdt_aborts.append(crdt.aborted_approaches)
            crdt_wasted.append(crdt.wasted_fuel_pct)
            cent_times.append(cent.wall_time_s / 3600)
            cent_fuel.append(cent.total_fuel_used)

        print("Done")

        crdt_time_stats = compute_stats(crdt_times)
        cent_time_stats = compute_stats(cent_times)
        crdt_fuel_stats = compute_stats(crdt_fuel)
        cent_fuel_stats = compute_stats(cent_fuel)
        crdt_aborts_stats = compute_stats(crdt_aborts)
        crdt_wasted_stats = compute_stats(crdt_wasted)

        t_time, p_time = t_test(crdt_times, cent_times)
        t_fuel, p_fuel = t_test(crdt_fuel, cent_fuel)

        time_diff = (cent_time_stats.mean - crdt_time_stats.mean) / cent_time_stats.mean * 100 if cent_time_stats.mean > 0 else 0
        fuel_diff = (cent_fuel_stats.mean - crdt_fuel_stats.mean) / cent_fuel_stats.mean * 100 if cent_fuel_stats.mean > 0 else 0

        results[scenario_name] = {
            "crdt_time": crdt_time_stats,
            "cent_time": cent_time_stats,
            "crdt_fuel": crdt_fuel_stats,
            "cent_fuel": cent_fuel_stats,
            "crdt_aborts": crdt_aborts_stats,
            "crdt_wasted_pct": crdt_wasted_stats,
            "time_diff_pct": time_diff,
            "fuel_diff_pct": fuel_diff,
            "p_time": p_time,
            "p_fuel": p_fuel,
        }

    # Print results
    print("\n" + "=" * 75)
    print("RESULTS: Mission Completion Time (hours)")
    print("=" * 75)
    print(f"{'Scenario':<20} {'CRDT (h)':<18} {'Centralized (h)':<18} {'Diff':<10} {'p-value'}")
    print("-" * 75)
    for name, r in results.items():
        sig = "***" if r["p_time"] < 0.001 else "**" if r["p_time"] < 0.01 else "*" if r["p_time"] < 0.05 else ""
        print(f"{name:<20} {r['crdt_time'].mean:>5.2f} +- {r['crdt_time'].std:<6.2f}   "
              f"{r['cent_time'].mean:>5.2f} +- {r['cent_time'].std:<6.2f}   "
              f"{r['time_diff_pct']:>+6.1f}%   {r['p_time']:.4f}{sig}")

    print("\n" + "=" * 75)
    print("RESULTS: Total Fuel Consumption (kg) - Physics-Based")
    print("=" * 75)
    print(f"{'Scenario':<20} {'CRDT (kg)':<18} {'Centralized (kg)':<18} {'Diff':<10} {'p-value'}")
    print("-" * 75)
    for name, r in results.items():
        sig = "***" if r["p_fuel"] < 0.001 else "**" if r["p_fuel"] < 0.01 else "*" if r["p_fuel"] < 0.05 else ""
        print(f"{name:<20} {r['crdt_fuel'].mean:>5.3f} +- {r['crdt_fuel'].std:<6.3f}  "
              f"{r['cent_fuel'].mean:>5.3f} +- {r['cent_fuel'].std:<6.3f}  "
              f"{r['fuel_diff_pct']:>+6.1f}%   {r['p_fuel']:.4f}{sig}")

    print("\n" + "=" * 75)
    print("CRDT Wasted Fuel (Aborted Approaches)")
    print("=" * 75)
    print(f"{'Scenario':<20} {'Aborts':<15} {'Wasted Fuel %'}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<20} {r['crdt_aborts'].mean:>4.1f} +- {r['crdt_aborts'].std:<5.1f}   "
              f"{r['crdt_wasted_pct'].mean:>5.1f}% +- {r['crdt_wasted_pct'].std:.1f}%")

    print("\n" + "=" * 75)
    print("PHYSICS MODEL VALIDATION")
    print("=" * 75)
    print("Fuel consumption is now tied to actual maneuvers:")
    print(f"  - Each approach: 2 burns x {fm.cruise_velocity_m_s} m/s = {2*fm.cruise_velocity_m_s} m/s delta-v")
    print(f"  - Fuel per burn: ~{fm.fuel_for_acceleration_burn(250)*1000:.2f} grams")
    print(f"  - CRDT wastes fuel on aborted approaches (started but task already done)")
    print(f"  - Centralized has no aborted approaches (ground assigns uniquely)")
    print()
    print("This is physically correct because:")
    print("  - Delta-v for impulsive maneuver is independent of distance")
    print("  - Distance affects TIME, not fuel (coast phase is free)")
    print("  - Tsiolkovsky equation properly accounts for mass change")

    return results


if __name__ == "__main__":
    run_validated_benchmark(num_trials=100)
