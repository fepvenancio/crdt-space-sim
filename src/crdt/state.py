"""
CRDT State Implementation for Robot Coordination.

This module contains the core CRDT data structures that enable
conflict-free distributed coordination between robots.

CRDT Types Used:
- G-Set (Grow-only Set): For completed tasks
- G-Counter: For task progress tracking
- LWW-Register (Last-Write-Wins): For robot positions
- FWW-Map (First-Write-Wins): For task claims

All merge operations satisfy:
- Commutativity: merge(A, B) == merge(B, A)
- Associativity: merge(merge(A, B), C) == merge(A, merge(B, C))
- Idempotency: merge(A, A) == A
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Optional
from copy import deepcopy


@dataclass
class Vector3:
    """3D position vector for robot coordinates."""
    
    x: float
    y: float
    z: float
    
    def distance_to(self, other: Vector3) -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def move_toward(self, target: Vector3, speed: float) -> Vector3:
        """
        Move toward target position at given speed.
        
        Args:
            target: Target position to move toward
            speed: Maximum distance to move per step
            
        Returns:
            New position after movement
        """
        dist = self.distance_to(target)
        if dist <= speed:
            return Vector3(target.x, target.y, target.z)
        
        ratio = speed / dist
        return Vector3(
            self.x + (target.x - self.x) * ratio,
            self.y + (target.y - self.y) * ratio,
            self.z + (target.z - self.z) * ratio
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector3):
            return False
        return (
            abs(self.x - other.x) < 1e-9 and
            abs(self.y - other.y) < 1e-9 and
            abs(self.z - other.z) < 1e-9
        )
    
    def __hash__(self) -> int:
        return hash((round(self.x, 6), round(self.y, 6), round(self.z, 6)))
    
    def __repr__(self) -> str:
        return f"Vector3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


@dataclass
class CRDTState:
    """
    Conflict-free Replicated Data Type state for robot coordination.
    
    This class implements a CRDT that allows multiple robots to maintain
    and merge state without conflicts. All operations are designed to
    be commutative, associative, and idempotent.
    
    Attributes:
        robot_id: Unique identifier for this robot
        version: Monotonically increasing version counter
        completed_tasks: G-Set of completed task IDs
        task_progress: G-Counter tracking progress per robot per task
        robot_positions: LWW-Register mapping robot IDs to positions
        claimed_tasks: FWW-Map tracking task claims
    
    Example:
        >>> state1 = CRDTState("robot_1")
        >>> state2 = CRDTState("robot_2")
        >>> state1.mark_task_complete("task_1", timestamp=100)
        >>> state2.add_progress("task_2", amount=5)
        >>> state1.merge(state2)
        >>> "task_1" in state1.completed_tasks
        True
        >>> state1.get_task_progress("task_2")
        5
    """
    
    robot_id: str
    version: int = 0
    completed_tasks: Set[str] = field(default_factory=set)
    task_progress: Dict[str, Dict[str, int]] = field(default_factory=dict)
    robot_positions: Dict[str, Tuple[Vector3, int]] = field(default_factory=dict)
    claimed_tasks: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    
    def mark_task_complete(self, task_id: str, timestamp: int) -> None:
        """
        Mark a task as completed (G-Set add operation).
        
        Once a task is marked complete, it can never be unmarked.
        This is the G-Set property: grow-only.
        
        Args:
            task_id: The task to mark as complete
            timestamp: When the task was completed
        """
        self.completed_tasks.add(task_id)
        self.version += 1
    
    def add_progress(self, task_id: str, amount: int) -> None:
        """
        Add progress to a task (G-Counter increment).
        
        Progress can only be added, never removed. Each robot tracks
        its own contributions separately.
        
        Args:
            task_id: The task to add progress to
            amount: Amount of progress to add (must be positive)
            
        Raises:
            ValueError: If amount is negative
        """
        if amount < 0:
            raise ValueError("Progress amount must be non-negative")
        
        if task_id not in self.task_progress:
            self.task_progress[task_id] = {}
        
        current = self.task_progress[task_id].get(self.robot_id, 0)
        self.task_progress[task_id][self.robot_id] = current + amount
        self.version += 1
    
    def get_task_progress(self, task_id: str) -> int:
        """
        Get total progress for a task across all robots.
        
        Args:
            task_id: The task to check
            
        Returns:
            Sum of all robot contributions to this task
        """
        if task_id not in self.task_progress:
            return 0
        return sum(self.task_progress[task_id].values())
    
    def update_position(self, robot_id: str, position: Vector3, timestamp: int) -> None:
        """
        Update a robot's position (LWW-Register write).
        
        Only updates if the timestamp is newer than the existing value.
        This is the Last-Write-Wins semantics.
        
        Args:
            robot_id: The robot whose position to update
            position: The new position
            timestamp: When this position was observed
        """
        current = self.robot_positions.get(robot_id)
        if current is None or timestamp > current[1]:
            self.robot_positions[robot_id] = (position, timestamp)
            self.version += 1
    
    def claim_task(self, task_id: str, robot_id: str, timestamp: int) -> bool:
        """
        Attempt to claim a task (FWW-Map operation).
        
        The first robot to claim a task wins. Later claims are ignored.
        This prevents duplicate work.
        
        Args:
            task_id: The task to claim
            robot_id: The robot claiming the task
            timestamp: When the claim was made
            
        Returns:
            True if claim was successful, False if already claimed
        """
        if task_id not in self.claimed_tasks:
            self.claimed_tasks[task_id] = (robot_id, timestamp)
            self.version += 1
            return True
        return False
    
    def is_task_claimed_by_other(self, task_id: str, my_robot_id: str) -> bool:
        """
        Check if a task is claimed by another robot.
        
        Args:
            task_id: The task to check
            my_robot_id: The robot asking
            
        Returns:
            True if task is claimed by a different robot
        """
        if task_id not in self.claimed_tasks:
            return False
        claimer_id, _ = self.claimed_tasks[task_id]
        return claimer_id != my_robot_id
    
    def get_task_claimer(self, task_id: str) -> Optional[str]:
        """
        Get the robot that claimed a task.
        
        Args:
            task_id: The task to check
            
        Returns:
            Robot ID of claimer, or None if unclaimed
        """
        if task_id not in self.claimed_tasks:
            return None
        return self.claimed_tasks[task_id][0]
    
    def merge(self, other: CRDTState) -> None:
        """
        Merge another robot's state into this one.
        
        This is the core CRDT operation. It satisfies:
        - Commutativity: merge(A, B) == merge(B, A)
        - Associativity: merge(merge(A, B), C) == merge(A, merge(B, C))
        - Idempotency: merge(A, A) == A
        
        After merging, this state contains all information from both
        the original state and the other state.
        
        Args:
            other: State from another robot to merge
        """
        # G-Set merge: union of completed tasks
        self.completed_tasks |= other.completed_tasks
        
        # G-Counter merge: max of each robot's contribution per task
        for task_id, contributions in other.task_progress.items():
            if task_id not in self.task_progress:
                self.task_progress[task_id] = {}
            for robot_id, progress in contributions.items():
                current = self.task_progress[task_id].get(robot_id, 0)
                self.task_progress[task_id][robot_id] = max(current, progress)
        
        # LWW-Register merge: keep newer timestamp
        for robot_id, (position, timestamp) in other.robot_positions.items():
            current = self.robot_positions.get(robot_id)
            if current is None or timestamp > current[1]:
                self.robot_positions[robot_id] = (position, timestamp)
        
        # FWW-Map merge: keep earlier timestamp (first claim wins)
        for task_id, (robot_id, timestamp) in other.claimed_tasks.items():
            if task_id not in self.claimed_tasks:
                self.claimed_tasks[task_id] = (robot_id, timestamp)
            else:
                current_timestamp = self.claimed_tasks[task_id][1]
                if timestamp < current_timestamp:
                    self.claimed_tasks[task_id] = (robot_id, timestamp)
        
        self.version += 1
    
    def copy(self) -> CRDTState:
        """Create a deep copy of this state."""
        return deepcopy(self)
    
    def __eq__(self, other: object) -> bool:
        """Check equality of two CRDT states."""
        if not isinstance(other, CRDTState):
            return False
        return (
            self.completed_tasks == other.completed_tasks and
            self.task_progress == other.task_progress and
            self.robot_positions == other.robot_positions and
            self.claimed_tasks == other.claimed_tasks
        )


def verify_crdt_properties(state_a: CRDTState, state_b: CRDTState, state_c: CRDTState) -> dict:
    """
    Verify that CRDT properties hold for given states.
    
    This is useful for testing and validation.
    
    Args:
        state_a, state_b, state_c: Three states to test
        
    Returns:
        Dictionary with test results for each property
    """
    results = {}
    
    # Test commutativity: merge(A, B) == merge(B, A)
    ab = state_a.copy()
    ab.merge(state_b)
    
    ba = state_b.copy()
    ba.merge(state_a)
    
    results["commutative"] = (ab == ba)
    
    # Test associativity: merge(merge(A, B), C) == merge(A, merge(B, C))
    ab_c = state_a.copy()
    ab_c.merge(state_b)
    ab_c.merge(state_c)
    
    bc = state_b.copy()
    bc.merge(state_c)
    a_bc = state_a.copy()
    a_bc.merge(bc)
    
    results["associative"] = (ab_c == a_bc)
    
    # Test idempotency: merge(A, A) == A
    a_copy = state_a.copy()
    original = state_a.copy()
    a_copy.merge(original)
    
    results["idempotent"] = (a_copy == original)
    
    return results
