"""
Tests for CRDT state implementation.

These tests verify the core CRDT properties:
- Commutativity
- Associativity  
- Idempotency
- Monotonicity
"""

import pytest
from src.crdt import CRDTState, Vector3, verify_crdt_properties


class TestVector3:
    """Tests for Vector3 class."""
    
    def test_distance_to_same_point(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.distance_to(v) == 0.0
    
    def test_distance_to_different_point(self):
        v1 = Vector3(0.0, 0.0, 0.0)
        v2 = Vector3(3.0, 4.0, 0.0)
        assert v1.distance_to(v2) == 5.0
    
    def test_move_toward_arrives_if_close(self):
        v = Vector3(0.0, 0.0, 0.0)
        target = Vector3(1.0, 0.0, 0.0)
        result = v.move_toward(target, speed=2.0)
        assert result == target
    
    def test_move_toward_partial(self):
        v = Vector3(0.0, 0.0, 0.0)
        target = Vector3(10.0, 0.0, 0.0)
        result = v.move_toward(target, speed=3.0)
        assert abs(result.x - 3.0) < 1e-9
        assert result.y == 0.0
        assert result.z == 0.0


class TestCRDTState:
    """Tests for CRDTState CRDT properties."""
    
    def test_mark_task_complete_adds_to_set(self):
        state = CRDTState(robot_id="robot_1")
        state.mark_task_complete("task_1", timestamp=100)
        assert "task_1" in state.completed_tasks
    
    def test_completed_tasks_never_removed(self):
        state = CRDTState(robot_id="robot_1")
        state.mark_task_complete("task_1", timestamp=100)
        # There's no way to remove - that's the point
        assert "task_1" in state.completed_tasks
    
    def test_add_progress_positive(self):
        state = CRDTState(robot_id="robot_1")
        state.add_progress("task_1", amount=5)
        assert state.get_task_progress("task_1") == 5
    
    def test_add_progress_negative_raises(self):
        state = CRDTState(robot_id="robot_1")
        with pytest.raises(ValueError):
            state.add_progress("task_1", amount=-5)
    
    def test_progress_accumulates(self):
        state = CRDTState(robot_id="robot_1")
        state.add_progress("task_1", amount=5)
        state.add_progress("task_1", amount=3)
        assert state.get_task_progress("task_1") == 8
    
    def test_claim_task_first_wins(self):
        state = CRDTState(robot_id="robot_1")
        assert state.claim_task("task_1", "robot_1", timestamp=100) == True
        assert state.claim_task("task_1", "robot_2", timestamp=200) == False
        assert state.get_task_claimer("task_1") == "robot_1"
    
    def test_position_update_lww(self):
        state = CRDTState(robot_id="robot_1")
        pos1 = Vector3(1.0, 2.0, 3.0)
        pos2 = Vector3(4.0, 5.0, 6.0)
        
        state.update_position("robot_1", pos1, timestamp=100)
        state.update_position("robot_1", pos2, timestamp=200)
        
        assert state.robot_positions["robot_1"][0] == pos2
    
    def test_position_update_ignores_older(self):
        state = CRDTState(robot_id="robot_1")
        pos1 = Vector3(1.0, 2.0, 3.0)
        pos2 = Vector3(4.0, 5.0, 6.0)
        
        state.update_position("robot_1", pos2, timestamp=200)
        state.update_position("robot_1", pos1, timestamp=100)  # Older, ignored
        
        assert state.robot_positions["robot_1"][0] == pos2


class TestCRDTMerge:
    """Tests for CRDT merge properties."""
    
    def test_merge_completed_tasks_union(self):
        state1 = CRDTState(robot_id="robot_1")
        state2 = CRDTState(robot_id="robot_2")
        
        state1.mark_task_complete("task_1", timestamp=100)
        state2.mark_task_complete("task_2", timestamp=100)
        
        state1.merge(state2)
        
        assert "task_1" in state1.completed_tasks
        assert "task_2" in state1.completed_tasks
    
    def test_merge_progress_takes_max(self):
        state1 = CRDTState(robot_id="robot_1")
        state2 = CRDTState(robot_id="robot_1")
        
        state1.add_progress("task_1", amount=5)
        state2.add_progress("task_1", amount=3)
        
        state1.merge(state2)
        
        # Should keep max (5), not sum (8)
        assert state1.get_task_progress("task_1") == 5
    
    def test_merge_progress_from_different_robots(self):
        state1 = CRDTState(robot_id="robot_1")
        state2 = CRDTState(robot_id="robot_2")
        
        state1.add_progress("task_1", amount=5)
        state2.add_progress("task_1", amount=3)
        
        state1.merge(state2)
        
        # Sum of contributions from different robots
        assert state1.get_task_progress("task_1") == 8
    
    def test_merge_commutativity(self):
        """merge(A, B) should equal merge(B, A)"""
        state_a = CRDTState(robot_id="robot_a")
        state_b = CRDTState(robot_id="robot_b")
        
        state_a.mark_task_complete("task_1", 100)
        state_a.add_progress("task_2", 5)
        
        state_b.mark_task_complete("task_3", 100)
        state_b.add_progress("task_2", 3)
        
        # merge(A, B)
        ab = state_a.copy()
        ab.merge(state_b)
        
        # merge(B, A)
        ba = state_b.copy()
        ba.merge(state_a)
        
        assert ab.completed_tasks == ba.completed_tasks
        assert ab.get_task_progress("task_2") == ba.get_task_progress("task_2")
    
    def test_merge_associativity(self):
        """merge(merge(A, B), C) should equal merge(A, merge(B, C))"""
        state_a = CRDTState(robot_id="robot_a")
        state_b = CRDTState(robot_id="robot_b")
        state_c = CRDTState(robot_id="robot_c")
        
        state_a.mark_task_complete("task_1", 100)
        state_b.mark_task_complete("task_2", 100)
        state_c.mark_task_complete("task_3", 100)
        
        # (A merge B) merge C
        ab = state_a.copy()
        ab.merge(state_b)
        ab_c = ab.copy()
        ab_c.merge(state_c)
        
        # A merge (B merge C)
        bc = state_b.copy()
        bc.merge(state_c)
        a_bc = state_a.copy()
        a_bc.merge(bc)
        
        assert ab_c.completed_tasks == a_bc.completed_tasks
    
    def test_merge_idempotency(self):
        """merge(A, A) should equal A"""
        state = CRDTState(robot_id="robot_1")
        state.mark_task_complete("task_1", 100)
        state.add_progress("task_2", 5)
        
        original = state.copy()
        state.merge(original)
        
        assert state.completed_tasks == original.completed_tasks
        assert state.get_task_progress("task_2") == original.get_task_progress("task_2")
    
    def test_merge_claims_first_write_wins(self):
        """Earlier claim should win after merge"""
        state1 = CRDTState(robot_id="robot_1")
        state2 = CRDTState(robot_id="robot_2")
        
        state1.claim_task("task_1", "robot_1", timestamp=100)
        state2.claim_task("task_1", "robot_2", timestamp=200)
        
        state1.merge(state2)
        state2.merge(state1)
        
        # robot_1 claimed first (timestamp 100)
        assert state1.get_task_claimer("task_1") == "robot_1"
        assert state2.get_task_claimer("task_1") == "robot_1"


class TestCRDTMonotonicity:
    """Tests that CRDT state only moves forward."""
    
    def test_completed_tasks_monotonic(self):
        state1 = CRDTState(robot_id="robot_1")
        state2 = CRDTState(robot_id="robot_2")
        
        state1.mark_task_complete("task_1", 100)
        original_count = len(state1.completed_tasks)
        
        state1.merge(state2)
        
        assert len(state1.completed_tasks) >= original_count
    
    def test_progress_monotonic(self):
        state1 = CRDTState(robot_id="robot_1")
        state2 = CRDTState(robot_id="robot_2")
        
        state1.add_progress("task_1", 5)
        original_progress = state1.get_task_progress("task_1")
        
        state2.add_progress("task_1", 3)
        state1.merge(state2)
        
        assert state1.get_task_progress("task_1") >= original_progress


class TestVerifyCRDTProperties:
    """Tests for the verification function."""
    
    def test_verify_properties_all_pass(self):
        state_a = CRDTState(robot_id="robot_a")
        state_b = CRDTState(robot_id="robot_b")
        state_c = CRDTState(robot_id="robot_c")
        
        state_a.mark_task_complete("task_1", 100)
        state_b.add_progress("task_2", 5)
        state_c.update_position("robot_c", Vector3(1, 2, 3), 100)
        
        results = verify_crdt_properties(state_a, state_b, state_c)
        
        assert results["commutative"] == True
        assert results["associative"] == True
        assert results["idempotent"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
