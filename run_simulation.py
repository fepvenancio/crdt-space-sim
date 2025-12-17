#!/usr/bin/env python
"""
Quick run script for CRDT Space Robotics Simulation.

Usage:
    python run_simulation.py
    python run_simulation.py --reliability 0.5
    python run_simulation.py --robots 10 --tasks 20
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from legacy for now (until refactor complete)
from legacy.simulation import Simulation, reliability_sweep


def main():
    parser = argparse.ArgumentParser(description="CRDT Space Robotics Simulation")
    parser.add_argument("--robots", type=int, default=5, help="Number of robots")
    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--reliability", type=float, default=0.7, help="Comms reliability (0-1)")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--sweep", action="store_true", help="Run reliability sweep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print(" CRDT SPACE ROBOTICS SIMULATION")
    print("=" * 60)
    
    if args.sweep:
        print("\nRunning reliability sweep...\n")
        reliability_sweep()
    else:
        print(f"\nConfiguration:")
        print(f"  Robots: {args.robots}")
        print(f"  Tasks: {args.tasks}")
        print(f"  Comms Reliability: {args.reliability * 100:.0f}%")
        print(f"  Trials: {args.trials}")
        print()
        
        sim = Simulation(
            num_robots=args.robots,
            num_tasks=args.tasks,
            comms_reliability=args.reliability,
            seed=args.seed
        )
        
        results = sim.run_comparison(num_trials=args.trials)
        
        print("\nSimulation complete!")
        print(f"Results saved to: output/simulation_results.json")


if __name__ == "__main__":
    main()
