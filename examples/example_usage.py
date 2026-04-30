#!/usr/bin/env python3
"""
Example usage of the SLAM Auto Tuning Agent
"""

import time
import numpy as np
from src import SLAMAutoTuningAgent

def main():
    """Example demonstrating basic usage"""
    print("Starting SLAM Auto Tuning Agent Example")
    print("=" * 50)
    
    # Initialize the agent
    agent = SLAMAutoTuningAgent()
    
    print("Agent initialized successfully!")
    print(f"Supported SLAM systems: {list(agent.param_space.keys())}")
    
    # Simulate running for a short period
    print("\nStarting monitoring and optimization...")
    print("(This would normally connect to actual SLAM system)")
    
    # In a real scenario, this would run continuously
    # For demonstration, we'll simulate some operations
    
    for i in range(5):
        print(f"\nIteration {i+1}/5")
        
        # Simulate collecting some data
        agent.collect_sensor_data()
        
        # Evaluate current performance
        metrics = agent.evaluate_performance()
        print(f"Current performance metrics: {metrics}")
        
        # Check for drift
        if agent.detect_drift(metrics):
            print("Drift detected! Starting optimization...")
            
            # Simulate environment type
            env_type = "general_indoor"
            
            # Start optimization (this would normally trigger the full process)
            analysis_result = agent.analysis_agent.analyze_problem(metrics)
            print(f"Analysis result: {analysis_result}")
            
            # Simulate optimization
            optimized_params = agent.optimization_agent.optimize_parameters(
                env_type, analysis_result, agent.current_params
            )
            print(f"Optimized parameters: {optimized_params}")
            
            # Validate and apply
            if agent.validate_parameters(optimized_params):
                agent.apply_parameters(optimized_params)
                print("Parameters applied successfully!")
        
        time.sleep(1)  # Simulate time between checks
    
    print("\nExample completed!")
    print("In a real application, the agent would run continuously,")
    print("monitoring the SLAM system and automatically adjusting parameters.")

def advanced_example():
    """Advanced usage with custom configuration"""
    print("\n" + "="*50)
    print("Advanced Example - Custom Configuration")
    print("="*50)
    
    # Example of how to customize the agent
    agent = SLAMAutoTuningAgent()
    
    # Add custom environment detection
    print("Customizing environment detection...")
    
    # Modify parameter space for specific use case
    agent.param_space['custom_sam'] = {
        'custom_param_1': (0.01, 0.5),
        'custom_param_2': (1.0, 10.0),
        'custom_param_3': (0.1, 2.0)
    }
    
    print("Custom configuration applied!")
    print(f"Available parameter spaces: {list(agent.param_space.keys())}")

if __name__ == "__main__":
    main()
    advanced_example()
