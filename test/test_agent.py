#!/usr/bin/env python3
"""
Unit tests for SLAM Auto Tuning Agent components
"""

import unittest
import numpy as np
from src import (
    SLAMAutoTuningAgent,
    AnalysisAgent,
    OptimizationAgent,
    ValidationAgent,
    EnvironmentDetector
)

class TestSLAMAutoTuningAgent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.agent = SLAMAutoTuningAgent()
    
    def test_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent)
        self.assertEqual(len(self.agent.pose_history), 0)
        self.assertEqual(len(self.agent.pointcloud_residuals), 0)
        self.assertEqual(len(self.agent.aruco_errors), 0)
        self.assertIsNotNone(self.agent.param_space)
    
    def test_collect_sensor_data(self):
        """Test sensor data collection"""
        initial_length = len(self.agent.pose_history)
        self.agent.collect_sensor_data()
        
        # Should have collected data
        self.assertGreater(len(self.agent.pose_history), initial_length)
        self.assertGreater(len(self.agent.pointcloud_residuals), 0)
        self.assertGreater(len(self.agent.aruco_errors), 0)
    
    def test_evaluate_performance(self):
        """Test performance evaluation"""
        # Add some test data
        for _ in range(100):
            self.agent.collect_sensor_data()
        
        metrics = self.agent.evaluate_performance()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('drift_rate', metrics)
        self.assertIn('avg_residual', metrics)
        self.assertIn('avg_aruco_error', metrics)
        self.assertIn('stability', metrics)
    
    def test_detect_drift(self):
        """Test drift detection"""
        # Test with normal metrics
        normal_metrics = {
            'drift_rate': 0.01,
            'stability': 0.9
        }
        self.assertFalse(self.agent.detect_drift(normal_metrics))
        
        # Test with high drift
        high_drift_metrics = {
            'drift_rate': 0.1,
            'stability': 0.5
        }
        self.assertTrue(self.agent.detect_drift(high_drift_metrics))
    
    def test_calculate_drift_rate(self):
        """Test drift rate calculation"""
        positions = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
        drift_rate = self.agent.calculate_drift_rate(positions)
        
        self.assertIsInstance(drift_rate, float)
        self.assertGreaterEqual(drift_rate, 0.0)

class TestAnalysisAgent(unittest.TestCase):
    def setUp(self):
        self.main_agent = SLAMAutoTuningAgent()
        self.analysis_agent = AnalysisAgent(self.main_agent)
    
    def test_analyze_problem(self):
        """Test problem analysis"""
        test_metrics = {
            'drift_rate': 0.05,
            'avg_residual': 0.1,
            'avg_aruco_error': 1.0,
            'stability': 0.8
        }
        
        result = self.analysis_agent.analyze_problem(test_metrics)
        
        self.assertIsInstance(result, dict)
        self.assertIn('problem_type', result)
        self.assertIn('severity_level', result)
        self.assertIn('affected_modules', result)
        self.assertIn('recommended_fixes', result)
        self.assertIn('confidence_score', result)

class TestOptimizationAgent(unittest.TestCase):
    def setUp(self):
        self.main_agent = SLAMAutoTuningAgent()
        self.optimization_agent = OptimizationAgent(self.main_agent)
    
    def test_create_objective_function(self):
        """Test objective function creation"""
        analysis_result = {
            'problem_type': 'normal',
            'severity_level': 1,
            'affected_modules': [],
            'recommended_fixes': [],
            'confidence_score': 0.8
        }
        
        obj_func = self.optimization_agent.create_objective_function(
            'lio_sam', analysis_result
        )
        
        # Test that function can be called
        test_params = {'mapping_process_noise': 0.01}
        result = obj_func(test_params)
        
        self.assertIsInstance(result, (int, float))
    
    def test_get_search_space(self):
        """Test search space retrieval"""
        analysis_result = {'affected_modules': ['tracking']}
        
        search_space = self.optimization_agent.get_search_space(
            'orb_slam', analysis_result
        )
        
        self.assertIsInstance(search_space, dict)
        self.assertIn('max_features', search_space)

class TestValidationAgent(unittest.TestCase):
    def setUp(self):
        self.validation_agent = ValidationAgent()
    
    def test_check_parameter_reasonableness(self):
        """Test parameter reasonableness checking"""
        # Valid parameters
        valid_params = {
            'feature_extraction_threshold': 0.5,
            'max_features': 1000,
            'scale_factor': 1.2
        }
        self.assertTrue(self.validation_agent.check_parameter_reasonableness(valid_params))
        
        # Invalid parameters
        invalid_params = {
            'feature_extraction_threshold': -0.1,  # Negative
            'max_features': 0,  # Zero or negative
            'scale_factor': 0.5  # Too small
        }
        self.assertFalse(self.validation_agent.check_parameter_reasonableness(invalid_params))

class TestEnvironmentDetector(unittest.TestCase):
    def setUp(self):
        self.detector = EnvironmentDetector()
    
    def test_rule_based_classification(self):
        """Test rule-based environment classification"""
        # Test various feature combinations
        features1 = np.array([1.0, 0.1, 2.0, 0.2, 5.0])  # Low texture indoor
        result1 = self.detector._rule_based_classification(features1)
        self.assertIsInstance(result1, str)
        
        features2 = np.array([10.0, 0.8, 8.0, 0.9, 100.0])  # Rich feature outdoor
        result2 = self.detector._rule_based_classification(features2)
        self.assertIsInstance(result2, str)

class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        """Test the full pipeline integration"""
        agent = SLAMAutoTuningAgent()
        
        # Collect some data
        for _ in range(10):
            agent.collect_sensor_data()
        
        # Evaluate performance
        metrics = agent.evaluate_performance()
        self.assertIsInstance(metrics, dict)
        
        # Analyze problems
        analysis_result = agent.analysis_agent.analyze_problem(metrics)
        self.assertIsInstance(analysis_result, dict)
        
        # Optimize parameters
        optimized_params = agent.optimization_agent.optimize_parameters(
            'lio_sam', analysis_result, {}
        )
        self.assertIsInstance(optimized_params, dict)
        
        # Validate parameters
        is_valid = agent.validate_parameters(optimized_params)
        self.assertIsInstance(is_valid, bool)

if __name__ == '__main__':
    print("Running SLAM Auto Tuning Agent tests...")
    unittest.main(verbosity=2)
