# validation_agent.py
import numpy as np
from sklearn.model_selection import train_test_split
import subprocess
import tempfile
import os
from typing import Dict, Tuple
import yaml

class ValidationAgent:
    def __init__(self, parent_agent=None):
        self.parent_agent = parent_agent
        self.simulation_validator = SimulationValidator()
        self.stability_checker = StabilityChecker()

    def validate(self, params: Dict) -> bool:
        """验证参数的有效性"""
        # 1. 参数合理性检查
        if not self.check_parameter_reasonableness(params):
            print("Parameter reasonableness check failed")
            return False
        
        # 2. 仿真验证
        sim_success, sim_metrics = self.simulation_validator.test_parameters(params)
        if not sim_success:
            print("Simulation validation failed")
            return False
        
        # 3. 稳定性检查
        stability_ok = self.stability_checker.check_stability(params)
        if not stability_ok:
            print("Stability check failed")
            return False
        
        print("All validations passed")
        return True

    def check_parameter_reasonableness(self, params: Dict) -> bool:
        """检查参数合理性"""
        if not params:
            return False
        
        # 检查关键参数是否在合理范围内
        checks = [
            self._check_positive_values(params),
            self._check_parameter_ranges(params),
            self._check_parameter_consistency(params)
        ]
        
        return all(checks)

    def _check_positive_values(self, params: Dict) -> bool:
        """检查正值参数"""
        positive_params = ['mapping_process_noise', 'feature_extraction_threshold', 
                          'max_features', 'scale_factor']
        
        for param_name in positive_params:
            if param_name in params and params[param_name] <= 0:
                return False
        
        return True

    def _check_parameter_ranges(self, params: Dict) -> bool:
        """检查参数范围"""
        ranges = {
            'feature_extraction_threshold': (0.01, 2.0),
            'max_features': (100, 5000),
            'scale_factor': (1.01, 2.0),
            'match_ratio_test': (0.1, 0.95),
            'convergence_epsilon': (1e-8, 1e-2)
        }
        
        for param_name, (min_val, max_val) in ranges.items():
            if param_name in params:
                val = params[param_name]
                if not (min_val <= val <= max_val):
                    return False
        
        return True

    def _check_parameter_consistency(self, params: Dict) -> bool:
        """检查参数一致性"""
        # 例如：特征点数量不能少于最小匹配数
        max_features = params.get('max_features', 1000)
        levels = params.get('levels', 8)
        
        if max_features < levels * 10:  # 每层至少10个特征点
            return False
        
        return True


class SimulationValidator:
    def __init__(self):
        self.temp_config_dir = "/tmp/slam_configs"
        os.makedirs(self.temp_config_dir, exist_ok=True)

    def test_parameters(self, params: Dict) -> Tuple[bool, Dict]:
        """在仿真环境中测试参数"""
        try:
            # 1. 创建临时配置文件
            config_path = self._create_temp_config(params)
            
            # 2. 启动仿真测试
            success, metrics = self._run_simulation_test(config_path)
            
            # 3. 清理临时文件
            self._cleanup_temp_files(config_path)
            
            return success, metrics
            
        except Exception as e:
            print(f"Simulation test error: {e}")
            return False, {}

    def _create_temp_config(self, params: Dict) -> str:
        """创建临时配置文件"""
        config = {
            'lio_sam': {},
            'orb_slam': {}
        }
        
        # 根据参数类型分组
        for param_name, value in params.items():
            if any(keyword in param_name.lower() for keyword in ['lidar', 'imu', 'mapping']):
                config['lio_sam'][param_name] = value
            elif any(keyword in param_name.lower() for keyword in ['feature', 'orb', 'track']):
                config['orb_slam'][param_name] = value
            else:
                # 默认分配到ORB-SLAM
                config['orb_slam'][param_name] = value
        
        # 生成临时文件
        temp_file = os.path.join(self.temp_config_dir, f"config_{int(time.time())}.yaml")
        with open(temp_file, 'w') as f:
            yaml.dump(config, f)
        
        return temp_file

    def _run_simulation_test(self, config_path: str) -> Tuple[bool, Dict]:
        """运行仿真测试"""
        # 这里应该连接到实际的仿真环境
        # 模拟仿真过程
        print(f"Running simulation with config: {config_path}")
        
        # 模拟测试过程
        time.sleep(2)  # 模拟仿真运行时间
        
        # 模拟测试结果
        success = np.random.random() > 0.2  # 80%成功率
        metrics = {
            'rmse_position': np.random.uniform(0.01, 0.5),
            'rmse_orientation': np.random.uniform(0.001, 0.1),
            'tracking_success_rate': np.random.uniform(0.8, 1.0),
            'processing_time': np.random.uniform(0.01, 0.1),
            'memory_usage': np.random.uniform(100, 1000)  # MB
        }
        
        # 判断是否成功
        success = (
            metrics['rmse_position'] < 0.3 and
            metrics['tracking_success_rate'] > 0.85 and
            metrics['processing_time'] < 0.05
        )
        
        return success, metrics

    def _cleanup_temp_files(self, config_path: str):
        """清理临时文件"""
        try:
            os.remove(config_path)
        except:
            pass


class StabilityChecker:
    def __init__(self):
        self.stability_history = []

    def check_stability(self, params: Dict) -> bool:
        """检查参数稳定性"""
        # 1. 参数敏感性分析
        sensitivity_score = self._analyze_sensitivity(params)
        
        # 2. 历史稳定性检查
        historical_stability = self._check_historical_stability(params)
        
        # 3. 综合评估
        overall_stability = (sensitivity_score + historical_stability) / 2.0
        
        # 记录本次检查结果
        self.stability_history.append({
            'params_hash': hash(str(sorted(params.items()))),
            'stability_score': overall_stability,
            'timestamp': time.time()
        })
        
        # 限制历史记录数量
        if len(self.stability_history) > 100:
            self.stability_history = self.stability_history[-50:]
        
        return overall_stability > 0.7  # 稳定性阈值

    def _analyze_sensitivity(self, params: Dict) -> float:
        """分析参数敏感性"""
        # 通过微小扰动测试参数稳定性
        base_performance = self._simulate_performance(params)
        
        perturbed_performances = []
        for param_name, base_value in params.items():
            # 对参数施加小扰动
            perturbation = base_value * 0.05  # 5%扰动
            perturbed_params = params.copy()
            perturbed_params[param_name] = base_value + perturbation
            
            perturbed_performance = self._simulate_performance(perturbed_params)
            perturbed_performances.append(abs(base_performance - perturbed_performance))
        
        # 敏感性越低越好（稳定性越高）
        avg_sensitivity = np.mean(perturbed_performances) if perturbed_performances else 0.1
        return max(0.0, 1.0 - avg_sensitivity)  # 转换为稳定性分数

    def _simulate_performance(self, params: Dict) -> float:
        """模拟性能评估（实际应用中应连接真实评估）"""
        # 模拟性能计算
        performance = 0.0
        
        # 基于参数值计算性能分数
        for value in params.values():
            if isinstance(value, (int, float)):
                performance += abs(value) * 0.01
        
        # 添加一些随机性
        performance += np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, performance))

    def _check_historical_stability(self, params: Dict) -> float:
        """检查历史稳定性"""
        if not self.stability_history:
            return 0.5  # 无历史数据，返回中等分数
        
        current_hash = hash(str(sorted(params.items())))
        
        # 查找相似参数的历史稳定性
        similar_records = [
            record for record in self.stability_history
            if record['params_hash'] == current_hash
        ]
        
        if similar_records:
            # 返回该参数的历史平均稳定性
            return np.mean([r['stability_score'] for r in similar_records])
        else:
            # 新参数，返回历史平均稳定性
            return np.mean([r['stability_score'] for r in self.stability_history])
