# analysis_agent.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import pandas as pd

class AnalysisAgent:
    def __init__(self, parent_agent):
        self.parent_agent = parent_agent
        self.pattern_classifier = PatternClassifier()
        self.drift_detector = DriftDetector()

    def analyze_problem(self, metrics: Dict) -> Dict:
        """分析问题原因"""
        with self.parent_agent.data_lock:
            # 获取历史数据进行深度分析
            pose_data = list(self.parent_agent.pose_history)
            residual_data = list(self.parent_agent.pointcloud_residuals)
            aruco_data = list(self.parent_agent.aruco_errors)
        
        # 模式识别
        pattern_analysis = self.pattern_classifier.classify_patterns(
            pose_data, residual_data, aruco_data
        )
        
        # 漂移类型判断
        drift_type = self.drift_detector.identify_drift_type(
            pose_data, residual_data
        )
        
        # 传感器故障检测
        sensor_health = self.check_sensor_health(
            residual_data, aruco_data
        )
        
        analysis_result = {
            'problem_type': drift_type,
            'severity_level': self.calculate_severity(metrics),
            'affected_modules': self.identify_affected_modules(pattern_analysis),
            'recommended_fixes': self.generate_recommendations(
                drift_type, sensor_health
            ),
            'confidence_score': self.calculate_confidence(pattern_analysis)
        }
        
        print(f"Analysis Result: {analysis_result}")
        return analysis_result

    def check_sensor_health(self, residuals: List, aruco_errors: List) -> Dict:
        """检查传感器健康状态"""
        health_status = {}
        
        # LiDAR健康度
        if residuals:
            lidar_health = 1.0 - min(np.std(residuals), 0.1) / 0.1
            health_status['lidar'] = max(lidar_health, 0.0)
        
        # 相机健康度
        if aruco_errors:
            camera_health = 1.0 - min(np.mean(aruco_errors), 5.0) / 5.0
            health_status['camera'] = max(camera_health, 0.0)
        
        return health_status

    def identify_affected_modules(self, pattern_analysis: Dict) -> List[str]:
        """识别受影响的模块"""
        affected = []
        
        if pattern_analysis.get('tracking_loss', False):
            affected.append('tracking')
        
        if pattern_analysis.get('mapping_inconsistency', False):
            affected.append('mapping')
        
        if pattern_analysis.get('sensor_fusion_issue', False):
            affected.append('fusion')
        
        return affected

    def generate_recommendations(self, drift_type: str, sensor_health: Dict) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        if drift_type == 'imu_bias':
            recommendations.append('increase_imu_weight')
            recommendations.append('calibrate_imu')
        
        if drift_type == 'feature_depletion':
            recommendations.append('adjust_feature_extraction')
            recommendations.append('increase_tracking_range')
        
        if sensor_health.get('lidar', 1.0) < 0.7:
            recommendations.append('reduce_lidar_dependence')
            recommendations.append('increase_visual_weight')
        
        if sensor_health.get('camera', 1.0) < 0.7:
            recommendations.append('reduce_camera_dependence')
            recommendations.append('increase_lidar_weight')
        
        return recommendations

    def calculate_severity(self, metrics: Dict) -> int:
        """计算问题严重程度"""
        severity = 0
        if metrics['drift_rate'] > 0.1:
            severity += 3
        elif metrics['drift_rate'] > 0.05:
            severity += 2
        else:
            severity += 1
        
        if metrics['avg_residual'] > 0.2:
            severity += 2
        elif metrics['avg_residual'] > 0.1:
            severity += 1
        
        return min(severity, 5)

    def calculate_confidence(self, pattern_analysis: Dict) -> float:
        """计算分析置信度"""
        confidence_factors = [
            pattern_analysis.get('pattern_strength', 0.5),
            pattern_analysis.get('data_quality', 0.5),
            pattern_analysis.get('historical_match', 0.5)
        ]
        return np.mean(confidence_factors)


class PatternClassifier:
    def __init__(self):
        self.model = IsolationForest(contamination=0.1)
        self.kmeans = KMeans(n_clusters=5)

    def classify_patterns(self, pose_data, residual_data, aruco_data) -> Dict:
        """分类漂移模式"""
        features = self.extract_features(pose_data, residual_data, aruco_data)
        
        if len(features) == 0:
            return {'unknown_pattern': True}
        
        # 使用聚类识别模式
        clusters = self.kmeans.fit_predict(features.reshape(-1, 1))
        
        # 分析模式特征
        pattern_analysis = {
            'tracking_loss': self.detect_tracking_loss(pose_data),
            'mapping_inconsistency': self.detect_mapping_inconsistency(residual_data),
            'sensor_fusion_issue': self.detect_sensor_fusion_issue(aruco_data, residual_data),
            'pattern_strength': self.calculate_pattern_strength(clusters),
            'data_quality': self.assess_data_quality(features)
        }
        
        return pattern_analysis

    def extract_features(self, pose_data, residual_data, aruco_data):
        """提取特征向量"""
        features = []
        
        if pose_data:
            positions = [p['position'] for p in pose_data]
            speeds = [np.linalg.norm(np.diff([pos[0] for pos in positions[-10:]])) 
                     if len(positions) >= 10 else 0.0]
            features.extend(speeds)
        
        if residual_data:
            features.extend([np.mean(residual_data[-50:]), np.std(residual_data[-50:])])
        
        if aruco_data:
            features.extend([np.mean(aruco_data[-50:]), np.std(aruco_data[-50:])])
        
        return np.array(features)

    def detect_tracking_loss(self, pose_data) -> bool:
        """检测跟踪丢失"""
        if len(pose_data) < 10:
            return False
        
        # 检查位置跳跃
        positions = [p['position'] for p in pose_data[-10:]]
        distances = [np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i])) 
                    for i in range(len(positions)-1)]
        
        return any(dist > 1.0 for dist in distances)  # 1米以上的跳跃

    def detect_mapping_inconsistency(self, residual_data) -> bool:
        """检测建图不一致性"""
        if len(residual_data) < 20:
            return False
        
        recent_avg = np.mean(residual_data[-10:])
        historical_avg = np.mean(residual_data[:-10])
        
        return recent_avg > historical_avg * 2.0

    def detect_sensor_fusion_issue(self, aruco_data, residual_data) -> bool:
        """检测传感器融合问题"""
        if not aruco_data or not residual_data:
            return False
        
        aruco_var = np.var(aruco_data[-20:])
        residual_var = np.var(residual_data[-20:])
        
        # 如果两个传感器的方差差异过大，可能存在融合问题
        return abs(aruco_var - residual_var) > 0.1

    def calculate_pattern_strength(self, clusters) -> float:
        """计算模式强度"""
        unique_clusters = len(np.unique(clusters))
        return min(unique_clusters / 5.0, 1.0)

    def assess_data_quality(self, features) -> float:
        """评估数据质量"""
        if len(features) == 0:
            return 0.0
        
        # 基于数据的稳定性和完整性评估质量
        quality = 1.0 - min(len(features) / 100.0, 1.0)  # 数据量影响
        quality *= (1.0 - np.std(features) / 10.0)  # 稳定性影响
        
        return max(quality, 0.0)


class DriftDetector:
    def __init__(self):
        pass

    def identify_drift_type(self, pose_data, residual_data) -> str:
        """识别漂移类型"""
        if not pose_data:
            return 'unknown'
        
        # 分析位姿变化模式
        positions = [p['position'] for p in pose_data[-50:]]
        
        # 计算移动速度和加速度
        if len(positions) >= 2:
            velocities = []
            for i in range(1, len(positions)):
                v = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
                velocities.append(v)
            
            if len(velocities) >= 2:
                accelerations = []
                for i in range(1, len(velocities)):
                    a = abs(velocities[i] - velocities[i-1])
                    accelerations.append(a)
                
                # 判断漂移类型
                if np.mean(accelerations) > 0.1 and np.mean(velocities) < 0.05:
                    return 'static_bias'  # 静态偏差
                elif np.mean(velocities) > 0.2 and np.mean(accelerations) > 0.05:
                    return 'dynamic_drift'  # 动态漂移
                else:
                    return 'imu_bias'  # IMU偏差
        
        # 检查特征点数量是否不足
        if residual_data and np.mean(residual_data[-20:]) > 0.15:
            return 'feature_depletion'  # 特征点不足
        
        return 'normal_drift'
