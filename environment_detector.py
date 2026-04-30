# environment_detector.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import cv2

class EnvironmentDetector:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_extractor = FeatureExtractor()

    def analyze_environment(self) -> str:
        """分析当前环境类型"""
        # 这里应该从传感器获取实时数据
        # 模拟环境检测
        features = self._extract_current_features()
        
        if not self.is_trained:
            # 如果未训练，使用默认规则
            return self._rule_based_classification(features)
        
        # 使用训练好的分类器
        features_scaled = self.scaler.transform([features])
        prediction = self.classifier.predict(features_scaled)[0]
        return prediction

    def _extract_current_features(self) -> np.ndarray:
        """提取当前环境特征"""
        # 模拟特征提取
        features = [
            np.random.uniform(0, 10),  # 特征密度
            np.random.uniform(0, 1),   # 光照变化
            np.random.uniform(0, 5),   # 几何复杂度
            np.random.uniform(0, 1),   # 纹理丰富度
            np.random.uniform(0, 100)  # 动态物体数量
        ]
        return np.array(features)

    def _rule_based_classification(self, features: np.ndarray) -> str:
        """基于规则的环境分类"""
        feature_density, lighting_change, geometry_complexity, texture_richness, dynamic_objects = features
        
        if feature_density < 2 and texture_richness < 0.3:
            return "low_texture_indoor"  # 低纹理室内
        elif lighting_change > 0.7:
            return "high_dynamic_lighting"  # 高动态光照
        elif geometry_complexity > 7:
            return "complex_geometry"  # 复杂几何
        elif dynamic_objects > 50:
            return "dynamic_environment"  # 动态环境
        elif feature_density > 8 and geometry_complexity > 5:
            return "rich_feature_outdoor"  # 丰富特征户外
        else:
            return "general_indoor"  # 一般室内

    def train_classifier(self, training_data: list, labels: list):
        """训练环境分类器"""
        features_array = np.array(training_data)
        labels_array = np.array(labels)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features_array)
        
        # 训练分类器
        self.classifier.fit(features_scaled, labels_array)
        self.is_trained = True

    def update_with_new_data(self, new_features: np.ndarray, environment_type: str):
        """使用新数据更新分类器"""
        if self.is_trained:
            # 在线学习更新
            features_scaled = self.scaler.transform([new_features])
            self.classifier.partial_fit(features_scaled, [environment_type])


class FeatureExtractor:
    def __init__(self):
        pass

    def extract_from_image(self, image: np.ndarray) -> Dict:
        """从图像提取特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算特征密度
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, 
                                        qualityLevel=0.01, minDistance=10)
        feature_density = len(corners) if corners is not None else 0
        
        # 计算光照变化
        lighting_change = np.std(gray) / 255.0
        
        # 计算纹理丰富度（使用LBP或其他纹理描述符）
        texture_richness = self._calculate_texture_richness(gray)
        
        return {
            'feature_density': feature_density,
            'lighting_change': lighting_change,
            'texture_richness': texture_richness
        }

    def extract_from_pointcloud(self, pointcloud: np.ndarray) -> Dict:
        """从点云提取特征"""
        if len(pointcloud) == 0:
            return {
                'geometry_complexity': 0,
                'point_density': 0
            }
        
        # 计算几何复杂度
        geometry_complexity = self._calculate_geometry_complexity(pointcloud)
        
        # 计算点密度
        point_density = len(pointcloud) / self._calculate_volume(pointcloud)
        
        return {
            'geometry_complexity': geometry_complexity,
            'point_density': point_density
        }

    def _calculate_texture_richness(self, gray_image: np.ndarray) -> float:
        """计算纹理丰富度"""
        # 使用梯度幅值作为纹理复杂度指标
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return np.mean(gradient_magnitude) / 255.0

    def _calculate_geometry_complexity(self, pointcloud: np.ndarray) -> float:
        """计算几何复杂度"""
        if len(pointcloud) < 3:
            return 0.0
        
        # 计算曲率（简化版）
        # 这里可以使用更复杂的几何分析方法
        distances = []
        for i in range(min(100, len(pointcloud))):
            for j in range(i+1, min(101, len(pointcloud))):
                dist = np.linalg.norm(pointcloud[i] - pointcloud[j])
                distances.append(dist)
        
        return np.std(distances) if distances else 0.0

    def _calculate_volume(self, pointcloud: np.ndarray) -> float:
        """计算包围体积"""
        if len(pointcloud) == 0:
            return 1.0
        
        min_coords = np.min(pointcloud, axis=0)
        max_coords = np.max(pointcloud, axis=0)
        volume = np.prod(max_coords - min_coords)
        
        return max(volume, 1.0)  # 避免除零
