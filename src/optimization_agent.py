# optimization_agent.py
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
import copy

class OptimizationAgent:
    def __init__(self, parent_agent):
        self.parent_agent = parent_agent
        self.gp_model = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            n_restarts_optimizer=10
        )
        self.experience_db = ParameterExperienceDB()
        self.optimizer = BayesianOptimizer()

    def optimize_parameters(self, environment_type: str, analysis_result: Dict, 
                          current_params: Dict) -> Dict:
        """优化参数"""
        # 获取目标函数
        objective_func = self.create_objective_function(
            environment_type, analysis_result
        )
        
        # 获取搜索空间
        search_space = self.get_search_space(environment_type, analysis_result)
        
        # 使用贝叶斯优化寻找最优参数
        best_params = self.optimizer.optimize(
            objective_func, search_space, current_params
        )
        
        # 应用约束条件
        constrained_params = self.apply_constraints(best_params, environment_type)
        
        print(f"Optimized parameters: {constrained_params}")
        return constrained_params

    def create_objective_function(self, env_type: str, analysis_result: Dict):
        """创建目标函数"""
        def objective(params):
            # 将参数转换为可优化格式
            param_vector = self.params_to_vector(params, env_type)
            
            # 预测性能
            predicted_performance = self.predict_performance(
                param_vector, env_type, analysis_result
            )
            
            # 目标是最小化负性能（即最大化性能）
            return -predicted_performance
        
        return objective

    def predict_performance(self, param_vector: np.ndarray, env_type: str, 
                          analysis_result: Dict) -> float:
        """预测参数性能"""
        # 结合历史经验和当前环境信息预测性能
        historical_performance = self.experience_db.query(
            env_type, param_vector
        )
        
        # 考虑当前问题的影响
        problem_penalty = self.calculate_problem_penalty(analysis_result)
        
        # 综合评分
        base_score = historical_performance.get('performance_score', 0.5)
        adjusted_score = base_score - problem_penalty
        
        return max(adjusted_score, 0.0)

    def get_search_space(self, env_type: str, analysis_result: Dict) -> Dict:
        """获取搜索空间"""
        base_space = self.parent_agent.param_space.get(env_type, {})
        
        # 根据分析结果调整搜索范围
        adjusted_space = copy.deepcopy(base_space)
        
        if 'tracking' in analysis_result.get('affected_modules', []):
            # 如果跟踪有问题，重点调整跟踪相关参数
            if env_type == 'orb_slam':
                adjusted_space['max_features'] = (1000, 2000)  # 增加特征点
                adjusted_space['match_ratio_test'] = (0.5, 0.7)  # 放宽匹配要求
        
        if 'mapping' in analysis_result.get('affected_modules', []):
            # 如果建图有问题，调整建图参数
            if env_type == 'lio_sam':
                adjusted_space['registration_fitness_score'] = (0.3, 0.8)  # 放宽配准要求
        
        return adjusted_space

    def calculate_problem_penalty(self, analysis_result: Dict) -> float:
        """计算问题惩罚"""
        penalty = 0.0
        
        severity = analysis_result.get('severity_level', 1)
        penalty += severity * 0.1
        
        affected_count = len(analysis_result.get('affected_modules', []))
        penalty += affected_count * 0.05
        
        confidence = analysis_result.get('confidence_score', 0.5)
        penalty *= (1.0 - confidence)  # 低置信度增加惩罚
        
        return min(penalty, 0.5)

    def apply_constraints(self, params: Dict, env_type: str) -> Dict:
        """应用约束条件"""
        constrained = {}
        
        for key, value in params.items():
            space = self.parent_agent.param_space[env_type].get(key)
            if space:
                min_val, max_val = space
                constrained[key] = np.clip(value, min_val, max_val)
            else:
                constrained[key] = value
        
        return constrained

    def params_to_vector(self, params: Dict, env_type: str) -> np.ndarray:
        """将参数字典转换为向量"""
        space = self.parent_agent.param_space[env_type]
        vector = []
        
        for param_name in sorted(space.keys()):
            if param_name in params:
                value = params[param_name]
                min_val, max_val = space[param_name]
                # 归一化到[0,1]
                normalized = (value - min_val) / (max_val - min_val)
                vector.append(normalized)
            else:
                vector.append(0.5)  # 默认值
        
        return np.array(vector)

    def vector_to_params(self, vector: np.ndarray, env_type: str) -> Dict:
        """将向量转换回参数字典"""
        space = self.parent_agent.param_space[env_type]
        params = {}
        
        param_names = sorted(space.keys())
        for i, param_name in enumerate(param_names):
            if i < len(vector):
                min_val, max_val = space[param_name]
                # 反归一化
                original_value = vector[i] * (max_val - min_val) + min_val
                params[param_name] = original_value
        
        return params


class BayesianOptimizer:
    def __init__(self):
        self.gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            n_restarts_optimizer=10
        )
        self.acquisition_function = ExpectedImprovement()

    def optimize(self, objective_func, search_space: Dict, 
                 current_params: Dict) -> Dict:
        """执行贝叶斯优化"""
        # 生成初始采样点
        initial_points = self.generate_initial_samples(search_space, 10)
        
        # 评估初始点
        X = []
        y = []
        for point in initial_points:
            X.append(point)
            y.append(objective_func(point))
        
        # 训练高斯过程模型
        X = np.array(X)
        y = np.array(y)
        self.gp.fit(X, y)
        
        # 进行迭代优化
        best_params = None
        best_value = float('inf')
        
        for iteration in range(20):  # 最多20次迭代
            # 选择下一个采样点
            next_point = self.acquisition_function.select_next_point(
                self.gp, search_space
            )
            
            # 评估新点
            new_value = objective_func(next_point)
            
            # 更新数据集
            X = np.vstack([X, next_point])
            y = np.append(y, new_value)
            
            # 重新训练模型
            self.gp.fit(X, y)
            
            # 更新最优解
            if new_value < best_value:
                best_value = new_value
                best_params = next_point
        
        return best_params

    def generate_initial_samples(self, search_space: Dict, n_samples: int) -> List[np.ndarray]:
        """生成初始采样点"""
        samples = []
        
        for _ in range(n_samples):
            sample = []
            for param_name in sorted(search_space.keys()):
                min_val, max_val = search_space[param_name]
                # 在[0,1]范围内随机采样（已归一化）
                sample.append(np.random.uniform(0, 1))
            
            samples.append(np.array(sample))
        
        return samples


class ExpectedImprovement:
    def __init__(self):
        self.xi = 0.01  # 探索参数

    def select_next_point(self, gp_model, search_space: Dict) -> np.ndarray:
        """选择下一个采样点"""
        # 在搜索空间中随机采样候选点
        candidates = []
        candidate_values = []
        
        for _ in range(100):  # 100个候选点
            candidate = []
            for param_name in sorted(search_space.keys()):
                candidate.append(np.random.uniform(0, 1))
            candidates.append(np.array(candidate))
        
        candidates = np.array(candidates)
        
        # 计算每个候选点的期望改进值
        mu, sigma = gp_model.predict(candidates, return_std=True)
        
        # 计算期望改进
        best_y = np.min(gp_model.y_train_)
        improvement = best_y - mu - self.xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # 选择EI最大的点
        best_idx = np.argmax(ei)
        return candidates[best_idx]


from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class ParameterExperienceDB:
    def __init__(self):
        self.experience_records = []
        self.performance_cache = {}

    def record_experience(self, env_type: str, params: Dict, 
                         performance: float, timestamp: float = None):
        """记录参数使用经验"""
        if timestamp is None:
            timestamp = time.time()
        
        record = {
            'environment': env_type,
            'parameters': params.copy(),
            'performance_score': performance,
            'timestamp': timestamp
        }
        
        self.experience_records.append(record)
        
        # 限制记录数量以节省内存
        if len(self.experience_records) > 1000:
            self.experience_records = self.experience_records[-500:]

    def query(self, env_type: str, param_vector: np.ndarray) -> Dict:
        """查询相似参数的历史性能"""
        # 找到相同环境类型的历史记录
        relevant_records = [
            rec for rec in self.experience_records 
            if rec['environment'] == env_type
        ]
        
        if not relevant_records:
            return {'performance_score': 0.5, 'similarity': 0.0}
        
        # 计算参数相似度
        best_record = None
        best_similarity = 0.0
        
        for record in relevant_records:
            record_vector = self.params_to_vector(record['parameters'], env_type)
            similarity = self.calculate_similarity(param_vector, record_vector)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_record = record
        
        if best_record:
            return {
                'performance_score': best_record['performance_score'],
                'similarity': best_similarity,
                'record_timestamp': best_record['timestamp']
            }
        else:
            return {'performance_score': 0.5, 'similarity': 0.0}

    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算参数向量相似度"""
        # 使用余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(similarity, 0.0)  # 确保非负

    def params_to_vector(self, params: Dict, env_type: str) -> np.ndarray:
        """将参数转换为向量用于相似度计算"""
        # 这里应该与OptimizationAgent中的方法保持一致
        # 简化版本
        return np.array(list(params.values()))
