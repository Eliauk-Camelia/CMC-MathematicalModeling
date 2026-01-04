"""
遗传算法使用示例
展示如何使用高度解耦的遗传算法框架
"""

from genetic_algorithm import (
    GeneticAlgorithm,
    FitnessFunction,
    InitializationFunction,
    PenaltyFunction,
    DefaultInitialization,
    NoPenalty,
    TournamentSelection,
    RouletteWheelSelection,
    UniformCrossover,
    SinglePointCrossover,
    ArithmeticCrossover,
    GaussianMutation,
    UniformMutation
)
import random
import numpy as np


# ==================== 示例1: 简单的二次函数优化 ====================
class QuadraticFitness(FitnessFunction):
    """二次函数适应度: f(x, y) = -(x^2 + y^2)，求最小值（负号转为最大化）"""
    
    def evaluate(self, individual):
        x, y = individual
        return -(x**2 + y**2)


def example1_simple_optimization():
    print("=" * 60)
    print("示例1: 优化二次函数 f(x, y) = -(x^2 + y^2)")
    print("=" * 60)
    
    # 定义数据范围
    bounds = [(-5, 5), (-5, 5)]
    
    # 创建组件
    fitness_func = QuadraticFitness()
    init_func = DefaultInitialization(bounds, gene_type='float')
    
    # 创建遗传算法
    ga = GeneticAlgorithm(
        fitness_function=fitness_func,
        initialization_function=init_func,
        mutation_strategy=GaussianMutation(mutation_rate=0.1, mutation_strength=0.2, bounds=bounds),
        crossover_strategy=ArithmeticCrossover(alpha=0.5),
        selection_strategy=TournamentSelection(tournament_size=3),
        crossover_probability=0.8,
        elitism_count=2
    )
    
    # 运行算法
    best_individual, best_fitness = ga.run(
        generations=50,
        population_size=50
    )
    
    print(f"\n最优解: x = {best_individual[0]:.4f}, y = {best_individual[1]:.4f}")
    print(f"理论最优: x = 0, y = 0")
    print()


# ==================== 示例2: 自定义初始化函数 ====================
class CustomInitialization(InitializationFunction):
    """自定义初始化：在特定区域生成初始种群"""
    
    def initialize(self, population_size):
        population = []
        for _ in range(population_size):
            # 在特定区域生成个体
            x = random.uniform(2, 4)  # x在[2, 4]范围内
            y = random.uniform(-3, -1)  # y在[-3, -1]范围内
            population.append([x, y])
        return population


def example2_custom_initialization():
    print("=" * 60)
    print("示例2: 使用自定义初始化函数")
    print("=" * 60)
    
    bounds = [(-5, 5), (-5, 5)]
    fitness_func = QuadraticFitness()
    init_func = CustomInitialization()
    
    ga = GeneticAlgorithm(
        fitness_function=fitness_func,
        initialization_function=init_func,
        mutation_strategy=GaussianMutation(mutation_rate=0.15, mutation_strength=0.3, bounds=bounds),
        crossover_strategy=UniformCrossover(crossover_rate=0.5),
        elitism_count=1
    )
    
    best_individual, best_fitness = ga.run(
        generations=30,
        population_size=40
    )
    
    print(f"\n最优解: {best_individual}")
    print()


# ==================== 示例3: 自定义惩罚函数 ====================
class ConstraintPenalty(PenaltyFunction):
    """约束惩罚：x + y >= 1"""
    
    def apply(self, individual, fitness):
        x, y = individual
        violation = max(0, 1 - (x + y))  # 违反约束的程度
        penalty = -1000 * violation  # 惩罚系数
        return fitness + penalty


class ConstrainedFitness(FitnessFunction):
    """带约束的适应度函数"""
    
    def evaluate(self, individual):
        x, y = individual
        return -(x**2 + y**2)


def example3_custom_penalty():
    print("=" * 60)
    print("示例3: 使用自定义惩罚函数 (约束: x + y >= 1)")
    print("=" * 60)
    
    bounds = [(-5, 5), (-5, 5)]
    fitness_func = ConstrainedFitness()
    penalty_func = ConstraintPenalty()
    init_func = DefaultInitialization(bounds, gene_type='float')
    
    ga = GeneticAlgorithm(
        fitness_function=fitness_func,
        initialization_function=init_func,
        penalty_function=penalty_func,
        mutation_strategy=GaussianMutation(mutation_rate=0.1, mutation_strength=0.2, bounds=bounds),
        crossover_strategy=ArithmeticCrossover(),
        elitism_count=2
    )
    
    best_individual, best_fitness = ga.run(
        generations=50,
        population_size=50
    )
    
    x, y = best_individual
    print(f"\n最优解: x = {x:.4f}, y = {y:.4f}")
    print(f"约束检查 (x + y >= 1): {x + y:.4f} >= 1? {x + y >= 1}")
    print()


# ==================== 示例4: Rastrigin函数优化（多模态函数）====================
class RastriginFitness(FitnessFunction):
    """Rastrigin函数：经典的多模态优化问题"""
    
    def __init__(self, n_dimensions=2):
        self.n_dimensions = n_dimensions
    
    def evaluate(self, individual):
        A = 10
        n = len(individual)
        return -(A * n + sum(x**2 - A * np.cos(2 * np.pi * x) for x in individual))


def example4_rastrigin():
    print("=" * 60)
    print("示例4: Rastrigin函数优化（多模态优化问题）")
    print("=" * 60)
    
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    fitness_func = RastriginFitness(n_dimensions=2)
    init_func = DefaultInitialization(bounds, gene_type='float')
    
    ga = GeneticAlgorithm(
        fitness_function=fitness_func,
        initialization_function=init_func,
        selection_strategy=TournamentSelection(tournament_size=5),
        crossover_strategy=ArithmeticCrossover(alpha=0.6),
        mutation_strategy=GaussianMutation(mutation_rate=0.2, mutation_strength=0.3, bounds=bounds),
        crossover_probability=0.9,
        elitism_count=3
    )
    
    best_individual, best_fitness = ga.run(
        generations=100,
        population_size=100
    )
    
    print(f"\n最优解: {[f'{x:.4f}' for x in best_individual]}")
    print(f"理论最优: [0, 0]")
    print()


# ==================== 示例5: 完整自定义所有组件 ====================
class CustomFitness(FitnessFunction):
    """
    自定义适应度函数：优化3维复杂函数
    
    适应度函数：f(x, y, z) = -(x² + y² + z² + x*y + y*z)
    - 这是一个3维优化问题，目标是找到使函数值最小（即适应度最大）的点
    - 使用负号将最小化问题转换为最大化问题（适应度越大越好）
    - 函数包含二次项和交叉项，形成复杂的优化曲面
    - 理论最优解应该在原点附近（具体取决于交叉项的权重）
    """
    
    def evaluate(self, individual):
        """
        评估个体的适应度
        
        Args:
            individual: 个体，包含3个基因值 [x, y, z]
            
        Returns:
            float: 适应度值（负数，值越大表示函数值越小，即越好）
        """
        x, y, z = individual
        # 计算函数值并取负（因为遗传算法是最大化适应度，而我们要最小化函数值）
        return -(x**2 + y**2 + z**2 + x*y + y*z)


class CustomInit3D(InitializationFunction):
    """
    自定义3维初始化函数
    
    在3维空间 [-3, 3] × [-3, 3] × [-3, 3] 范围内均匀随机生成初始种群
    这种方式可以：
    - 覆盖整个搜索空间
    - 提供多样化的初始解
    - 避免初始种群过于集中在某个区域
    """
    
    def initialize(self, population_size):
        """
        初始化第一代种群
        
        Args:
            population_size: 种群大小，即需要生成的个体数量
            
        Returns:
            List[List[float]]: 初始种群列表，每个个体包含3个随机生成的基因值
        """
        # 为每个个体生成3个在 [-3, 3] 范围内的随机浮点数
        # 外层循环：生成 population_size 个个体
        # 内层循环：为每个个体生成3个基因值
        return [[random.uniform(-3, 3) for _ in range(3)] for _ in range(population_size)]


def example5_full_customization():
    """
    示例5：完全自定义所有组件（3维优化）
    
    本示例展示了如何使用框架的所有自定义功能：
    1. 自定义适应度函数（3维复杂函数优化）
    2. 自定义初始化函数（3维空间随机生成）
    3. 自定义选择策略（轮盘赌选择）
    4. 自定义交叉策略（单点交叉）
    5. 自定义变异策略（均匀变异）
    6. 完全控制算法参数（代数、种群大小等）
    """
    print("=" * 60)
    print("示例5: 完全自定义所有组件（3维优化）")
    print("=" * 60)
    
    # 定义每个基因的搜索范围：[x范围, y范围, z范围]
    # 每个基因的取值范围都是 [-3, 3]
    bounds = [(-3, 3), (-3, 3), (-3, 3)]
    
    # 创建自定义适应度函数实例
    fitness_func = CustomFitness()
    
    # 创建自定义初始化函数实例
    init_func = CustomInit3D()
    
    # 创建遗传算法实例，完全自定义所有组件
    ga = GeneticAlgorithm(
        # 必需参数：适应度函数（自定义的3维函数优化）
        fitness_function=fitness_func,
        
        # 必需参数：初始化函数（自定义的3维随机初始化）
        initialization_function=init_func,
        
        # 可选参数：选择策略 - 轮盘赌选择（按适应度比例选择）
        # 适应度越高的个体被选中的概率越大
        selection_strategy=RouletteWheelSelection(),
        
        # 可选参数：交叉策略 - 单点交叉
        # 在随机选择一个交叉点，交换两个父代在该点之后的所有基因
        crossover_strategy=SinglePointCrossover(),
        
        # 可选参数：变异策略 - 均匀变异
        # mutation_rate=0.15: 每个基因有15%的概率发生变异
        # bounds=bounds: 变异后的值会被限制在定义范围内
        mutation_strategy=UniformMutation(mutation_rate=0.15, bounds=bounds),
        
        # 交叉概率：75% 的概率执行交叉操作
        # 如果随机数 < 0.75，则对选中的父代进行交叉；否则直接复制父代
        crossover_probability=0.75,
        
        # 变异概率：10%（如果mutation_strategy有自己的概率，此参数可能被忽略）
        # 在本例中，UniformMutation已经定义了mutation_rate=0.15，所以实际使用0.15
        mutation_probability=0.1,
        
        # 精英保留数量：每代保留2个最优秀的个体直接进入下一代
        # 这可以确保算法不会丢失当前找到的最佳解
        elitism_count=2
    )
    
    # 运行遗传算法
    best_individual, best_fitness = ga.run(
        generations=60,      # 进化60代
        population_size=80   # 每代种群包含80个个体
    )
    
    # 输出结果：最优解（3维坐标）保留4位小数
    print(f"\n最优解: {[f'{x:.4f}' for x in best_individual]}")
    print()


if __name__ == "__main__":
    # 运行所有示例
    example1_simple_optimization()
    example2_custom_initialization()
    example3_custom_penalty()
    example4_rastrigin()
    example5_full_customization()

