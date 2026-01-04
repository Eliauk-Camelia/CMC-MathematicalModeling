"""
高度解耦的遗传算法框架
支持自定义适应度函数、初始化函数、惩罚函数、数据范围等
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Callable, Optional
import random
import numpy as np


class FitnessFunction(ABC):
    """适应度函数抽象基类"""
    
    @abstractmethod
    def evaluate(self, individual: Any) -> float:
        """
        评估个体的适应度
        
        Args:
            individual: 个体（可以是列表、数组等）
            
        Returns:
            适应度值（越大越好）
        """
        pass


class InitializationFunction(ABC):
    """初始化函数抽象基类"""
    
    @abstractmethod
    def initialize(self, population_size: int) -> List[Any]:
        """
        初始化第一代种群
        
        Args:
            population_size: 种群大小
            
        Returns:
            初始种群列表
        """
        pass


class PenaltyFunction(ABC):
    """惩罚函数抽象基类"""
    
    @abstractmethod
    def apply(self, individual: Any, fitness: float) -> float:
        """
        对个体应用惩罚函数
        
        Args:
            individual: 个体
            fitness: 原始适应度值
            
        Returns:
            应用惩罚后的适应度值
        """
        pass


class DefaultInitialization(InitializationFunction):
    """默认初始化函数 - 在指定范围内随机生成"""
    
    def __init__(self, bounds: List[Tuple[float, float]], gene_type: str = 'float'):
        """
        Args:
            bounds: 每个基因的范围列表，例如 [(min1, max1), (min2, max2), ...]
            gene_type: 基因类型，'float' 或 'int'
        """
        self.bounds = bounds
        self.gene_type = gene_type
    
    def initialize(self, population_size: int) -> List[List[float]]:
        population = []
        for _ in range(population_size):
            individual = []
            for min_val, max_val in self.bounds:
                if self.gene_type == 'int':
                    gene = random.randint(int(min_val), int(max_val))
                else:
                    gene = random.uniform(min_val, max_val)
                individual.append(gene)
            population.append(individual)
        return population


class NoPenalty(PenaltyFunction):
    """无惩罚函数（默认）"""
    
    def apply(self, individual: Any, fitness: float) -> float:
        return fitness


class SelectionStrategy(ABC):
    """选择策略抽象基类"""
    
    @abstractmethod
    def select(self, population: List[Any], fitnesses: List[float], num_parents: int) -> List[Any]:
        """
        从种群中选择父代
        
        Args:
            population: 种群
            fitnesses: 适应度列表
            num_parents: 需要选择的父代数量
            
        Returns:
            选中的父代列表
        """
        pass


class TournamentSelection(SelectionStrategy):
    """锦标赛选择"""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: List[Any], fitnesses: List[float], num_parents: int) -> List[Any]:
        parents = []
        for _ in range(num_parents):
            tournament_indices = random.sample(range(len(population)), 
                                             min(self.tournament_size, len(population)))
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            parents.append(population[winner_idx])
        return parents


class RouletteWheelSelection(SelectionStrategy):
    """轮盘赌选择"""
    
    def select(self, population: List[Any], fitnesses: List[float], num_parents: int) -> List[Any]:
        # 确保适应度为正值
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1 for f in fitnesses]
        
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.sample(population, num_parents)
        
        probabilities = [f / total_fitness for f in fitnesses]
        parents = []
        for _ in range(num_parents):
            idx = np.random.choice(len(population), p=probabilities)
            parents.append(population[idx])
        return parents


class CrossoverStrategy(ABC):
    """交叉策略抽象基类"""
    
    @abstractmethod
    def crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """
        对两个父代进行交叉操作
        
        Args:
            parent1: 父代1
            parent2: 父代2
            
        Returns:
            (子代1, 子代2)
        """
        pass


class UniformCrossover(CrossoverStrategy):
    """均匀交叉"""
    
    def __init__(self, crossover_rate: float = 0.5):
        self.crossover_rate = crossover_rate
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        child1, child2 = parent1.copy(), parent2.copy()
        for i in range(len(parent1)):
            if random.random() < self.crossover_rate:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2


class SinglePointCrossover(CrossoverStrategy):
    """单点交叉"""
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2


class ArithmeticCrossover(CrossoverStrategy):
    """算术交叉（适用于连续值）"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        child1 = [self.alpha * p1 + (1 - self.alpha) * p2 for p1, p2 in zip(parent1, parent2)]
        child2 = [self.alpha * p2 + (1 - self.alpha) * p1 for p1, p2 in zip(parent1, parent2)]
        return child1, child2


class MutationStrategy(ABC):
    """变异策略抽象基类"""
    
    @abstractmethod
    def mutate(self, individual: Any) -> Any:
        """
        对个体进行变异
        
        Args:
            individual: 个体
            
        Returns:
            变异后的个体
        """
        pass


class GaussianMutation(MutationStrategy):
    """高斯变异（适用于连续值）"""
    
    def __init__(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1, bounds: Optional[List[Tuple[float, float]]] = None):
        """
        Args:
            mutation_rate: 变异概率
            mutation_strength: 变异强度（标准差）
            bounds: 每个基因的范围，用于边界约束
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.bounds = bounds
    
    def mutate(self, individual: List[float]) -> List[float]:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += random.gauss(0, self.mutation_strength * abs(mutated[i]) if mutated[i] != 0 else self.mutation_strength)
                
                # 边界约束
                if self.bounds:
                    min_val, max_val = self.bounds[i]
                    mutated[i] = max(min_val, min(max_val, mutated[i]))
        
        return mutated


class UniformMutation(MutationStrategy):
    """均匀变异"""
    
    def __init__(self, mutation_rate: float = 0.1, bounds: List[Tuple[float, float]] = None):
        """
        Args:
            mutation_rate: 变异概率
            bounds: 每个基因的范围
        """
        self.mutation_rate = mutation_rate
        self.bounds = bounds
    
    def mutate(self, individual: List[float]) -> List[float]:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                if self.bounds:
                    min_val, max_val = self.bounds[i]
                    mutated[i] = random.uniform(min_val, max_val)
        
        return mutated


class GeneticAlgorithm:
    """高度解耦的遗传算法主类"""
    
    def __init__(
        self,
        fitness_function: FitnessFunction,
        initialization_function: InitializationFunction,
        penalty_function: Optional[PenaltyFunction] = None,
        selection_strategy: Optional[SelectionStrategy] = None,
        crossover_strategy: Optional[CrossoverStrategy] = None,
        mutation_strategy: Optional[MutationStrategy] = None,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.1,
        elitism_count: int = 1
    ):
        """
        初始化遗传算法
        
        Args:
            fitness_function: 适应度函数
            initialization_function: 初始化函数
            penalty_function: 惩罚函数（可选，默认无惩罚）
            selection_strategy: 选择策略（可选，默认锦标赛选择）
            crossover_strategy: 交叉策略（可选，默认均匀交叉）
            mutation_strategy: 变异策略（可选，需要提供bounds或使用默认）
            crossover_probability: 交叉概率
            mutation_probability: 变异概率（如果mutation_strategy有自己的概率，此参数可能被忽略）
            elitism_count: 精英保留数量
        """
        self.fitness_function = fitness_function
        self.initialization_function = initialization_function
        self.penalty_function = penalty_function or NoPenalty()
        self.selection_strategy = selection_strategy or TournamentSelection()
        self.crossover_strategy = crossover_strategy or UniformCrossover()
        self.mutation_strategy = mutation_strategy
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_count = elitism_count
        
        self.population = []
        self.fitness_history = []
        self.best_individual_history = []
    
    def _evaluate_population(self, population: List[Any]) -> List[float]:
        """评估整个种群的适应度"""
        fitnesses = []
        for individual in population:
            fitness = self.fitness_function.evaluate(individual)
            fitness = self.penalty_function.apply(individual, fitness)
            fitnesses.append(fitness)
        return fitnesses
    
    def run(
        self,
        generations: int,
        population_size: int,
        verbose: bool = True,
        callback: Optional[Callable[[int, List[Any], List[float]], None]] = None
    ) -> Tuple[Any, float]:
        """
        运行遗传算法
        
        Args:
            generations: 进化代数
            population_size: 每代种群大小
            verbose: 是否打印进度
            callback: 每代回调函数 callback(generation, population, fitnesses)
            
        Returns:
            (最佳个体, 最佳适应度)
        """
        # 初始化种群
        self.population = self.initialization_function.initialize(population_size)
        
        # 评估初始种群
        fitnesses = self._evaluate_population(self.population)
        
        # 记录历史
        best_idx = np.argmax(fitnesses)
        self.fitness_history = [max(fitnesses)]
        self.best_individual_history = [self.population[best_idx].copy() if hasattr(self.population[best_idx], 'copy') else self.population[best_idx]]
        
        if verbose:
            print(f"初始代: 最佳适应度 = {max(fitnesses):.6f}")
        
        # 进化循环
        for generation in range(1, generations + 1):
            # 精英保留
            elite_indices = np.argsort(fitnesses)[-self.elitism_count:][::-1]
            elite = [self.population[i].copy() if hasattr(self.population[i], 'copy') else self.population[i] 
                    for i in elite_indices]
            
            # 创建新种群
            new_population = elite.copy()
            
            # 生成新个体
            while len(new_population) < population_size:
                # 选择
                parents = self.selection_strategy.select(self.population, fitnesses, 2)
                
                # 交叉
                if random.random() < self.crossover_probability:
                    child1, child2 = self.crossover_strategy.crossover(parents[0], parents[1])
                else:
                    child1, child2 = parents[0], parents[1]
                
                # 变异
                if self.mutation_strategy:
                    child1 = self.mutation_strategy.mutate(child1)
                    child2 = self.mutation_strategy.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 确保种群大小正确
            self.population = new_population[:population_size]
            
            # 评估新种群
            fitnesses = self._evaluate_population(self.population)
            
            # 记录历史
            best_idx = np.argmax(fitnesses)
            self.fitness_history.append(max(fitnesses))
            self.best_individual_history.append(
                self.population[best_idx].copy() if hasattr(self.population[best_idx], 'copy') 
                else self.population[best_idx]
            )
            
            if verbose and generation % max(1, generations // 10) == 0:
                print(f"第 {generation} 代: 最佳适应度 = {max(fitnesses):.6f}")
            
            # 回调
            if callback:
                callback(generation, self.population, fitnesses)
        
        # 返回最佳个体
        best_idx = np.argmax(fitnesses)
        best_individual = self.population[best_idx]
        best_fitness = fitnesses[best_idx]
        
        if verbose:
            print(f"\n最终结果: 最佳适应度 = {best_fitness:.6f}")
            print(f"最佳个体: {best_individual}")
        
        return best_individual, best_fitness
    
    def get_history(self) -> Tuple[List[float], List[Any]]:
        """获取进化历史"""
        return self.fitness_history, self.best_individual_history
