import signal
import subprocess
import sys

import random
import queue
import operator
import pickle
import os
import ast
import time
import json
import math
import shutil
import openai
import fire
import functools
import traceback
from datetime import datetime
from collections import Counter
from multiprocessing import Process, Queue
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from deap import base, creator, gp, tools, algorithms

from sklearn.metrics import roc_auc_score, f1_score

from utils.utils import cprint
from chat_llm import llama_main
from config import SRConfig
from exceptions import (
    SRException,
    ExpressionError,
    ExpressionParseError,
    ConfigError,
    ProcessError,
    ResourceError,
    EvaluationError
)
from message import Message, MessageType, Suggestion

# 从环境变量获取 API_KEY
LLM_SERVER_URL = "https://api.siliconflow.cn/v1"
API_KEY = os.getenv('SR_API_KEY')

class Constants:
    """Class for storing constant values"""
    # Default paths
    DEFAULT_PATHS = {
        'output_dir': "sr_generation_special/",
        'metric_save_path': "a_4metric_result_1105/"
    }
    
    # Evolution parameters
    EVOLUTION = {
        'crossover_prob': 0.5,
        'mutation_prob': 0.3,
        'generation_step': 40,
        'ephemeral_range': (0, 40)
    }
    
    # Result directory names
    RESULT_DIRS = ['tn', 'tp', 'fn', 'fp']
    
    # Visualization colors
    COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

class GPOperators:
    """Class containing genetic programming operators"""
    
    @staticmethod
    def c_and(a, b):
        """Logical AND operator"""
        try:
            return (a >= 1) & (b >= 1)
        except Exception as e:
            print(f"[c_and]: {a} {b}")
            raise e

    @staticmethod
    def c_or(a, b):
        """Logical OR operator"""
        try:
            return (a >= 1) | (b >= 1)
        except Exception as e:
            print(f"[c_or]: {a} {b}")
            raise e

    @staticmethod
    def c_not(a):
        """Logical NOT operator"""
        return a >= 1

    @staticmethod
    def get_all_operators():
        """Get all available operators"""
        return {
            'and_': GPOperators.c_and,
            'or_': GPOperators.c_or,
            'not_': GPOperators.c_not,
            'gt': operator.gt,
            'lt': operator.lt,
            'eq': operator.eq
        }

class ExpressionToTreeConverter:
    """Class for converting natural language expressions to DEAP primitive trees"""
    
    # 类变量，确保全局唯一
    _global_ephemeral_counter = 0
    
    def __init__(self, pset: gp.PrimitiveSet, labels: List[str]):
        """初始化转换器"""
        self._validate_labels(labels)
        self.pset = pset
        # 直接使用索引映射
        self.labels = {item.strip().replace(' ', '_').lower(): index 
                      for index, item in enumerate(labels)}
        self.operator_map = {
            ast.And: 'and_',
            ast.Or: 'or_',
            ast.Not: 'not_',
            ast.Gt: 'gt',
            ast.Lt: 'lt',
            ast.Eq: 'eq'
        }
        self._validate_primitives()
        # 不再需要实例级别的计数器
        # self._ephemeral_counter = 0

    def _validate_labels(self, labels: List[str]):
        """验证标签有效性"""
        if not labels:
            raise ConfigError("Empty labels list")
        if len(labels) != len(set(labels)):
            raise ConfigError("Duplicate labels detected")

    def _validate_primitives(self) -> None:
        """Validate that all required primitives exist in the primitive set"""
        required_primitives = set(self.operator_map.values())
        missing = required_primitives - set(self.pset.mapping.keys())
        if missing:
            raise ConfigError(
                f"Missing required primitives in pset: {', '.join(missing)}"
            )

    def parse(self, expression: str) -> List[Any]:
        """
        Convert expression string to DEAP primitive tree
        """
        if not expression.strip():
            raise ExpressionParseError("Empty expression provided")
            
        try:
            expr_ast = ast.parse(expression, mode='eval').body
            return self._convert_node(expr_ast)
        except SyntaxError as e:
            raise ExpressionParseError(f"Syntax error in expression: {expression}") from e
            
    def _convert_node(self, node: ast.AST) -> List[Any]:
        """转换AST节点到原始树"""
        try:
            # 处理二元操作符
            if isinstance(node, ast.BinOp):
                left = self._convert_node(node.left)
                right = self._convert_node(node.right)
                op_type = type(node.op)
                if op_type not in self.operator_map:
                    raise ExpressionParseError(f"不支持的二元操作符: {op_type.__name__}")
                
                op_name = self.operator_map[op_type]
                try:
                    prim = self.pset.mapping[op_name]
                except KeyError:
                    cprint(f"找不到原始操作符: {op_name}", 'r')
                    raise ExpressionParseError(f"找不到原始操作符: {op_name}")
                return [prim, left, right]
            
            # 处理一元操作符
            elif isinstance(node, ast.UnaryOp):
                operand = self._convert_node(node.operand)
                if isinstance(node.op, ast.Not):
                    try:
                        prim = self.pset.mapping['not_']
                        return [prim, operand]
                    except KeyError:
                        raise ExpressionParseError("找不到 not_ 操作符")
                raise ExpressionParseError(f"不支持的一元操作符: {type(node.op).__name__}")
            
            # 处理函数调用
            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ExpressionParseError("不支持复杂的函数调用")
                
                func_name = node.func.id.lower()
                args = [self._convert_node(arg) for arg in node.args]
                
                try:
                    prim = self.pset.mapping[func_name]
                except KeyError:
                    cprint(f"找不到原始操作符: {func_name}", 'r')
                    raise ExpressionParseError(f"找不到原始操作符: {func_name}")
                return [prim] + args
            
            # 处理变量名
            elif isinstance(node, ast.Name):
                var_name = node.id.strip().replace(' ', '_').lower()
                if var_name not in self.labels:
                    cprint(f"未知变量: {var_name}", 'r')
                    raise ExpressionParseError(f"未知变量: {var_name}")
                
                # 直接使用变量名，不需要转换为ARG形式
                if var_name not in self.pset.arguments:
                    raise ExpressionParseError(f"变量未在原始集中定义: {var_name}")
                
                return self.pset.mapping[var_name]
            
            # 处理常量
            elif isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float)):
                    raise ExpressionParseError(
                        f"不支持的常量类型: {type(node.value).__name__}"
                    )
                return gp.Terminal(float(node.value), True, ret=float)
            
            # 处理比较操作
            elif isinstance(node, ast.Compare):
                if len(node.ops) != 1 or len(node.comparators) != 1:
                    raise ExpressionParseError("不支持复杂的比较操作")
                
                op_type = type(node.ops[0])
                if op_type not in self.operator_map:
                    raise ExpressionParseError(f"不支持的比较操作符: {op_type.__name__}")
                
                op_name = self.operator_map[op_type]
                try:
                    prim = self.pset.mapping[op_name]
                except KeyError:
                    raise ExpressionParseError(f"找不到原始操作符: {op_name}")
                
                left = self._convert_node(node.left)
                right = self._convert_node(node.comparators[0])
                return [prim, left, right]
            
            # 处理布尔操作
            elif isinstance(node, ast.BoolOp):
                op_type = type(node.op)
                if op_type not in self.operator_map:
                    raise ExpressionParseError(f"不支持的布尔操作符: {op_type.__name__}")
                
                op_name = self.operator_map[op_type]
                try:
                    prim = self.pset.mapping[op_name]
                except KeyError:
                    raise ExpressionParseError(f"找不到原始操作符: {op_name}")
                
                values = [self._convert_node(value) for value in node.values]
                return [prim] + values
            
            else:
                raise ExpressionParseError(f"不支持的节点类型: {type(node).__name__}")
            
        except Exception as e:
            if not isinstance(e, ExpressionParseError):
                cprint(f"处理节点时出错: {ast.dump(node)}", 'r')
                raise ExpressionParseError(f"节点处理失败: {str(e)}")
            raise

class DataProcessor:
    """Class for handling data processing operations"""
    
    @staticmethod
    def extract_targets(json_data):
        """Extract target categories from JSON data"""
        return [ann['category_name'] for ann in json_data if isinstance(ann, dict)] 

    @staticmethod
    def get_all_target_class(obj_results_dir):
        """Get all target classes from a directory"""
        all_targets_categories = set()
        for filename in os.listdir(obj_results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(obj_results_dir, filename), 'r') as f:
                    json_data = json.load(f)
                    targets = [ann['category_name'] for ann in json_data 
                             if isinstance(ann, dict)]
                    all_targets_categories.update(targets)
        return all_targets_categories

    @staticmethod
    def generate_search_space(json_files, targets_set, label_dict=None):
        """Generate search space from JSON files"""
        x = []
        y = []
        for json_data in json_files:
            targets_counters = []
            image_path = (json_data[-1] if isinstance(json_data[-1], str) 
                         else json_data[0]['image_name'])
            
            # Label processing
            if label_dict is None:
                is_positive = ('正样本' in image_path or 'positive' in image_path or 
                             'P' in os.path.basename(image_path) or 
                             'event_whx' in image_path)
                y.append(int(is_positive))
            else:
                image_path_parts = image_path.split('/')
                label_key = '/'.join(image_path_parts[-4:])
                y.append(label_dict[label_key])
            
            # Feature processing
            targets = DataProcessor.extract_targets(json_data)
            c = Counter(targets)
            c_dict = dict(c)
            targets_counters = [c_dict.get(target, 0) for target in targets_set]
            x.append(targets_counters)
            
        return x, y

    @staticmethod
    def process_json_data(json_data, targets_set):
        """Process JSON data and update target set"""
        targets = DataProcessor.extract_targets(json_data)
        targets_set.update(targets)

class DataLoader:
    """Class for loading and processing data files"""
    
    @staticmethod
    def load_json_files(directory, threshold, is_iou=False, iou_threshold=0.5, 
                       tt_ratio=0.1, search_scale=None):
        """Load and process JSON files from directory"""
        train_data_list = []
        val_data_list = []
        json_files_list = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('.json')
        ]
        
        # Split data into train and validation sets
        use_same_data = not (0 < tt_ratio < 1)
        
        ori_img_paths = []
        proc_img_paths = []
        all_data_list = []
        
        for idx, json_file in enumerate(json_files_list):
            data = DataLoader._process_single_file(
                json_file, threshold, is_iou, iou_threshold
            )
            
            if data is None:
                continue
                
            filter_data, img_path = data
            ori_img_paths.append(img_path)
            proc_img_paths.append(json_file[:-5])
            all_data_list.append(filter_data)
        
        if use_same_data:
            # 如果tt_ratio不在(0,1)区间，训练集和测试集使用相同的数据
            train_data_list = all_data_list
            val_data_list = all_data_list
        else:
            # 正常划分训练集和测试集
            num_train = int(len(all_data_list) * tt_ratio) if search_scale is None else search_scale
            train_indices = set(random.sample(range(len(all_data_list)), num_train))
            train_data_list = [all_data_list[i] for i in train_indices]
            val_data_list = [all_data_list[i] for i in range(len(all_data_list)) if i not in train_indices]
                
        return {
            "Train": train_data_list,
            "Val": val_data_list
        }, ori_img_paths, proc_img_paths
    
    @staticmethod
    def _process_single_file(json_file, threshold, is_iou, iou_threshold):
        """Process a single JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                
                img_path = (json_data[-1] if isinstance(json_data[-1], str)
                           else json_data[-1].get('image_name') if isinstance(json_data[-1], dict)
                           else None)
                
                if is_iou:
                    filter_data = DataLoader._apply_iou_filter(
                        json_data, threshold, iou_threshold
                    )
                    filter_data.append(img_path)
                else:
                    filter_data = json_data
                    
                return filter_data, img_path
                
        except Exception as e:
            print(f"Error processing file {json_file}: {e}")
            return None
    
    @staticmethod
    def _apply_iou_filter(json_data, threshold, iou_threshold):
        """Apply IOU-based filtering to annotations"""
        filtered_data = []
        processed_boxes = []
        
        for ann in json_data:
            if not isinstance(ann, dict) or ann['score'] < threshold:
                continue
                
            current_box = ann['bbox']
            if not DataLoader._check_box_overlap(current_box, processed_boxes, iou_threshold):
                filtered_data.append(ann)
                processed_boxes.append(current_box)
                
        return filtered_data
    
    @staticmethod
    def _check_box_overlap(current_box, processed_boxes, iou_threshold):
        """Check if current box overlaps with any processed boxes"""
        return any(
            GeometryUtils.compute_iou_xywh_coco(current_box, processed_box) > iou_threshold
            for processed_box in processed_boxes
        )


class GeometryUtils:
    """Utility class for geometric calculations"""
    
    @staticmethod
    def get_bbox_center(bbox):
        """Calculate center point of a bounding box"""
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        return x_center, y_center

    @staticmethod
    def compute_iou_xxyy(bbox1, bbox2):
        """Calculate IoU of two bounding boxes in xxyy format"""
        x1_max, y1_max = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
        x2_min, y2_min = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    @staticmethod
    def compute_iou_xywh_coco(bbox1, bbox2):
        """Calculate IoU of two bounding boxes in COCO format (xywh)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x1_max, y1_max = max(x1, x2), max(y1, y2)
        x2_min, y2_min = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        
        inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

class ExpressionEvaluator:
    """Class for evaluating expressions and calculating metrics"""
    
    @staticmethod
    def evaluate_performance(func, X, y):
        """Evaluate individual using AUROC score"""
        try:
            y_pred = [int(bool(func(*sample))) for sample in X]
            return roc_auc_score(y, y_pred)
        except Exception as e:
            print(f"Error evaluating performance: {e}")
            return -np.inf

    @staticmethod
    def evaluate_loss(individual, X, y, compile_func, alpha=0.01):
        """Evaluate expression loss using BCE and complexity penalty"""
        try:
            func = compile_func(individual)
            y_pred = []
            
            for sample in X:
                pred = func(*sample)
                prob = 1 / (1 + math.exp(-pred))  # sigmoid
                y_pred.append(prob)
            
            y = np.array(y)
            y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
            bce_loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            complexity_penalty = alpha * len(individual)
            
            return bce_loss + complexity_penalty,
        except Exception as e:
            print(f"Error evaluating loss: {e}")
            return float('inf'),
    
    @staticmethod
    def flatten_expression(expr):
        """Flatten nested expression list"""
        if not isinstance(expr, (list, tuple)):
            return [expr]
        flat_expr = []
        for item in expr:
            if isinstance(item, (list, tuple)):
                flat_expr.extend(ExpressionEvaluator.flatten_expression(item))
            else:
                flat_expr.append(item)
        return flat_expr

class PrimitiveSetBuilder:
    """Class for building primitive sets for genetic programming"""
    
    @staticmethod
    def build_pset(config: SRConfig) -> gp.PrimitiveSet:
        """Build primitive set with operators and arguments"""
        pset = gp.PrimitiveSet("MAIN", arity=len(config.data.labels))
        
        # Rename arguments
        arg_names = {f'ARG{i}': label.replace(' ', '_').lower() 
                    for i, label in enumerate(config.data.labels)}
        pset.renameArguments(**arg_names)
        
        # Add operators
        operators = GPOperators.get_all_operators()
        for name, op in operators.items():
            if name in ['and_', 'or_']:
                pset.addPrimitive(op, 2, name=name)
            elif name == 'not_':
                pset.addPrimitive(op, 1, name=name)
            else:
                pset.addPrimitive(op, 2)
                
        # # 添加算术运算符
        # arithmetic_operators = {
        #     'add': operator.add,
        #     'sub': operator.sub,
        #     'mul': operator.mul,
        #     'div': operator.truediv  # 使用 truediv 而不是 div
        # }
        # for name, func in arithmetic_operators.items():
        #     pset.addPrimitive(func, 2)
        
        # Add ephemeral constants
        pset.addEphemeralConstant(
            "const",
            functools.partial(random.uniform, *Constants.EVOLUTION['ephemeral_range'])
        )
        
        return pset

class VisualizationUtils:
    """Class for visualization utilities"""
    
    @staticmethod
    def draw_bbox(ax, bbox, color='r', label=None):
        """Draw a single bounding box on the axis"""
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Plot center point
        center_x, center_y = GeometryUtils.get_bbox_center(bbox)
        ax.plot(center_x, center_y, 'o', color=color)

        if label:
            ax.text(center_x, center_y, label, 
                   color=color, fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.6))

    @staticmethod
    def plot_pairing_results(pairing_results, img_size=(400, 400)):
        """Plot pairing results with connected center points"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.set_xlim(0, img_size[0])
        ax.set_ylim(0, img_size[1])
        ax.invert_yaxis()

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        color_idx = 0

        for pair_key, pairs in pairing_results.items():
            cat1_name, cat2_name = pair_key.split('-')
            color = colors[color_idx % len(colors)]
            color_idx += 1

            for pair in pairs:
                bbox1, bbox2 = pair['bbox1'], pair['bbox2']
                
                # Draw bboxes
                VisualizationUtils.draw_bbox(ax, bbox1, color=color, label=cat1_name)
                VisualizationUtils.draw_bbox(ax, bbox2, color=color, label=cat2_name)

                # Draw connection line
                center1 = GeometryUtils.get_bbox_center(bbox1)
                center2 = GeometryUtils.get_bbox_center(bbox2)
                ax.plot([center1[0], center2[0]], [center1[1], center2[1]], 
                       linestyle='--', color=color)

                # Add metrics text
                mid_x = (center1[0] + center2[0]) / 2
                mid_y = (center1[1] + center2[1]) / 2
                ax.text(mid_x, mid_y, 
                       f"IOU: {pair['iou']:.2f}, Angle: {pair['angle']:.2f}°",
                       fontsize=10, color=color, ha='center',
                       bbox=dict(facecolor='white', alpha=0.6))

        plt.show()

class GPRunner:
    """Class for running genetic programming optimization"""
    
    def __init__(self, config: SRConfig, pset: gp.PrimitiveSet):
        self.config = config
        self.pset = pset
        self.toolbox = None
        self.stats = None
        self.hof = None
        self.population = None  # 添加 population 属性
        self.llm_suggestions_history = []
        self._setup_toolbox()

    def _setup_toolbox(self):
        """Setup DEAP toolbox with genetic operators"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, 
                            pset=self.pset, min_=1, max_=self.config.gp.max_tree_height)
        self.toolbox.register("individual", tools.initIterate, 
                            creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, 
                            list, self.toolbox.individual)
        
        # 这里需要修改，将evaluate方法绑定到实例方法
        self.toolbox.register("evaluate", self._evaluate_individual)
        
        self.toolbox.register("select", tools.selTournament, 
                            tournsize=self.config.gp.select_tour_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0,
                            max_=self.config.gp.max_tree_height)
        self.toolbox.register("mutate", gp.mutUniform, 
                            expr=self.toolbox.expr_mut, pset=self.pset)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        self.hof = tools.HallOfFame(self.config.gp.hof_max_size)

        # 添加装饰器来限制树的高度
        self.toolbox.decorate("mate", 
                            gp.staticLimit(key=operator.attrgetter("height"), 
                                         max_value=self.config.gp.max_tree_height))
        self.toolbox.decorate("mutate", 
                            gp.staticLimit(key=operator.attrgetter("height"), 
                                         max_value=self.config.gp.max_tree_height))

    def _evaluate_individual(self, individual):
        """Evaluate a single individual"""
        try:
            # 编译表达式
            func = gp.compile(individual, self.pset)
            # 评估性能
            score = ExpressionEvaluator.evaluate_performance(func, self.X, self.y)
            return (score,)
        except Exception as e:
            raise EvaluationError(f"Individual evaluation failed: {e}")

    def run_evolution(self, X, y, generations, queue_snd=None, queue_recv=None):
        """Run evolution process with LLM interaction"""
        self.X = X
        self.y = y
        
        self.population = self.toolbox.population(n=self.config.gp.population_size)  # 使用类属性
        best_history = []
        
        # 进化参数
        llm_interval = self.config.llm.interaction_interval
        retry_limit = self.config.llm.max_retries
        
        for gen in range(0, generations, self.config.gp.generation_step):
            # 运行进化
            self.population, logbook = algorithms.eaMuPlusLambda(  # 更新类属性
                self.population, self.toolbox,
                mu=self.config.gp.select_tour_size,
                lambda_=self.config.gp.hof_max_size * 2,
                cxpb=self.config.gp.crossover_prob,
                mutpb=self.config.gp.mutation_prob,
                ngen=min(self.config.gp.generation_step, generations - gen),
                stats=self.stats,
                halloffame=self.hof,
                verbose=False
            )
            
            # 记录最佳个体
            current_best = self.hof[0]
            best_history.append({
                'generation': gen,
                'individual': str(current_best),
                'fitness': current_best.fitness.values[0]
            })
            print(f"Generation {gen}: Best Expression: {str(current_best)} | Fitness: {current_best.fitness.values[0]}")
            
            if queue_snd and queue_recv and (gen % llm_interval == 0):
                self._communicate_with_llm(gen, queue_snd, queue_recv, retry_limit)
                
        best_individual = self.hof[0]  # 保存最佳个体
        return gp.compile(best_individual, self.pset), str(best_individual)  # 返回编译后的函数和表达式字符串

    def _communicate_with_llm(self, gen, queue_snd, queue_recv, retry_limit=3):
        """与LLM进行交互"""
        # 获取当前种群中前k个最优个体
        top_k = self.config.llm.top_k_individuals
        top_individuals = []
        for ind in self.hof[:top_k]:
            top_individuals.append({
                "expression": str(ind),
                "fitness": ind.fitness.values[0]
            })
        
        # 获取上一次LLM建议的评估结果
        previous_suggestions = self.llm_suggestions_history[-1] if self.llm_suggestions_history else None
        
        # 构造消息
        message = Message(
            msg_type=MessageType.EVOLUTION_UPDATE,
            payload={
                "generation": gen,
                "top_individuals": top_individuals,
                "previous_suggestions": previous_suggestions,
                "labels": self.config.data.labels,
                "operators": list(GPOperators.get_all_operators().keys())
            }
        )
        
        retries = 0
        while retries < retry_limit:
            try:
                queue_snd.put(message.serialize())
                print(f"已发送第{gen}代数据给LLM，等待响应...")
                
                # response_data = queue_recv.get(timeout=self.config.llm.response_timeout)
                response_data = queue_recv.get()
                response = Message.deserialize(response_data)
                suggestions = response.get_suggestions()
                
                # 评估建议并整合到种群
                suggestion_results = self._process_suggestions(suggestions)
                self.llm_suggestions_history.append(suggestion_results)
                return
                
            except (queue.Empty, ValueError) as e:
                print(f"LLM交互失败 (重试 {retries+1}/{retry_limit}): {str(e)}")
                retries += 1
                time.sleep(1)

    def _process_suggestions(self, suggestions: List[Suggestion]) -> Dict:
        """处理LLM建议：评估并整合到种群"""
        results = []
        converter = ExpressionToTreeConverter(self.pset, self.config.data.labels)
        
        for suggestion in suggestions:
            try:
                # 解析表达式
                primitive_expr = converter.parse(suggestion.expression)
                flat_expr = ExpressionEvaluator.flatten_expression(primitive_expr)
                
                # 创建个体并检查高度
                expr_tree = gp.PrimitiveTree(flat_expr)
                if expr_tree.height > self.config.gp.max_tree_height:
                    raise ExpressionParseError(f"Expression tree height {expr_tree.height} exceeds maximum allowed height {self.config.gp.max_tree_height}")
                
                # 创建个体并评估
                individual = creator.Individual(expr_tree)
                individual.fitness.values = self.toolbox.evaluate(individual)
                
                # 更新名人堂
                self.hof.update([individual])
                
                # 替换种群中最差个体
                worst_idx = np.argmin([ind.fitness.values[0] for ind in self.population])
                self.population[worst_idx] = individual
                
                results.append({
                    "expression": suggestion.expression,
                    "reason": suggestion.reason,
                    "fitness": individual.fitness.values[0],
                    "status": "success"
                })
                print(f"成功整合LLM建议: {suggestion.expression}")
                
            except Exception as e:
                results.append({
                    "expression": suggestion.expression,
                    "reason": suggestion.reason,
                    "error": str(e).replace(suggestion.expression, ""),
                    "status": "failed"
                })
                print(f"整合LLM建议失败: {suggestion.expression}\n错误: {str(e).replace(suggestion.expression, '')}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "suggestions": results
        }

    @classmethod
    def run(cls, X, y, pset, config, output_fd, queue_snd: Queue = None, queue_recv: Queue = None):
        """Run genetic programming optimization"""
        try:
            runner = cls(config, pset)
            best_func, best_expr = runner.run_evolution(
                X, y, config.gp.num_generations, 
                queue_snd, queue_recv
            )
            
            return best_func, best_expr  # 返回函数和表达式字符串
            
        except Exception as e:
            print(f"Error in GP run: {e}")
            traceback.print_exc()
            return None, None
        finally:
            ResourceManager.cleanup_gp_resources(config)

class FileUtils:
    """Utility class for file operations"""
    
    @staticmethod
    def get_all_directories(path):
        """Get all directories in the given path"""
        directories = []
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                if ".ipynb" in dir_name:
                    continue
                dir_path = os.path.join(root, dir_name)
                directories.append(os.path.abspath(dir_path))
            break
        return directories

    @staticmethod
    def add_suffix_to_filename(file_path, suffix):
        """Add suffix to filename while preserving extension"""
        directory, filename = os.path.split(file_path)
        name, extension = os.path.splitext(filename)
        new_filename = f"{name}_{suffix}{extension}"
        return os.path.join(directory, new_filename)

    @staticmethod
    def save_checkpoint(hof, generation, filename="checkpoint.pkl"):
        """Save checkpoint to file"""
        with open(filename, 'wb') as f:
            pickle.dump((hof, generation), f)

    @staticmethod
    def load_checkpoint(filename="checkpoint.pkl"):
        """Load checkpoint from file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

class GPEvolutionManager:
    """Manager class for genetic programming evolution process"""
    
    def __init__(self, config, pset, log_manager=None):
        self.config = config
        self.pset = pset
        self.log_manager = log_manager or LogManager()
        self.gp_runner = GPRunner(config, pset)
        self.converter = None
        self.best_individual = None
        self.best_function = None
    
    def _flatten_expr(self, expr):
        """Flatten nested expression list"""
        return ExpressionEvaluator.flatten_expression(expr)
    
    def cleanup(self):
        """Cleanup resources"""
        ResourceManager.cleanup_gp_resources(self.config)

class ExperimentManager:
    """Manager class for running SR experiments"""
    
    def __init__(self, config: SRConfig):
        self.config = config
        self.cur_task_info = None
        self._setup_directories()
        
    def _setup_directories(self):
        """Setup necessary directories"""
        os.makedirs(self.config.paths.output_dir, exist_ok=True)
        os.makedirs(self.config.paths.metric_save_path, exist_ok=True)
    
    def _prepare_experiment(self):
        """Prepare single experiment"""
        task_name = self.cur_task_info.path.split('/')[-1]
        timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")

        is_iou_filter = "ape" in self.cur_task_info.path.lower()
        
        output_file = os.path.join(self.config.paths.output_dir, f"{task_name}{timestamp}.log")
        
        all_targets = DataProcessor.get_all_target_class(self.cur_task_info.path)
        return is_iou_filter, output_file, all_targets
    
    def run_experiment(self):
        """Run single experiment"""
        is_iou_filter, output_file, all_targets = self._prepare_experiment()
        
        result_list = []
        for threshold in tqdm(self.cur_task_info.thresholds):
            json_files, ori_paths, proc_paths = DataLoader.load_json_files(
                self.cur_task_info.path, threshold, is_iou_filter,
                tt_ratio=self.config.data.tt_ratio,
                search_scale=self.config.data.search_scale
            )
            
            train_data = json_files["Train"]
            val_data = json_files["Val"]
            
            x_train, y_train = DataProcessor.generate_search_space(train_data, all_targets)
            x_val, y_val = DataProcessor.generate_search_space(val_data, all_targets)
            
            result_list.append([threshold, [x_train, y_train], [x_val, y_val]])
            
        return output_file, ExperimentResult(
            result_list, list(all_targets), self.cur_task_info.prior_expressions, ori_paths, proc_paths
        )
    
    def process_experiment_results(self, output_file, results, queue_snd, queue_recv):
        """Process results from a single experiment"""
        # Update output file name
        output_file = FileUtils.add_suffix_to_filename(
            output_file, 
            f"gen{self.config.gp.num_generations}_th{self.config.gp.max_tree_height}"
        )
        
        # Update config for current experiment
        self.config.update(
            labels=results.processed_labels,
            hof_max_size=results.hof_size,
            opt_expr_list=results.opt_expr_list
        )
        
        # Setup primitive set and run optimization
        pset = PrimitiveSetBuilder.build_pset(self.config)
        
        with open(output_file, 'w') as output_fd:
            self._run_threshold_experiments(
                results.result_list, self.config, pset, output_fd,
                queue_snd, queue_recv, results.proc_paths
            )
    
    def run_all_experiments(self, queue_snd=None, queue_recv=None):
        """Run all experiments"""
        all_results = {}
        tasks_info = self.config.task_settings.task_list
        for task_info in tasks_info:
            self.cur_task_info = task_info
            print(f"\n{'='*40}")
            print(f"开始处理任务目录: {self.cur_task_info.path}")
            print(f"使用预定义表达式列表: {self.cur_task_info.prior_expressions or '无'}")
            print(f"总任务数: {len(tasks_info)}")
            print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*40}\n")
            try:
                output_file, results = self.run_experiment()
                all_results[output_file] = results
                
                # Process results immediately
                self.process_experiment_results(
                    output_file, results, queue_snd, queue_recv
                )
                
            except Exception as e:
                print(f"Error running experiment for {task_dir}: {e}")
                traceback.print_exc()
                
        return all_results

    def _run_threshold_experiments(self, result_list, config, pset, output_fd, 
                                 queue_snd=None, queue_recv=None, proc_paths=None):
        """运行阈值实验"""
        # 发送初始化消息
        if queue_snd and queue_recv:
            init_msg = Message(
                msg_type=MessageType.INIT,
                payload={
                    "labels": config.data.labels,
                    "operators": list(GPOperators.get_all_operators().keys())
                }
            )
            queue_snd.put(init_msg.serialize())
        
        output_fd.write(f"GP Configuration:\n{config.gp}\n")
        output_fd.write(f"Data Configuration:\n{config.data}\n")
        
        # 添加阈值实验前的说明
        print(f"\n{'='*40}")
        print(f"开始阈值实验")
        print(f"总阈值数量: {len(result_list)}")
        print(f"当前配置:")
        print(f"- 遗传代数: {config.gp.num_generations}")
        print(f"- 种群大小: {config.gp.population_size}")
        print(f"- 最大树高度: {config.gp.max_tree_height}")
        print(f"{'='*40}\n")
        
        for threshold, [X_train, y_train], [X_test, y_test] in tqdm(result_list):
            output_fd.write(f"-------------threshold:{threshold}-------------\n")
            output_fd.write(f"The size of search space is {len(X_train)} and the size of test space is {len(X_test)}!\n")
            
            if queue_snd and queue_recv:
                threshold_msg = Message(
                    msg_type=MessageType.THRESHOLD_START,
                    payload={
                        "threshold": threshold,
                        "train_size": len(X_train),
                        "test_size": len(X_test)
                    }
                )
                queue_snd.put(threshold_msg.serialize())
                
            # 运行GP时传入队列
            best_func, best_expr_str = GPRunner.run(
                X_train, y_train, pset, config, output_fd, queue_snd, queue_recv
            )
            
            # 评估结果
            y_pred = []
            for idx, sample in enumerate(X_test):
                pred = best_func(*sample)
                y_pred.append(int(bool(pred)))
                
                # 如果需要重排列结果
                if config.is_rearrange_result and threshold == 0.09:
                    ResultHandler.handle_result_rearrangement(
                        config=config,
                        threshold=threshold,
                        pred=bool(pred),
                        true_label=y_test[idx],
                        img_path=proc_paths[idx]
                    )
                
            # Calculate and log metrics with best expression
            metrics = MetricsCalculator.calculate_metrics(y_test, y_pred)
            MetricsLogger.log_metrics(
                output_fd, 
                metrics, 
                threshold,
                best_expr=best_expr_str  # 传递表达式字符串而不是函数
            )
            
            # 添加每个阈值后的总结
            print(f"完成阈值 {threshold}：")
            print(f"Best Expression: {str(best_expr_str)}")
            print(f"AUROC: {metrics['auroc']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"耗时: {datetime.now().strftime('%H:%M:%S')}\n")
            output_fd.flush()

class ResultHandler:
    """处理实验结果的工具类"""
    
    RESULT_DIRS = ['tp', 'tn', 'fp', 'fn']
    
    @staticmethod
    def handle_result_rearrangement(config: SRConfig, threshold: float, pred: bool, 
                                  true_label: int, img_path: str) -> None:
        """
        根据预测结果和真实标签重新整理图片
        
        Args:
            config: 配置对象
            threshold: 当前阈值
            pred: 预测结果
            true_label: 真实标签
            img_path: 图片路径
        """
        try:
            # 构建目标路径
            timestamp = datetime.now().strftime("%Y%m%d_%H.%M.%S")
            base_path = os.path.join(
                config.paths.metric_save_path,
                f"_{threshold}_gen{config.gp.num_generations}_th{config.gp.max_tree_height}_{timestamp}"
            )
            
            # 确定结果类型
            result_type = ResultHandler._get_result_type(pred, true_label)
            result_dir = os.path.join(base_path, result_type)
            
            # 创建目录
            os.makedirs(result_dir, exist_ok=True)
            
            # 复制图片
            img_name = os.path.basename(img_path)
            dst_path = os.path.join(result_dir, img_name)
            shutil.copy(img_path, dst_path)
            
        except Exception as e:
            raise ProcessError(f"结果重排列失败: {str(e)}")
    
    @staticmethod
    def _get_result_type(pred: bool, true_label: int) -> str:
        """
        根据预测结果和真实标签确定结果类型
        
        Returns:
            'tp', 'tn', 'fp', 或 'fn'
        """
        if pred:
            return 'tp' if true_label == 1 else 'fp'
        else:
            return 'tn' if true_label == 0 else 'fn'

class MetricsCalculator:
    """Class for calculating various metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate multiple metrics for evaluation"""
        try:
            auc_score = roc_auc_score(y_true, y_pred)
            f1_score_val = f1_score(y_true, y_pred)
            
            return {
                'auroc': auc_score,
                'f1': f1_score_val
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'auroc': 0.0,
                'f1': 0.0
            }

class MetricsLogger:
    """Class for logging metrics and results"""
    
    @staticmethod
    def log_metrics(file_handler, metrics, threshold, best_expr=None):
        """Log metrics to file"""
        file_handler.write(f"\nMetrics at threshold {threshold}:\n")
        
        if best_expr is not None:
            file_handler.write(f"Best Expression: {str(best_expr)}\n")
        
        for metric_name, value in metrics.items():
            file_handler.write(f"{metric_name.upper()}: {value:.4f}\n")
        file_handler.write("\n")

class ExperimentResult:
    """Class for storing experiment results"""
    
    def __init__(self, result_list, labels, opt_expr_list, ori_paths, proc_paths):
        self.result_list = result_list
        self.labels = labels
        self.opt_expr_list = opt_expr_list
        self.ori_paths = ori_paths
        self.proc_paths = proc_paths
        
    @property
    def processed_labels(self):
        """Get processed labels"""
        return [label.replace('-', '_').lower() for label in self.labels]
    
    @property
    def hof_size(self):
        """Calculate hall of fame size"""
        return len(self.labels) + 10

class ResourceManager:
    """资源管理类"""
    
    @staticmethod
    def monitor_resources() -> dict:
        """
        监控系统资源使用情况
        返回:
            dict: 包含内存(MB)、CPU利用率(%)、线程数的字典
        """
        import psutil
        process = psutil.Process()
        return {
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=0.1),
            'threads': process.num_threads()
        }

    @staticmethod
    def cleanup_temp_files(config: SRConfig):
        """清理临时文件"""
        try:
            temp_dir = config.temp_dir
            if not os.path.exists(temp_dir):
                return
                
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.tmp'):
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        print(f"清理临时文件: {file_path}")
        except Exception as e:
            raise ResourceError(f"临时文件清理失败: {e}")

    @staticmethod
    def cleanup_gp_resources(config: SRConfig) -> None:
        """清理遗传编程资源"""
        try:
            # 清理 creator 类
            creator_classes = ["FitnessMin", "FitnessMax", "Individual"]
            for class_name in creator_classes:
                if hasattr(creator, class_name):
                    delattr(creator, class_name)
            
            # 清理 toolbox
            if hasattr(config.gp, 'toolbox'):
                toolbox = config.gp.toolbox
                toolbox_methods = [
                    "expr", "individual", "population", "evaluate",
                    "select", "mate", "expr_mut", "mutate"
                ]
                for method in toolbox_methods:
                    if hasattr(toolbox, method):
                        toolbox.unregister(method)
            
            # 清理种群和统计信息
            if hasattr(config.gp, 'population'):
                del config.gp.population[:]
            if hasattr(config.gp, 'stats'):
                del config.gp.stats
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
        except Exception as e:
            raise ResourceError(f"清理GP资源失败: {e}")

    @classmethod
    def full_cleanup(cls, config: SRConfig):
        """
        执行完整资源清理流程
        步骤:
        1. 清理GP资源
        2. 清理临时文件
        3. 强制垃圾回收
        """
        try:
            # 清理遗传编程资源
            cls.cleanup_gp_resources(config)
            
            # 清理临时文件
            cls.cleanup_temp_files(config)
            
            # 强制垃圾回收
            import gc
            gc.collect()

            # 打印资源报告
            resources = cls.monitor_resources()
            print(f"清理后资源状态 | 内存: {resources['memory_mb']:.2f}MB | "
                  f"CPU: {resources['cpu_percent']}% | "
                  f"线程: {resources['threads']}")
                  
        except Exception as e:
            raise ResourceError(f"完整清理失败: {e}") from e

class LogManager:
    """Class for managing logging operations"""
    
    def __init__(self, log_file=None, debug=False):
        """Initialize log manager"""
        self.log_file = log_file
        self.debug = debug
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.log_file:
            self.file_handler = open(self.log_file, 'w')
        else:
            self.file_handler = None
    
    def log(self, message, level='info'):
        """Log a message with specified level"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level.upper()}] {message}"
        
        # Print to console
        if level == 'error':
            cprint(formatted_message, 'r')
        elif level == 'warning':
            cprint(formatted_message, 'y')
        elif level == 'debug' and self.debug:
            cprint(formatted_message, 'b')
        else:
            print(formatted_message)
            
        # Write to file
        if self.file_handler:
            self.file_handler.write(f"{formatted_message}\n")
            self.file_handler.flush()
    
    def debug(self, message):
        """Log debug message"""
        if self.debug:
            self.log(message, 'debug')
    
    def error(self, message, exc_info=None):
        """Log error message"""
        if exc_info:
            message = f"{message}\n{traceback.format_exc()}"
        self.log(message, 'error')
    
    def close(self):
        """Close log file"""
        if self.file_handler:
            self.file_handler.close()

class Utils:
    """Utility class for common operations"""
    
    @staticmethod
    def ensure_directory(path):
        """Ensure directory exists"""
        os.makedirs(path, exist_ok=True)
        return path
    
    @staticmethod
    def get_timestamp(format="%Y%m%d_%H%M%S"):
        """Get formatted timestamp"""
        return datetime.now().strftime(format)
    
    @staticmethod
    def safe_file_operation(func):
        """Decorator for safe file operations"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except IOError as e:
                raise SRException(f"File operation failed: {str(e)}")
        return wrapper

class ProcessManager:
    """Class for managing multiprocessing"""
    
    def __init__(self, queue_size=100):
        self.question_queue = Queue(maxsize=queue_size)
        self.answer_queue = Queue(maxsize=queue_size)
        self.processes = []
    
    def add_process(self, target, daemon=True, args=()):
        """Add new process"""
        process = Process(target=target, args=args, daemon=daemon)
        self.processes.append(process)
    
    def start_all(self):
        """Start all processes"""
        for process in self.processes:
            process.start()
    
    def join_all(self):
        """Wait for all processes to complete"""
        for process in self.processes:
            process.join()
    
    def cleanup(self):
        """Cleanup process resources"""
        for process in self.processes:
            if process.is_alive():
                # Unix系统使用进程组终止
                if hasattr(os, 'killpg'):
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                # Windows系统使用taskkill
                else:
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                 capture_output=True)
                process.terminate()
        self.processes.clear()

class ExperimentRunner:
    """Class for running experiments"""
    
    def __init__(self, config: SRConfig, log_manager: LogManager = None, enable_llm: bool = True):
        self.config = config
        self.log_manager = log_manager or LogManager()
        self.experiment_manager = ExperimentManager(config)
        self._exit_sent = False  # 添加标志
        self.enable_llm = enable_llm
        
    def run(self, queue_snd: Queue = None, queue_recv: Queue = None):
        """Run all experiments"""
        try:
            # 确保所需目录存在
            self.config.ensure_directories()
            
            start_time = time.process_time()
            
            # Run experiments
            self.experiment_manager.run_all_experiments(
                queue_snd if self.enable_llm else None, 
                queue_recv if self.enable_llm else None
            )
            
            # Log completion
            total_time = time.process_time() - start_time
            self.log_manager.log(f"Total processing time: {total_time:.2f} seconds")
            
        except Exception as e:
            self.log_manager.error("Error in experiment execution", exc_info=True)
            
        finally:
            self._send_exit_message(queue_snd)

    def _send_exit_message(self, queue_snd):
        if not self._exit_sent and queue_snd and self.enable_llm:
            exit_msg = Message(
                msg_type=MessageType.COMMAND,
                payload={"command": "exit"}
            )
            queue_snd.put(exit_msg.serialize())
            self._exit_sent = True

    @classmethod
    def start_sr_generation(cls, queue_snd: Queue = None, queue_recv: Queue = None, enable_llm: bool = True, config_path: str = "config/default_config.yaml"):
        """Start SR generation process"""
        try:
            # Load configuration
            config = SRConfig.from_yaml(config_path)
            
            # Initialize managers
            log_manager = LogManager(debug=config.debug)
            runner = cls(config, log_manager, enable_llm)
            
            # Run experiments
            runner.run(queue_snd, queue_recv)
            
        except Exception as e:
            print(f"Error in SR generation: {e}")
            traceback.print_exc()
        finally:
            if queue_snd and enable_llm:
                exit_msg = Message(
                    msg_type=MessageType.COMMAND,
                    payload={"command": "exit"}
                )
                queue_snd.put(exit_msg.serialize())

class MessageHandler:
    """消息处理工具类"""
    
    @staticmethod
    def create_evolution_message(gen: int, best_individual, config) -> Message:
        return Message(
            msg_type=MessageType.EVOLUTION_UPDATE,
            payload={
                "generation": gen,
                "best_expression": str(best_individual),
                "performance": best_individual.fitness.values[0],
                "labels": config.data.labels,
                "operators": list(GPOperators.get_all_operators().keys()),
            }
        )
    
    
    @staticmethod
    def parse_suggestion(response: Message) -> list:
        if response.msg_type != MessageType.SUGGESTION:
            raise ProcessError(f"无效的响应类型: {response.msg_type}")
            
        return [
            {
                "expression": s['expression'],
                "reason": s['reason'],
                "priority": s.get('priority', 1)
            }
            for s in response.payload['suggestions']
        ]

def main(llm_client: openai.OpenAI = None, enable_llm: bool = True, config_path: str = "config/default_config.yaml"):
    """Main entry point with signal handling"""
    def signal_handler(sig, frame):
        print(f"\n捕获信号 {sig}，执行清理...")
        if process_manager:
            process_manager.cleanup()
        sys.exit(128 + sig)  # 遵循Unix退出码规范
    
    # 注册所有可捕获信号
    catchable_sigs = [
        signal.SIGINT,   # Ctrl+C (2)
        signal.SIGTERM,  # 默认终止信号 (15)
        signal.SIGHUP,   # 终端断开 (1)
        signal.SIGQUIT,  # Ctrl+\ (3)
        signal.SIGABRT   # 异常中止 (6)
    ]
    
    for sig in catchable_sigs:
        try:
            signal.signal(sig, signal_handler)
        except (ValueError, OSError, AttributeError):
            pass  # 处理平台不支持某些信号的情况
    
    log_manager = None
    process_manager = None
    
    try:
        if enable_llm:
            # 启用LLM功能时，初始化进程管理器和消息队列
            process_manager = ProcessManager()
            
            # 添加LLM进程
            if llm_client:
                process_manager.add_process(
                    llama_main,
                    args=(
                        process_manager.question_queue, 
                        process_manager.answer_queue,
                        llm_client
                    )
                )
            
            # 添加SR生成进程
            process_manager.add_process(
                ExperimentRunner.start_sr_generation,
                daemon=True,
                args=(process_manager.question_queue, 
                      process_manager.answer_queue,
                      enable_llm, config_path)
            )
            
            # 运行进程
            process_manager.start_all()
            process_manager.join_all()
        else:
            # 不启用LLM功能时，直接运行SR生成
            ExperimentRunner.start_sr_generation(enable_llm=enable_llm, config_path=config_path)
        
    except (ConfigError, ExpressionError) as e:
        if log_manager:
            log_manager.error(f"Configuration/Expression error: {e}")
        else:
            print(f"Configuration/Expression error: {e}")
    except Exception as e:
        if log_manager:
            log_manager.error(f"Unexpected error: {e}", exc_info=True)
        else:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
    finally:
        if process_manager:
            process_manager.cleanup()
        if log_manager:
            log_manager.close()

def cli_main(enable_llm=True, config_path="config/default_config.yaml"):
    """Command line interface for SR generation"""
    if enable_llm and not API_KEY:
        raise ConfigError("启用 LLM 功能时需要设置 SR_API_KEY 环境变量")
        
    llm_client = openai.OpenAI(base_url=LLM_SERVER_URL, api_key=API_KEY) if enable_llm else None
    return main(llm_client=llm_client, enable_llm=enable_llm, config_path=config_path)

if __name__ == '__main__':
    fire.Fire(cli_main)