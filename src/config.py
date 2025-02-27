"""Configuration management for the SR system"""

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, fields
from exceptions import ConfigError

@dataclass
class TaskConfig:
    """Task configuration"""
    path: str = field(default_factory=str)
    prior_expressions: list = field(default_factory=list)
    thresholds: list = field(default_factory=list)

@dataclass
class TaskSettings:
    """任务全局设置"""
    default_thresholds: List[float] = field(
        default_factory=lambda: [i*0.02+0.01 for i in range(10)]
    )
    task_list: List[TaskConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TaskSettings':
        """从字典创建TaskSettings实例"""
        # 获取默认阈值
        default_thresholds = config_dict.get('default_thresholds')
        
        # 处理task_list，如果thresholds为空则使用default_thresholds
        task_list = []
        for task in config_dict.get('task_list', []):
            thresholds = task.get('thresholds', [])
            if len(thresholds) == 0:  # 判断是否为空列表
                task['thresholds'] = default_thresholds
            task_list.append(TaskConfig(**task))

        return cls(
            default_thresholds=default_thresholds,
            task_list=task_list
        )

@dataclass
class GPConfig:
    """Genetic Programming configuration"""
    num_generations: int = 500
    population_size: int = 50
    max_tree_height: int = 4
    select_tour_size: int = 4
    hof_max_size: int = 10
    crossover_prob: float = 0.5
    mutation_prob: float = 0.3
    generation_step: int = 40

@dataclass
class DataConfig:
    """Data processing configuration"""
    tt_ratio: float = 0.1
    search_scale: int = 200
    labels: list = field(default_factory=list)
    opt_expr_list: Optional[list] = None

@dataclass
class PathConfig:
    """Path configuration"""
    output_base_dir: str = "output/"
    _output_dir: str = "sr_generation_special/"
    _metric_save_path: str = "a_4metric_result/"
    
    def __post_init__(self):
        """在初始化后处理路径组合"""
        # 组合完整路径
        self.output_dir = os.path.join(self.output_base_dir, self._output_dir)
        self.metric_save_path = os.path.join(self.output_base_dir, self._metric_save_path)
        # 创建临时文件目录路径
        self.temp_dir = os.path.join(self.output_base_dir, 'temp')

@dataclass
class LLMConfig:
    """LLM交互配置"""
    enable_llm: bool = True
    interaction_interval: int = 20  # 交互间隔代数
    max_retries: int = 3
    top_k_individuals: int = 5 
    response_timeout: float = 60.0  # 秒

@dataclass
class SRConfig:
    """Main configuration class"""
    gp: GPConfig = field(default_factory=GPConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    task_settings: TaskSettings = field(default_factory=TaskSettings)
    is_rearrange_result: bool = False
    debug: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SRConfig':
        """Create configuration from dictionary"""
        gp_config = GPConfig(**config_dict.get('gp', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        path_config = PathConfig(**config_dict.get('paths', {}))
        task_settings = TaskSettings.from_dict(config_dict.get('tasks', {}))
        
        return cls(
            gp=gp_config,
            data=data_config,
            paths=path_config,
            task_settings=task_settings,
            is_rearrange_result=config_dict.get('is_rearrange_result', False),
            debug=config_dict.get('debug', False),
            llm=LLMConfig(**config_dict.get('llm', {}))
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SRConfig':
        """
        Load configuration from YAML file
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            SRConfig instance with loaded configuration
            
        Raises:
            ConfigError: If file not found or invalid YAML format
        """
        try:
            # 如果未提供配置文件路径，使用默认配置
            if not yaml_path:
                yaml_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.yaml')
            
            # 检查文件是否存在
            if not os.path.exists(yaml_path):
                raise ConfigError(f"配置文件不存在: {yaml_path}")
                
            # 读取YAML文件
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
                
            # 验证配置字典
            if not isinstance(config_dict, dict):
                raise ConfigError("配置文件格式错误：应为YAML字典格式")
                
            # 创建子配置对象
            gp_config = GPConfig(**config_dict.get('gp', {}))
            data_config = DataConfig(**config_dict.get('data', {}))
            path_config = PathConfig(**config_dict.get('paths', {}))
            task_settings = TaskSettings.from_dict(config_dict.get('tasks', {}))
            
            # 创建主配置对象
            config = cls(
                gp=gp_config,
                data=data_config,
                paths=path_config,
                task_settings=task_settings,
                is_rearrange_result=config_dict.get('is_rearrange_result', False),
                debug=config_dict.get('debug', False),
                llm=LLMConfig(**config_dict.get('llm', {}))
            )
            
            # 验证配置
            config.validate()
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML解析错误: {e}")
        except Exception as e:
            raise ConfigError(f"加载配置文件失败: {e}")
    
    def validate(self) -> None:
        """
        验证配置的有效性
        
        Raises:
            ConfigError: 如果配置无效
        """
        # 验证GP配置
        if self.gp.num_generations <= 0:
            raise ConfigError("num_generations 必须大于0")
        if self.gp.population_size <= 0:
            raise ConfigError("population_size 必须大于0")
        if self.gp.max_tree_height <= 0:
            raise ConfigError("max_tree_height 必须大于0")
        if not (0 <= self.gp.crossover_prob <= 1):
            raise ConfigError("crossover_prob 必须在0到1之间")
        if not (0 <= self.gp.mutation_prob <= 1):
            raise ConfigError("mutation_prob 必须在0到1之间")
            
        # 验证数据配置
        # if not (0 < self.data.tt_ratio < 1):
        #     raise ConfigError("tt_ratio 必须在0到1之间")
        if self.data.search_scale <= 0:
            raise ConfigError("search_scale 必须大于0")
            
        # 验证路径配置
        if not self.paths.output_base_dir:
            raise ConfigError("output_base_dir 不能为空")
        if not self.paths.output_dir:
            raise ConfigError("output_dir 不能为空")
        if not self.paths.metric_save_path:
            raise ConfigError("metric_save_path 不能为空")

    def update(self, **kwargs):
        """
        智能更新配置参数
        
        自动将参数更新到正确的子配置对象中
        """
        data_fields = set(f.name for f in fields(DataConfig))
        gp_fields = set(f.name for f in fields(GPConfig))
        paths_fields = set(f.name for f in fields(PathConfig))
        
        for key, value in kwargs.items():
            if key in data_fields:
                setattr(self.data, key, value)
            elif key in gp_fields:
                setattr(self.gp, key, value)
            elif key in paths_fields:
                setattr(self.paths, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ConfigError(f"Unknown configuration parameter: {key}")

    @property
    def output_dir(self) -> str:
        """Get full output directory path"""
        return os.path.join(self.paths.output_base_dir, self.paths._output_dir)

    @property
    def metric_save_path(self) -> str:
        """Get full metric save path"""
        return os.path.join(self.paths.output_base_dir, self.paths._metric_save_path)

    @property
    def temp_dir(self) -> str:
        """Get temporary directory path"""
        return os.path.join(self.paths.output_base_dir, 'temp')

    def ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metric_save_path, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def get_experiment_path(self, experiment_name: str) -> str:
        """Get path for specific experiment"""
        return os.path.join(self.output_dir, experiment_name)

    def get_metric_path(self, metric_name: str) -> str:
        """Get path for specific metric"""
        return os.path.join(self.metric_save_path, metric_name) 