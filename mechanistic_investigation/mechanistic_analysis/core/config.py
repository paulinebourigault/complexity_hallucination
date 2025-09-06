"""
config.py - Configuration and data structures for mechanistic analysis
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np

@dataclass
class ExperimentConfig:
    """Configuration for experiments with validation"""
    model_name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    n_replication: int = 10
    max_sequence_length: int = 512
    generation_max_tokens: int = 100
    temperature: float = 0.1
    confidence_level: float = 0.95
    effect_size_threshold: float = 0.2
    attention_analysis_layers: Optional[List[int]] = None
    ablation_n_samples: int = 20
    intervention_scales: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0])
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.n_replication >= 3, "Need at least 3 replications for statistical tests"
        assert 0 < self.confidence_level < 1, "Confidence level must be between 0 and 1"
        assert self.max_sequence_length > 0, "Max sequence length must be positive"
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

@dataclass
class MechanisticResult:
    """Container for mechanistic findings"""
    component: str
    layer: Optional[int]
    head: Optional[int]
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    n_samples: int
    causal: bool
    intervention_result: Optional[Dict] = None
    raw_data: Optional[np.ndarray] = None
    
    @property
    def is_significant(self) -> bool:
        """Check statistical significance"""
        return self.p_value < 0.05 and abs(self.effect_size) > 0.2