"""
base_analyzer.py - Base analyzer class with model loading and architecture detection
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional
import logging
from .config import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAnalyzer:
    """Base class with model loading and architecture detection"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.architecture_info = {}
        self.hooks = {}
        self.activations = {}
        self.results_cache = {}
        self._load_and_validate_model()
        
    def _load_and_validate_model(self):
        """Load model"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.warning("No pad token found, using eos_token")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                device_map=self.config.device if self.config.device == "cuda" else None,
                output_hidden_states=True,
                output_attentions=True,
                trust_remote_code=False
            )
            
            if self.config.device == "cpu":
                self.model = self.model.to(self.config.device)
            self.model.eval()
            self._detect_architecture()
            self._validate_attention_output()
            
            logger.info(f"Model loaded successfully: {self.architecture_info}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _detect_architecture(self):
        """Detect model architecture and extract configuration"""
        model_class = type(self.model).__name__
        
        if "GPT2" in model_class:
            self.model_type = "gpt2"
            self.architecture_info = {
                "type": "gpt2",
                "n_layers": self.model.config.n_layer,
                "n_heads": self.model.config.n_head,
                "d_model": self.model.config.n_embd,
                "vocab_size": self.model.config.vocab_size,
                "max_position_embeddings": self.model.config.n_positions,
                "transformer_module": self.model.transformer,
                "embed_tokens": self.model.transformer.wte,
                "embed_positions": self.model.transformer.wpe,
                "layers": self.model.transformer.h,
                "lm_head": self.model.lm_head
            }
        elif "GPTNeoX" in model_class or "Pythia" in self.config.model_name.lower():
            self.model_type = "pythia"
            self.architecture_info = {
                "type": "pythia",
                "n_layers": self.model.config.num_hidden_layers,
                "n_heads": self.model.config.num_attention_heads,
                "d_model": self.model.config.hidden_size,
                "vocab_size": self.model.config.vocab_size,
                "max_position_embeddings": self.model.config.max_position_embeddings,
                "transformer_module": self.model.gpt_neox,
                "embed_tokens": self.model.gpt_neox.embed_in,
                "layers": self.model.gpt_neox.layers,
                "lm_head": self.model.embed_out
            }
        else:
            self.model_type = "unknown"
            config = self.model.config
            self.architecture_info = {
                "type": "unknown",
                "n_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12)),
                "n_heads": getattr(config, 'num_attention_heads', getattr(config, 'n_head', 12)),
                "d_model": getattr(config, 'hidden_size', getattr(config, 'n_embd', 768)),
                "vocab_size": config.vocab_size,
                "warning": "Unknown architecture - some features may not work"
            }
            logger.warning(f"Unknown architecture: {model_class}")
    
    def _validate_attention_output(self):
        """Validate that model actually outputs attention weights"""
        test_input = self.tokenizer("Test", return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(**test_input, output_attentions=True)
        
        if outputs.attentions is None:
            raise RuntimeError(f"Model {self.config.model_name} does not output attention weights")
        
        expected_shape = (
            1,
            self.architecture_info["n_heads"],
            test_input.input_ids.shape[1],
            test_input.input_ids.shape[1]
        )
        
        actual_shape = outputs.attentions[0].shape
        if actual_shape != expected_shape:
            logger.warning(f"Unexpected attention shape: {actual_shape} vs {expected_shape}")
    
    def _create_head_mask(self, layer: int, head: int) -> torch.Tensor:
        """Create a mask that zeros out specific attention head"""
        mask = torch.ones(
            self.architecture_info["n_layers"],
            self.architecture_info["n_heads"]
        ).to(self.config.device)
        mask[layer, head] = 0
        return mask
    
    def _measure_generation_metrics(
        self,
        prompt: str,
        expected: str,
        head_mask: Optional[torch.Tensor],
        n_runs: int
    ) -> Dict:
        """Measure generation metrics with replication"""
        response_lengths = []
        has_answers = []
        has_hallucinations = []
        
        for run in range(n_runs):
            torch.manual_seed(self.config.seed + run)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_sequence_length,
                truncation=True
            ).to(self.config.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.config.generation_max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    head_mask=head_mask,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response_only = response[len(prompt):]
            
            response_lengths.append(len(response_only))
            has_answers.append(expected in response_only)
            has_hallucinations.append(
                len(response_only) > 50 or 
                "Calculate:" in response_only[10:] or
                response_only.count('\n') > 3
            )
        
        return {
            "response_lengths": response_lengths,
            "has_answers": has_answers, 
            "has_hallucinations": has_hallucinations,
            "mean_length": np.mean(response_lengths),
            "std_length": np.std(response_lengths),
            "accuracy": np.mean(has_answers),
            "hallucination_rate": np.mean(has_hallucinations)
        }