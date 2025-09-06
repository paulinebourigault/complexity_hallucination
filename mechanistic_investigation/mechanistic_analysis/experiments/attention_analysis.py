"""
attention_analysis.py - Attention pattern analysis for position sensitivity
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy import stats
import logging
from ..core.base_analyzer import BaseAnalyzer
from ..core.config import MechanisticResult

logger = logging.getLogger(__name__)

class AttentionAnalyzer(BaseAnalyzer):
    """Analyze attention patterns for position sensitivity"""
    
    def analyze_attention_patterns(self, n_bootstrap: int = 100) -> Dict:
        """
        Analyze attention patterns.
        Uses bootstrap for confidence intervals and multiple test corrections.
        """
        logger.info("Starting attention pattern analysis")
        
        test_prompts = [
            "Calculate: 23 + 45\nAnswer:",
            "Calculate: 5 + 3\nAnswer:", 
            "Calculate: 100 - 50\nAnswer:",
            "Calculate: 7 * 8\nAnswer:",
            "Calculate: 15 / 3\nAnswer:"
        ]
        
        position_paddings = [0, 25, 50, 75, 100]
        
        all_attention_data = []
        
        for prompt in tqdm(test_prompts, desc="Testing prompts"):
            for padding_len in position_paddings:
                padding = "." * padding_len if padding_len > 0 else ""
                full_prompt = padding + prompt
                
                if len(self.tokenizer.encode(full_prompt)) > self.config.max_sequence_length:
                    logger.warning(f"Skipping - prompt too long: {len(full_prompt)} chars")
                    continue
                
                inputs = self.tokenizer(
                    full_prompt, 
                    return_tensors="pt",
                    max_length=self.config.max_sequence_length,
                    truncation=True
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, output_attentions=True)
                
                if outputs.attentions:
                    attention_data = self._extract_attention_statistics(
                        outputs.attentions,
                        padding_len,
                        len(prompt)
                    )
                    all_attention_data.append(attention_data)
        
        # Statistical analysis with bootstrap
        position_sensitive_heads = self._identify_position_sensitive_heads_statistical(
            all_attention_data,
            n_bootstrap=n_bootstrap
        )
        
        # Multiple testing correction (Bonferroni)
        n_tests = self.architecture_info["n_layers"] * self.architecture_info["n_heads"]
        corrected_threshold = 0.05 / n_tests
        
        significant_heads = [
            head for head in position_sensitive_heads
            if head.p_value < corrected_threshold
        ]
        
        logger.info(f"Found {len(significant_heads)} statistically significant position-sensitive heads")
        
        return {
            "all_attention_data": all_attention_data,
            "position_sensitive_heads": position_sensitive_heads,
            "significant_heads": significant_heads,
            "corrected_threshold": corrected_threshold,
            "n_tests": n_tests
        }
    
    def _extract_attention_statistics(
        self, 
        attentions: Tuple[torch.Tensor], 
        padding_len: int,
        prompt_len: int
    ) -> Dict:
        """Extract statistics from attention tensors"""
        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]
        
        stats = {
            "padding_len": padding_len,
            "prompt_len": prompt_len,
            "layer_stats": []
        }
        
        for layer_idx, layer_attn in enumerate(attentions):
            layer_data = {
                "layer": layer_idx,
                "heads": []
            }
            
            for head_idx in range(n_heads):
                head_attn = layer_attn[0, head_idx]
                attn_probs = head_attn[-1, :]
                entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10)).item()
                
                if padding_len > 0:
                    padding_attention = attn_probs[:padding_len].sum().item()
                    content_attention = attn_probs[padding_len:].sum().item()
                    position_bias = padding_attention / (padding_attention + content_attention + 1e-10)
                else:
                    position_bias = 0.0
                
                sorted_attn = torch.sort(attn_probs, descending=True)[0]
                n = len(sorted_attn)
                index = torch.arange(1, n + 1).float().to(sorted_attn.device)
                gini = (2 * torch.sum(index * sorted_attn)) / (n * torch.sum(sorted_attn)) - (n + 1) / n
                
                head_stats = {
                    "head": head_idx,
                    "entropy": entropy,
                    "position_bias": position_bias,
                    "gini_coefficient": gini.item(),
                    "max_attention": attn_probs.max().item(),
                    "attention_std": attn_probs.std().item()
                }
                
                layer_data["heads"].append(head_stats)
            
            stats["layer_stats"].append(layer_data)
        
        return stats
    
    def _identify_position_sensitive_heads_statistical(
        self,
        attention_data: List[Dict],
        n_bootstrap: int = 100
    ) -> List[MechanisticResult]:
        """Identify position-sensitive heads"""
        results = []
        
        n_layers = self.architecture_info["n_layers"]
        n_heads = self.architecture_info["n_heads"]
        
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                entropies_by_position = {}
                
                for data in attention_data:
                    if layer_idx < len(data["layer_stats"]):
                        layer_data = data["layer_stats"][layer_idx]
                        if head_idx < len(layer_data["heads"]):
                            head_data = layer_data["heads"][head_idx]
                            padding = data["padding_len"]
                            
                            if padding not in entropies_by_position:
                                entropies_by_position[padding] = []
                            entropies_by_position[padding].append(head_data["entropy"])
                
                if len(entropies_by_position) < 2:
                    continue
                
                groups = list(entropies_by_position.values())
                if all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    all_values = np.concatenate(groups)
                    grand_mean = np.mean(all_values)
                    ss_between = sum(
                        len(g) * (np.mean(g) - grand_mean) ** 2 
                        for g in groups
                    )
                    ss_total = np.sum((all_values - grand_mean) ** 2)
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    
                    bootstrap_effects = []
                    for _ in range(n_bootstrap):
                        resampled_groups = [
                            np.random.choice(g, size=len(g), replace=True)
                            for g in groups
                        ]
                        resampled_all = np.concatenate(resampled_groups)
                        resampled_mean = np.mean(resampled_all)
                        resampled_ss_between = sum(
                            len(g) * (np.mean(g) - resampled_mean) ** 2
                            for g in resampled_groups
                        )
                        resampled_ss_total = np.sum((resampled_all - resampled_mean) ** 2)
                        if resampled_ss_total > 0:
                            bootstrap_effects.append(resampled_ss_between / resampled_ss_total)
                    
                    ci_lower = np.percentile(bootstrap_effects, 2.5)
                    ci_upper = np.percentile(bootstrap_effects, 97.5)
                    
                    result = MechanisticResult(
                        component="attention_head",
                        layer=layer_idx,
                        head=head_idx,
                        effect_size=eta_squared,
                        p_value=p_value,
                        confidence_interval=(ci_lower, ci_upper),
                        n_samples=sum(len(g) for g in groups),
                        causal=False,
                        raw_data=np.array(all_values)
                    )
                    
                    results.append(result)
        
        results.sort(key=lambda x: x.effect_size, reverse=True)
        return results