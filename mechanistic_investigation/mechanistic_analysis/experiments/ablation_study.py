"""
ablation_study.py - Causal ablation study for attention heads
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

class AblationAnalyzer(BaseAnalyzer):
    """Perform causal ablation studies on attention heads"""
    
    def ablation_study_controlled(
        self,
        heads_to_test: List[Tuple[int, int]],
        n_control_heads: int = 5
    ) -> Dict:
        """
        Perform ablation study. Tests both target heads and random control heads.
        """
        logger.info("Starting controlled ablation study")
        
        # Select control heads randomly
        all_heads = [
            (l, h) 
            for l in range(self.architecture_info["n_layers"])
            for h in range(self.architecture_info["n_heads"])
        ]
        control_heads = []
        for head in all_heads:
            if head not in heads_to_test and len(control_heads) < n_control_heads:
                control_heads.append(head)
        
        test_suite = [
            ("Calculate: 5\nAnswer:", "5"),
            ("Calculate: 2 + 2\nAnswer:", "4"),
            ("Calculate: 23 + 45\nAnswer:", "68"),
            ("Calculate: 100 - 37\nAnswer:", "63"),
            ("Calculate: 8 * 7\nAnswer:", "56")
        ]
        
        paddings = [0, 50, 100]
        
        ablation_results = {
            "target_heads": [],
            "control_heads": [],
            "baseline_metrics": []
        }
        
        for padding_len in paddings:
            padding = "." * padding_len if padding_len > 0 else ""
            
            for prompt, expected in tqdm(test_suite, desc=f"Padding {padding_len}"):
                full_prompt = padding + prompt
                
                baseline_metrics = self._measure_generation_metrics(
                    full_prompt, 
                    expected,
                    head_mask=None,
                    n_runs=self.config.n_replication
                )
                ablation_results["baseline_metrics"].append(baseline_metrics)
                
                for layer, head in heads_to_test[:10]:
                    mask = self._create_head_mask(layer, head)
                    ablated_metrics = self._measure_generation_metrics(
                        full_prompt,
                        expected,
                        head_mask=mask,
                        n_runs=self.config.n_replication
                    )
                    
                    t_stat, p_value = stats.ttest_rel(
                        baseline_metrics["response_lengths"],
                        ablated_metrics["response_lengths"]
                    )
                    
                    effect_size = (
                        np.mean(baseline_metrics["response_lengths"]) - 
                        np.mean(ablated_metrics["response_lengths"])
                    ) / (np.std(baseline_metrics["response_lengths"]) + 1e-10)
                    
                    ablation_results["target_heads"].append({
                        "layer": layer,
                        "head": head,
                        "padding": padding_len,
                        "prompt": prompt[:20],
                        "baseline_mean": np.mean(baseline_metrics["response_lengths"]),
                        "ablated_mean": np.mean(ablated_metrics["response_lengths"]),
                        "effect_size": effect_size,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    })
                
                for layer, head in control_heads:
                    mask = self._create_head_mask(layer, head)
                    control_metrics = self._measure_generation_metrics(
                        full_prompt,
                        expected, 
                        head_mask=mask,
                        n_runs=3
                    )
                    
                    ablation_results["control_heads"].append({
                        "layer": layer,
                        "head": head,
                        "padding": padding_len,
                        "baseline_mean": np.mean(baseline_metrics["response_lengths"]),
                        "control_mean": np.mean(control_metrics["response_lengths"])
                    })
        
        causal_heads = self._identify_causal_heads(ablation_results)
        
        return {
            "ablation_results": ablation_results,
            "causal_heads": causal_heads,
            "control_comparison": self._compare_target_vs_control(ablation_results)
        }
    
    def _identify_causal_heads(self, ablation_results: Dict) -> List[MechanisticResult]:
        """Identify causally important heads from ablation results"""
        causal_heads = []
        
        head_effects = {}
        for result in ablation_results["target_heads"]:
            key = (result["layer"], result["head"])
            if key not in head_effects:
                head_effects[key] = []
            head_effects[key].append(result)
        
        for (layer, head), effects in head_effects.items():
            all_effect_sizes = [e["effect_size"] for e in effects]
            all_p_values = [e["p_value"] for e in effects]
            
            p_values_array = np.array(all_p_values)
            chi2_stat = -2 * np.sum(np.log(p_values_array + 1e-10))
            combined_p_value = stats.chi2.sf(chi2_stat, 2 * len(all_p_values))
            
            mean_effect = np.mean(all_effect_sizes)
            ci_lower = np.percentile(all_effect_sizes, 2.5)
            ci_upper = np.percentile(all_effect_sizes, 97.5)
            
            result = MechanisticResult(
                component="attention_head",
                layer=layer,
                head=head,
                effect_size=mean_effect,
                p_value=combined_p_value,
                confidence_interval=(ci_lower, ci_upper),
                n_samples=len(effects),
                causal=combined_p_value < 0.05 and abs(mean_effect) > self.config.effect_size_threshold,
                raw_data=np.array(all_effect_sizes)
            )
            
            causal_heads.append(result)
        
        causal_heads.sort(key=lambda x: abs(x.effect_size), reverse=True)
        return causal_heads
    
    def _compare_target_vs_control(self, ablation_results: Dict) -> Dict:
        """Statistical comparison of target vs control ablations"""
        target_effects = [
            r["effect_size"] 
            for r in ablation_results["target_heads"]
            if "effect_size" in r
        ]
        
        control_effects = []
        for r in ablation_results["control_heads"]:
            if "baseline_mean" in r and "control_mean" in r:
                effect = (r["baseline_mean"] - r["control_mean"]) / (r["baseline_mean"] + 1e-10)
                control_effects.append(effect)
        
        if target_effects and control_effects:
            u_stat, p_value = stats.mannwhitneyu(
                np.abs(target_effects),
                np.abs(control_effects),
                alternative='greater'
            )
            
            return {
                "target_mean_effect": np.mean(np.abs(target_effects)),
                "control_mean_effect": np.mean(np.abs(control_effects)),
                "u_statistic": u_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "interpretation": "Target heads have significantly larger effects" if p_value < 0.05 else "No significant difference"
            }
        
        return {"error": "Insufficient data for comparison"}