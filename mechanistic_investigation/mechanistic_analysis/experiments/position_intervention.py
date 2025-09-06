"""
position_intervention.py - Position embedding intervention experiments
"""

import torch
import numpy as np
from typing import Dict, List
from scipy.optimize import curve_fit
import logging
from ..core.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class PositionInterventionAnalyzer(BaseAnalyzer):
    """Test direct interventions on position embeddings"""
    
    def position_embedding_intervention(self) -> Dict:
        """
        Direct intervention on position embeddings.
        """
        logger.info("Starting position embedding intervention")
        
        if "embed_positions" not in self.architecture_info:
            logger.warning("Position embeddings not accessible for this architecture")
            return {"error": "Position embeddings not accessible"}
        
        test_suite = [
            ("Calculate: 5\nAnswer:", "5"),
            ("Calculate: 2 + 2\nAnswer:", "4"),
            ("Calculate: 23 + 45\nAnswer:", "68"),
            ("Calculate: (12 + 8) * 3\nAnswer:", "60")
        ]
        
        intervention_results = []
        
        pos_embed_module = self.architecture_info["embed_positions"]
        original_weights = pos_embed_module.weight.data.clone()
        
        try:
            for scale in self.config.intervention_scales:
                logger.info(f"Testing position scale: {scale}x")
                
                pos_embed_module.weight.data = original_weights * scale
                
                scale_results = []
                for prompt, expected in test_suite:
                    metrics = self._measure_generation_metrics(
                        prompt,
                        expected,
                        head_mask=None,
                        n_runs=self.config.n_replication
                    )
                    
                    scale_results.append({
                        "prompt": prompt[:20],
                        "expected": expected,
                        **metrics
                    })
                
                intervention_results.append({
                    "scale": scale,
                    "mean_response_length": np.mean([r["mean_length"] for r in scale_results]),
                    "std_response_length": np.std([r["mean_length"] for r in scale_results]),
                    "mean_accuracy": np.mean([r["accuracy"] for r in scale_results]),
                    "mean_hallucination": np.mean([r["hallucination_rate"] for r in scale_results]),
                    "detailed_results": scale_results
                })
                
        finally:
            pos_embed_module.weight.data = original_weights
            logger.info("Position embeddings restored")
        
        analysis = self._analyze_intervention_results(intervention_results)
        
        return {
            "intervention_results": intervention_results,
            "analysis": analysis,
            "optimal_scale": analysis["optimal_scale"]
        }
    
    def _analyze_intervention_results(self, results: List[Dict]) -> Dict:
        """Analyze intervention results to find optimal scale"""
        scales = [r["scale"] for r in results]
        response_lengths = [r["mean_response_length"] for r in results]
        accuracies = [r["mean_accuracy"] for r in results]
        hallucinations = [r["mean_hallucination"] for r in results]
        
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c
        
        try:
            popt_length, pcov_length = curve_fit(quadratic, scales, response_lengths)
            
            optimal_scale = -popt_length[1] / (2 * popt_length[0])
            optimal_scale = np.clip(optimal_scale, min(scales), max(scales))
            
            residuals = response_lengths - quadratic(np.array(scales), *popt_length)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((response_lengths - np.mean(response_lengths))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
        except:
            optimal_idx = np.argmin(response_lengths)
            optimal_scale = scales[optimal_idx]
            r_squared = 0.0
            popt_length = None
        
        return {
            "optimal_scale": optimal_scale,
            "r_squared": r_squared,
            "length_at_optimal": quadratic(optimal_scale, *popt_length) if popt_length is not None else min(response_lengths),
            "correlation_length_scale": np.corrcoef(scales, response_lengths)[0, 1],
            "correlation_accuracy_scale": np.corrcoef(scales, accuracies)[0, 1],
            "correlation_hallucination_scale": np.corrcoef(scales, hallucinations)[0, 1]
        }