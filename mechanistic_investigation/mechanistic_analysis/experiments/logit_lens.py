"""
logit_lens.py - Layer-wise prediction tracking through logit lens
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import logging
from ..core.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class LogitLensAnalyzer(BaseAnalyzer):
    """Analyze predictions through layers using logit lens"""
    
    def logit_lens_analysis(self) -> Dict:
        """
        Logit lens analysis tracking predictions through layers.
        """
        logger.info("Starting logit lens analysis")
        
        test_prompts = [
            ("Calculate: 2 + 2\nAnswer:", "4"),
            ("Calculate: 23 + 45\nAnswer:", "68"),
            ("Calculate: 100 - 37\nAnswer:", "63")
        ]
        
        all_layer_predictions = []
        
        for prompt, expected in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            lm_head = self.architecture_info.get("lm_head")
            if lm_head is None:
                logger.warning("LM head not found")
                continue
            
            layer_predictions = []
            
            for layer_idx, hidden_state in enumerate(outputs.hidden_states[1:]):
                if hasattr(lm_head, 'weight'):
                    logits = F.linear(hidden_state, lm_head.weight)
                else:
                    logits = lm_head(hidden_state)
                
                probs = F.softmax(logits[0, -1], dim=0)
                
                top_k = 20
                top_probs, top_indices = torch.topk(probs, top_k)
                top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
                
                answer_rank = -1
                answer_prob = 0.0
                for rank, (token, prob) in enumerate(zip(top_tokens, top_probs)):
                    if expected in token:
                        answer_rank = rank
                        answer_prob = prob.item()
                        break
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                
                halluc_indicators = ["\n", "Calculate", "The", "...", "Let", "Step"]
                halluc_probs = sum(
                    probs[self.tokenizer.encode(ind, add_special_tokens=False)[0]].item()
                    for ind in halluc_indicators
                    if len(self.tokenizer.encode(ind, add_special_tokens=False)) > 0
                )
                
                layer_predictions.append({
                    "layer": layer_idx,
                    "top_token": top_tokens[0],
                    "top_prob": top_probs[0].item(),
                    "answer_rank": answer_rank,
                    "answer_prob": answer_prob,
                    "entropy": entropy,
                    "hallucination_probability": halluc_probs,
                    "top_5_tokens": [(t, p.item()) for t, p in zip(top_tokens[:5], top_probs[:5])]
                })
            
            all_layer_predictions.append({
                "prompt": prompt,
                "expected": expected,
                "layer_predictions": layer_predictions
            })
        
        analysis = self._analyze_logit_lens_progression(all_layer_predictions)
        
        return {
            "predictions_by_prompt": all_layer_predictions,
            "analysis": analysis
        }
    
    def _analyze_logit_lens_progression(self, all_predictions: List[Dict]) -> Dict:
        """Analyze how predictions evolve through layers"""
        answer_appearance_layers = []
        hallucination_onset_layers = []
        
        for prompt_data in all_predictions:
            predictions = prompt_data["layer_predictions"]
            
            answer_layer = next(
                (p["layer"] for p in predictions if p["answer_rank"] >= 0 and p["answer_rank"] < 5),
                -1
            )
            answer_appearance_layers.append(answer_layer)
            
            halluc_layer = next(
                (p["layer"] for p in predictions if p["hallucination_probability"] > 0.1),
                -1
            )
            hallucination_onset_layers.append(halluc_layer)
        
        valid_answer_layers = [l for l in answer_appearance_layers if l >= 0]
        valid_halluc_layers = [l for l in hallucination_onset_layers if l >= 0]
        
        return {
            "mean_answer_appearance_layer": np.mean(valid_answer_layers) if valid_answer_layers else -1,
            "std_answer_appearance_layer": np.std(valid_answer_layers) if valid_answer_layers else 0,
            "mean_hallucination_onset_layer": np.mean(valid_halluc_layers) if valid_halluc_layers else -1,
            "std_hallucination_onset_layer": np.std(valid_halluc_layers) if valid_halluc_layers else 0,
            "answer_before_hallucination": np.mean(valid_answer_layers) < np.mean(valid_halluc_layers) if valid_answer_layers and valid_halluc_layers else False,
            "interpretation": self._interpret_logit_lens_results(valid_answer_layers, valid_halluc_layers)
        }
    
    def _interpret_logit_lens_results(
        self,
        answer_layers: List[int],
        halluc_layers: List[int]
    ) -> str:
        """Interpret logit lens findings"""
        if not answer_layers:
            return "Answer never strongly predicted"
        elif not halluc_layers:
            return "No hallucination signals detected"
        elif np.mean(answer_layers) < np.mean(halluc_layers):
            return "Answer computed before hallucination - supports override hypothesis"
        else:
            return "Hallucination and answer emerge simultaneously"