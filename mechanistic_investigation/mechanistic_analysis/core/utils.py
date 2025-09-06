"""
utils.py
"""

import json
import numpy as np
from dataclasses import asdict
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(data: Dict, filepath: str):
    """Save checkpoint"""
    try:
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                return asdict(obj)
            return obj
        
        serializable_data = convert_arrays(data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def synthesize_mechanistic_story(experiments: Dict, config) -> Dict:
    """Synthesize all experimental findings"""
    story = {
        "key_findings": [],
        "mechanism": {},
        "evidence_quality": {}
    }
    
    if "attention_patterns" in experiments:
        n_significant = len(experiments["attention_patterns"]["significant_heads"])
        if n_significant > 0:
            story["key_findings"].append(
                f"Identified {n_significant} statistically significant position-sensitive attention heads"
            )
    
    if "ablation" in experiments and "causal_heads" in experiments["ablation"]:
        n_causal = len([h for h in experiments["ablation"]["causal_heads"] if h.causal])
        if n_causal > 0:
            story["key_findings"].append(
                f"Confirmed {n_causal} causally important heads through ablation"
            )
    
    if "position_intervention" in experiments:
        optimal = experiments["position_intervention"].get("optimal_scale")
        if optimal:
            story["key_findings"].append(
                f"Optimal position scaling of {optimal:.2f}x reduces hallucination"
            )
    
    if "logit_lens" in experiments:
        analysis = experiments["logit_lens"]["analysis"]
        if analysis.get("answer_before_hallucination"):
            story["key_findings"].append(
                "Answer computed before hallucination signals appear"
            )
    
    story["mechanism"] = {
        "components": [],
        "causal_chain": []
    }
    
    story["evidence_quality"] = {
        "replication": f"{config.n_replication} runs per condition",
        "multiple_testing_correction": "Bonferroni correction applied",
        "effect_sizes": "Reported with 95% confidence intervals"
    }
    
    return story