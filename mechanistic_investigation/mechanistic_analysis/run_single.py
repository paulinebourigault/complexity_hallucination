"""
run_single.py - Run individual experiments
"""

import argparse
import json
from dataclasses import asdict
import logging

from core.config import ExperimentConfig
from experiments.attention_analysis import AttentionAnalyzer
from experiments.ablation_study import AblationAnalyzer
from experiments.position_intervention import PositionInterventionAnalyzer
from experiments.logit_lens import LogitLensAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Run single mechanistic experiment"
    )
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["attention", "ablation", "position", "logit"],
                       help="Experiment to run")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-replication", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        model_name=args.model,
        device=args.device,
        n_replication=args.n_replication,
        seed=args.seed
    )
    
    print(f"Running {args.experiment} experiment on {args.model}")
    
    if args.experiment == "attention":
        analyzer = AttentionAnalyzer(config)
        results = analyzer.analyze_attention_patterns()
    elif args.experiment == "ablation":
        # For ablation, we need attention heads first
        attention_analyzer = AttentionAnalyzer(config)
        attention_results = attention_analyzer.analyze_attention_patterns()
        heads_to_test = [
            (h.layer, h.head)
            for h in attention_results["significant_heads"][:10]
        ]
        analyzer = AblationAnalyzer(config)
        results = analyzer.ablation_study_controlled(heads_to_test)
    elif args.experiment == "position":
        analyzer = PositionInterventionAnalyzer(config)
        results = analyzer.position_embedding_intervention()
    elif args.experiment == "logit":
        analyzer = LogitLensAnalyzer(config)
        results = analyzer.logit_lens_analysis()
    
    output_file = f"{args.experiment}_{args.model}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    results = main()