"""
run_all.py - Mechanistic analysis pipeline
"""

import os
import json
import argparse
from datetime import datetime
from dataclasses import asdict
import logging

from core.config import ExperimentConfig
from core.utils import save_checkpoint, synthesize_mechanistic_story
from experiments.attention_analysis import AttentionAnalyzer
from experiments.ablation_study import AblationAnalyzer
from experiments.position_intervention import PositionInterventionAnalyzer
from experiments.logit_lens import LogitLensAnalyzer
from visualisation.plots import MechanisticVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_complete_analysis(config: ExperimentConfig, save_intermediate: bool = True) -> dict:
    """Run mechanistic analysis with all experiments"""
    
    logger.info("="*60)
    logger.info("STARTING COMPLETE MECHANISTIC ANALYSIS")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Replication: {config.n_replication}")
    logger.info("="*60)
    
    results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "experiments": {}
    }
    
    results_dir = f"results_{config.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Experiment 1: Attention patterns
        logger.info("\n--- Experiment 1: Attention Pattern Analysis ---")
        attention_analyzer = AttentionAnalyzer(config)
        results["experiments"]["attention_patterns"] = attention_analyzer.analyze_attention_patterns()
        results["architecture"] = attention_analyzer.architecture_info
        
        if save_intermediate:
            save_checkpoint(results, f"{results_dir}/checkpoint_1.json")
        
        # Experiment 2: Ablation study
        if results["experiments"]["attention_patterns"]["significant_heads"]:
            logger.info("\n--- Experiment 2: Causal Ablation Study ---")
            heads_to_test = [
                (h.layer, h.head)
                for h in results["experiments"]["attention_patterns"]["significant_heads"][:10]
            ]
            ablation_analyzer = AblationAnalyzer(config)
            results["experiments"]["ablation"] = ablation_analyzer.ablation_study_controlled(heads_to_test)
            
            if save_intermediate:
                save_checkpoint(results, f"{results_dir}/checkpoint_2.json")
        
        # Experiment 3: Position intervention
        logger.info("\n--- Experiment 3: Position Embedding Intervention ---")
        position_analyzer = PositionInterventionAnalyzer(config)
        results["experiments"]["position_intervention"] = position_analyzer.position_embedding_intervention()
        
        if save_intermediate:
            save_checkpoint(results, f"{results_dir}/checkpoint_3.json")
        
        # Experiment 4: Logit lens
        logger.info("\n--- Experiment 4: Logit Lens Analysis ---")
        logit_analyzer = LogitLensAnalyzer(config)
        results["experiments"]["logit_lens"] = logit_analyzer.logit_lens_analysis()
        
        if save_intermediate:
            save_checkpoint(results, f"{results_dir}/checkpoint_4.json")
        
        # Synthesize mechanistic story
        logger.info("\n--- Synthesizing Mechanistic Story ---")
        results["mechanistic_story"] = synthesize_mechanistic_story(results["experiments"], config)
        
        save_checkpoint(results, f"{results_dir}/final_results.json")
        
        logger.info("\n--- Generating Visualizations ---")
        visualizer = MechanisticVisualizer(results["architecture"])
        visualizer.create_comprehensive_visualizations(results, results_dir)
        
        logger.info("\n--- Generating Report ---")
        generate_report(results, results_dir)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        results["error"] = str(e)
        save_checkpoint(results, f"{results_dir}/error_checkpoint.json")
        raise
    
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("="*60)
    
    return results

def generate_report(results: dict, save_dir: str):
    """Generate comprehensive text report"""
    report = []
    report.append("="*80)
    report.append("MECHANISTIC ANALYSIS REPORT")
    report.append(f"Model: {results['config']['model_name']}")
    report.append(f"Date: {results['timestamp']}")
    report.append("="*80)
    
    if "mechanistic_story" in results:
        story = results["mechanistic_story"]
        report.append("\nEXECUTIVE SUMMARY")
        report.append("-"*40)
        report.append(story["executive_summary"])
        
        report.append("\nKEY FINDINGS")
        report.append("-"*40)
        for i, finding in enumerate(story["key_findings"], 1):
            report.append(f"{i}. {finding}")
    
    report.append("\nMETHODOLOGY")
    report.append("-"*40)
    report.append(f"- Replication: {results['config']['n_replication']} runs per condition")
    report.append(f"- Statistical confidence: {results['config']['confidence_level']*100:.0f}%")
    report.append(f"- Effect size threshold: {results['config']['effect_size_threshold']}")
    report.append(f"- Multiple testing correction: Bonferroni")
    
    report_path = f"{save_dir}/report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Report saved to {report_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Mechanistic analysis of position-driven hallucination"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-replication", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=100)
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        model_name=args.model,
        device=args.device,
        n_replication=args.n_replication,
        seed=args.seed,
        generation_max_tokens=args.max_tokens
    )
    
    print("="*80)
    print("MECHANISTIC ANALYSIS")
    print("="*80)
    print(f"Configuration:")
    print(json.dumps(asdict(config), indent=2))
    print("="*80)
    
    results = run_complete_analysis(config)
    
    return results

if __name__ == "__main__":
    results = main()