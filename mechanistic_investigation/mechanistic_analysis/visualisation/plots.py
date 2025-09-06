"""
plots.py - Plots for mechanistic analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MechanisticVisualizer:
    """Create visualizations"""
    
    def __init__(self, architecture_info: Dict):
        self.architecture_info = architecture_info
        self._setup_style()
    
    def _setup_style(self):
        """Set plotting style"""
        plt.style.use('seaborn-v0_8-white')
        sns.set_palette("husl")
        sns.set_style("white")
        plt.rcParams['axes.grid'] = False
    
    def create_comprehensive_visualizations(self, results: Dict, save_dir: str):
        """Create all visualizations"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Attention head sensitivity heatmap
        ax1 = fig.add_subplot(gs[0, :])
        if "attention_patterns" in results["experiments"]:
            self._plot_attention_heatmap(ax1, results["experiments"]["attention_patterns"])
        
        # 2. Ablation effects
        ax2 = fig.add_subplot(gs[1, 0])
        if "ablation" in results["experiments"]:
            self._plot_ablation_effects(ax2, results["experiments"]["ablation"])
        
        # 3. Position intervention
        ax3 = fig.add_subplot(gs[1, 1])
        if "position_intervention" in results["experiments"]:
            self._plot_position_intervention(ax3, results["experiments"]["position_intervention"])
        
        # 4. Logit lens progression
        ax4 = fig.add_subplot(gs[1, 2])
        if "logit_lens" in results["experiments"]:
            self._plot_logit_lens(ax4, results["experiments"]["logit_lens"])
        
        # 5. Statistical summary
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_statistical_summary(ax5, results)
        
        # 6. Mechanistic diagram
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_mechanistic_diagram(ax6, results)
        
        plt.suptitle(f"Mechanistic Analysis: {results['config']['model_name']}", fontsize=16, fontweight='bold')
        plt.savefig(f"{save_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/comprehensive_analysis.pdf", format='pdf', bbox_inches='tight')
        
        logger.info(f"Visualizations saved to {save_dir}")
    
    def _plot_attention_heatmap(self, ax, attention_data):
        """Plot attention head sensitivity heatmap"""
        if "significant_heads" not in attention_data:
            return
        
        n_layers = self.architecture_info["n_layers"]
        n_heads = self.architecture_info["n_heads"]
        
        sensitivity_matrix = np.zeros((n_layers, n_heads))
        significance_matrix = np.zeros((n_layers, n_heads))
        
        for head in attention_data["significant_heads"]:
            if hasattr(head, 'layer'):
                sensitivity_matrix[head.layer, head.head] = head.effect_size
                significance_matrix[head.layer, head.head] = 1 if head.is_significant else 0.5
        
        im = ax.imshow(sensitivity_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        
        for i in range(n_layers):
            for j in range(n_heads):
                if significance_matrix[i, j] == 1:
                    ax.text(j, i, '*', ha='center', va='center', color='white', fontweight='bold')
        
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title("Attention Head Position Sensitivity (* = significant after correction)")
        plt.colorbar(im, ax=ax, label="Effect Size (η²)")
    
    def _plot_ablation_effects(self, ax, ablation_data):
        """Plot ablation effects"""
        if "causal_heads" not in ablation_data:
            return
        
        causal_heads = ablation_data["causal_heads"]
        if not causal_heads:
            return
        
        effects = [h.effect_size for h in causal_heads[:10]]
        labels = [f"L{h.layer}H{h.head}" for h in causal_heads[:10]]
        errors = [(h.confidence_interval[1] - h.confidence_interval[0])/2 for h in causal_heads[:10]]
        
        x = np.arange(len(effects))
        bars = ax.bar(x, effects, yerr=errors, capsize=5)
        
        for i, head in enumerate(causal_heads[:10]):
            if head.causal:
                bars[i].set_color('red')
            else:
                bars[i].set_color('gray')
        
        ax.set_xlabel("Head")
        ax.set_ylabel("Effect Size (Cohen's d)")
        ax.set_title("Causal Effects of Head Ablation")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(0.2, color='red', linestyle='--', alpha=0.5, label='Effect threshold')
        ax.axhline(-0.2, color='red', linestyle='--', alpha=0.5)
        ax.legend()
    
    def _plot_position_intervention(self, ax, intervention_data):
        """Plot position intervention results"""
        if "intervention_results" not in intervention_data:
            return
        
        results = intervention_data["intervention_results"]
        
        scales = [r["scale"] for r in results]
        lengths = [r["mean_response_length"] for r in results]
        stds = [r["std_response_length"] for r in results]
        
        ax.errorbar(scales, lengths, yerr=stds, marker='o', linewidth=2, capsize=5)
        
        if "analysis" in intervention_data:
            optimal = intervention_data["analysis"]["optimal_scale"]
            ax.axvline(optimal, color='red', linestyle='--', label=f'Optimal: {optimal:.2f}')
        
        ax.set_xlabel("Position Embedding Scale")
        ax.set_ylabel("Response Length (chars)")
        ax.set_title("Effect of Position Embedding Scaling")
        ax.legend()
    
    def _plot_logit_lens(self, ax, logit_data):
        """Plot logit lens progression"""
        if "analysis" not in logit_data:
            return
        
        analysis = logit_data["analysis"]
        
        categories = ["Answer\nAppears", "Hallucination\nOnset"]
        means = [
            analysis["mean_answer_appearance_layer"],
            analysis["mean_hallucination_onset_layer"]
        ]
        stds = [
            analysis["std_answer_appearance_layer"],
            analysis["std_hallucination_onset_layer"]
        ]
        
        x = np.arange(len(categories))
        ax.bar(x, means, yerr=stds, capsize=10, color=['green', 'red'], alpha=0.7)
        
        ax.set_ylabel("Layer")
        ax.set_title("Logit Lens: Where Predictions Emerge")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        
        ax.text(0.5, max(means) * 0.9, analysis["interpretation"],
               ha='center', fontsize=10, style='italic', wrap=True)
    
    def _plot_statistical_summary(self, ax, results):
        """Plot statistical summary table"""
        ax.axis('tight')
        ax.axis('off')
        
        summary_data = []
        
        if "attention_patterns" in results["experiments"]:
            n_tested = results["experiments"]["attention_patterns"]["n_tests"]
            n_significant = len(results["experiments"]["attention_patterns"]["significant_heads"])
            summary_data.append(["Attention Heads Tested", n_tested])
            summary_data.append(["Significant Heads", f"{n_significant} ({n_significant/n_tested*100:.1f}%)"])
        
        if "ablation" in results["experiments"] and "causal_heads" in results["experiments"]["ablation"]:
            n_causal = len([h for h in results["experiments"]["ablation"]["causal_heads"] if h.causal])
            summary_data.append(["Causally Important Heads", n_causal])
        
        if "position_intervention" in results["experiments"]:
            optimal = results["experiments"]["position_intervention"].get("optimal_scale", "N/A")
            summary_data.append(["Optimal Position Scale", f"{optimal:.2f}x" if isinstance(optimal, float) else optimal])
        
        summary_data.append(["Replication", f"{results['config']['n_replication']} runs per condition"])
        summary_data.append(["Confidence Level", f"{results['config']['confidence_level']*100:.0f}%"])
        
        table = ax.table(cellText=summary_data,
                        colLabels=["Metric", "Value"],
                        cellLoc='left',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title("Statistical Summary", fontweight='bold', pad=20)
    
    def _plot_mechanistic_diagram(self, ax, results):
        """Plot mechanistic story diagram"""
        ax.axis('off')
        
        if "mechanistic_story" in results:
            story = results["mechanistic_story"]
            
            text = story["executive_summary"] + "\n\n"
            text += "KEY FINDINGS:\n"
            for i, finding in enumerate(story["key_findings"][:5], 1):
                text += f"{i}. {finding}\n"
            
            ax.text(0.5, 0.5, text, ha='center', va='center',
                   fontsize=11, wrap=True, transform=ax.transAxes)
        
        ax.set_title("Mechanistic Story", fontweight='bold', fontsize=12)