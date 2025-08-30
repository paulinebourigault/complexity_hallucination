"""
layer_causality.py - Layer-wise Template Override Hypothesis
================================================================
Tests if late layers ignore early complexity signals and execute templates.

RESULT: REJECTED - Late layers maintain MORE distinction

Usage:
    python layer_causality.py [--model MODEL_NAME]
    
Output:
    - layer_causality_results.json
    - layer_causality_analysis.png
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import json
from typing import Dict, List
from sklearn.linear_model import LogisticRegression

class LayerCausalityAnalysis:
    """Test if late layers override early complexity detection"""
    
    def __init__(self, model_name="google/gemma-2b"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.load_model()
        
    def load_model(self):
        """Load model with output access"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            output_hidden_states=True,
            output_attentions=True
        )
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        print(f"Loaded {self.model_name} with {self.n_layers} layers")
    
    def extract_layer_representations(self, text: str) -> Dict:
        """Extract representations from all layers"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        
        layer_data = []
        for layer_idx, hidden in enumerate(hidden_states):
            layer_norm = torch.norm(hidden[0], p=2).item()
            layer_mean = hidden[0].mean().item()
            layer_data.append({
                "layer": layer_idx,
                "norm": layer_norm,
                "mean": layer_mean
            })
        
        return layer_data
    
    def test_complexity_detection(self) -> Dict:
        """Test if model can distinguish simple from complex problems"""
        simple_problems = [
            "Calculate: 5\nAnswer:",
            "Calculate: 2 + 2\nAnswer:",
            "Calculate: 10 - 5\nAnswer:"
        ]
        
        complex_problems = [
            "Calculate: (12 + 8) * 3\nAnswer:",
            "Calculate: 25% of 80\nAnswer:",
            "Calculate: If 2x + 5 = 13, x = ?\nAnswer:"
        ]
        
        # Extract representations
        simple_reps = []
        for prob in simple_problems:
            reps = self.extract_layer_representations(prob)
            simple_reps.append([r["norm"] for r in reps[-5:]])  # Last 5 layers
        
        complex_reps = []
        for prob in complex_problems:
            reps = self.extract_layer_representations(prob)
            complex_reps.append([r["norm"] for r in reps[-5:]])
        
        # Train probe
        X = np.array(simple_reps + complex_reps)
        y = np.array([0] * len(simple_reps) + [1] * len(complex_reps))
        probe = LogisticRegression(random_state=42)
        probe.fit(X, y)
        
        return {
            "probe_accuracy": probe.score(X, y),
            "can_distinguish": probe.score(X, y) > 0.8
        }
    
    def test_information_flow(self) -> Dict:
        """Test how information flows through layers"""
        test_pairs = [
            ("Calculate: 5\nAnswer:", "Calculate: (2+3)×4\nAnswer:"),
            ("Calculate: 2+2\nAnswer:", "Calculate: 15% of 60\nAnswer:"),
            ("Calculate: 3×4\nAnswer:", "Calculate: x^2 = 16\nAnswer:")
        ]
        
        flow_data = []
        for simple, complex in test_pairs:
            simple_reps = self.extract_layer_representations(simple)
            complex_reps = self.extract_layer_representations(complex)
            
            distinctions = []
            for i in range(len(simple_reps)):
                dist = abs(simple_reps[i]["norm"] - complex_reps[i]["norm"])
                distinctions.append(dist)
            
            flow_data.append(distinctions)
        
        avg_distinctions = np.mean(flow_data, axis=0)
        
        early_avg = np.mean(avg_distinctions[:6])
        late_avg = np.mean(avg_distinctions[-6:])
        
        return {
            "distinctions_by_layer": avg_distinctions.tolist(),
            "early_layers_distinction": early_avg,
            "late_layers_distinction": late_avg,
            "information_flow_ratio": early_avg / late_avg if late_avg > 0 else 0
        }
    
    def run_analysis(self) -> Dict:
        """Run layer causality analysis"""
        print("\nTesting Layer-wise Template Override Hypothesis")
        print("=" * 50)
        results = {}
        print("1. Testing complexity detection...")
        results["complexity_detection"] = self.test_complexity_detection()
        print("2. Testing information flow...")
        results["information_flow"] = self.test_information_flow()
        
        with open("layer_causality_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        self.visualize_results(results)
        
        return results
    
    def visualize_results(self, results: Dict):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if "information_flow" in results:
            distinctions = results["information_flow"]["distinctions_by_layer"]
            layers = list(range(len(distinctions)))
            
            ax1.plot(layers, distinctions, 'o-', linewidth=2, color='blue')
            ax1.axvspan(0, 5, alpha=0.2, color='green', label='Early layers')
            ax1.axvspan(len(layers)-6, len(layers)-1, alpha=0.2, color='red', label='Late layers')
            ax1.set_xlabel("Layer Index")
            ax1.set_ylabel("Simple vs Complex Distinction")
            ax1.set_title("Information Flow Through Layers")
            ax1.legend()
        
        ax2.axis('off')
        ratio = results["information_flow"]["information_flow_ratio"]
        summary = f"LAYER CAUSALITY FINDINGS:\n\n"
        summary += f"1. Probe accuracy: {results['complexity_detection']['probe_accuracy']:.1%}\n"
        summary += f"2. Information flow ratio: {ratio:.2f}x\n"
        ax2.text(0.1, 0.5, summary, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig("layer_causality_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    analyzer = LayerCausalityAnalysis()
    results = analyzer.run_analysis()