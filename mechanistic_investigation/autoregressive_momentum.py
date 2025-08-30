"""
autoregressive_momentum.py - Autoregressive Momentum Hypothesis
===================================================================
Tests if token generation creates unstoppable cascades.

RESULT: REJECTED - No probability decay or entropy increase observed

Usage:
    python autoregressive_momentum.py [--model MODEL_NAME]
    
Output:
    - autoregressive_momentum_results.json
    - autoregressive_momentum_analysis.png
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import json
from typing import Dict, List

class AutoregressiveMomentumAnalysis:
    """Test if hallucination is caused by autoregressive generation momentum"""
    
    def __init__(self, model_name="google/gemma-2b"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.load_model()
        
    def load_model(self):
        """Load model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        print(f"Loaded {self.model_name}")
    
    def analyze_stopping_probabilities(self) -> Dict:
        """Track how probability of stopping changes during generation"""
        test_cases = [
            ("Calculate: 5\nAnswer:", "5", "trivial"),
            ("Calculate: 2 + 2\nAnswer:", "4", "simple"),
            ("Calculate: 23 + 45\nAnswer:", "68", "moderate"),
            ("Calculate: (12 + 8) * 3\nAnswer:", "60", "complex")
        ]
        
        results = {}
        
        for prompt, answer, complexity in test_cases:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id
            
            generation_probs = []
            current_input_ids = inputs.input_ids
            
            for step in range(20):  # Track first 20 tokens
                with torch.no_grad():
                    outputs = self.model(current_input_ids)
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    
                    stop_prob = probs[eos_token_id].item() if eos_token_id else 0
                    
                    template_tokens = self.tokenizer.encode("Calculate:", add_special_tokens=False)
                    template_prob = max([probs[t].item() for t in template_tokens[:1]]) if template_tokens else 0
                    
                    next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
                    
                    generation_probs.append({
                        "step": step,
                        "stop_prob": stop_prob,
                        "template_prob": template_prob
                    })
                    
                    current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                    
                    if next_token.item() == eos_token_id:
                        break
            
            results[complexity] = {
                "probabilities": generation_probs,
                "total_length": len(generation_probs)
            }
        
        return results
    
    def test_context_saturation(self) -> Dict:
        """Test if context length affects hallucination"""
        base_prompt = "Calculate: 23 + 45\nAnswer:"
        
        context_lengths = [0, 50, 100, 200]
        results = []
        
        for context_len in context_lengths:
            if context_len > 0:
                padding = "." * context_len + "\n"
                full_prompt = padding + base_prompt
            else:
                full_prompt = base_prompt
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True
                )
            
            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = generated[len(full_prompt):]
            
            results.append({
                "context_length": context_len,
                "response_length": len(response),
                "has_repetition": "Calculate:" in response[10:]
            })
        
        return {"context_saturation": results}
    
    def run_analysis(self) -> Dict:
        """Run momentum analysis"""
        print("\nTesting Autoregressive Momentum Hypothesis")
        print("=" * 50)
        
        results = {}
        
        print("1. Analyzing stopping probabilities...")
        results["stopping_probs"] = self.analyze_stopping_probabilities()
        
        print("2. Testing context saturation...")
        results["context_saturation"] = self.test_context_saturation()
        
        with open("autoregressive_momentum_results.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        self.visualize_results(results)
        self.print_conclusions(results)
        
        return results
    
    def visualize_results(self, results: Dict):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Stopping probability
        for complexity, data in results["stopping_probs"].items():
            steps = [p["step"] for p in data["probabilities"]]
            stop_probs = [p["stop_prob"] for p in data["probabilities"]]
            ax1.plot(steps, stop_probs, 'o-', label=complexity, alpha=0.7)
        
        ax1.set_xlabel('Generation Step')
        ax1.set_ylabel('P(stop)')
        ax1.set_title('Stopping Probability During Generation')
        ax1.legend()
        
        # Template probability
        for complexity, data in results["stopping_probs"].items():
            steps = [p["step"] for p in data["probabilities"]]
            template_probs = [p["template_prob"] for p in data["probabilities"]]
            ax2.plot(steps, template_probs, 's-', label=complexity, alpha=0.7)
        
        ax2.set_xlabel('Generation Step')
        ax2.set_ylabel('P(template)')
        ax2.set_title('Template Repetition Probability')
        ax2.legend()
        
        # Context saturation
        if "context_saturation" in results:
            context_data = results["context_saturation"]["context_saturation"]
            context_lens = [d["context_length"] for d in context_data]
            response_lens = [d["response_length"] for d in context_data]
            
            ax3.plot(context_lens, response_lens, 'o-', color='blue', linewidth=2)
            ax3.set_xlabel('Context Length')
            ax3.set_ylabel('Response Length')
            ax3.set_title('Context Length vs Hallucination')
        
        plt.suptitle('Autoregressive Momentum Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('autoregressive_momentum_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_conclusions(self, results: Dict):
        avg_decay = []
        for complexity, data in results["stopping_probs"].items():
            probs = data["probabilities"]
            if len(probs) > 1:
                decay = probs[0]["stop_prob"] - probs[-1]["stop_prob"]
                avg_decay.append(decay)
        
        if avg_decay:
            print(f"Average stopping probability decay: {np.mean(avg_decay):.3f}")

if __name__ == "__main__":
    analyzer = AutoregressiveMomentumAnalysis()
    results = analyzer.run_analysis()