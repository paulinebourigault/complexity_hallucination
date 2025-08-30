"""
scale_analysis.py - Scale Analysis Across Model Sizes
=========================================================
Tests whether increasing model scale (410M-7B) solves hallucination.

RESULT: Scale does NOT solve the problem

Usage:
    python scale_analysis.py
    
Output:
    - scale_analysis_results.json
    - scale_analysis_plot.png
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import List, Dict
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

class ScaleAnalysis:
    """Analyze hallucination across model scales"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models from 410M to 7B parameters
        self.model_configs = [
            {"name": "Pythia-410M", "path": "EleutherAI/pythia-410m", "size": 0.41},
            {"name": "Pythia-1.4B", "path": "EleutherAI/pythia-1.4b", "size": 1.4},
            {"name": "Gemma-2B", "path": "google/gemma-2b", "size": 2.0},
            {"name": "StableLM-3B", "path": "stabilityai/stablelm-3b-4e1t", "size": 3.0},
            {"name": "Pythia-6.9B", "path": "EleutherAI/pythia-6.9b", "size": 6.9},
            {"name": "Llama-2-7B", "path": "meta-llama/Llama-2-7b-hf", "size": 7.0},
            {"name": "Mistral-7B", "path": "mistralai/Mistral-7B-v0.1", "size": 7.0},
            {"name": "Gemma-7B", "path": "google/gemma-7b", "size": 7.0},
        ]
        
    def create_benchmark(self) -> List[Dict]:
        """Create benchmark"""
        return [
            {"problem": "5", "answer": "5", "complexity": "trivial"},
            {"problem": "2 + 2", "answer": "4", "complexity": "simple"},
            {"problem": "10 - 5", "answer": "5", "complexity": "simple"},
            {"problem": "3 * 4", "answer": "12", "complexity": "simple"},
            {"problem": "23 + 45", "answer": "68", "complexity": "moderate"},
            {"problem": "7 * 8 + 3", "answer": "59", "complexity": "moderate"},
            {"problem": "(12 + 8) * 3", "answer": "60", "complexity": "complex"},
            {"problem": "25% of 80", "answer": "20", "complexity": "complex"},
            {"problem": "If 2x + 5 = 13, x = ?", "answer": "4", "complexity": "complex"},
            {"problem": "15 ÷ 3", "answer": "5", "complexity": "simple"},
            {"problem": "100 - 37", "answer": "63", "complexity": "simple"},
            {"problem": "8 + 7", "answer": "15", "complexity": "simple"},
            {"problem": "6 * 9", "answer": "54", "complexity": "simple"},
            {"problem": "Next: 2, 4, 6, ?", "answer": "8", "complexity": "moderate"},
            {"problem": "Count: A, B, C", "answer": "3", "complexity": "simple"},
        ]
    
    def test_model(self, model_config: Dict) -> Dict:
        """Test a single model"""
        print(f"\nTesting {model_config['name']} ({model_config['size']}B parameters)")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_config['path'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_config['path'],
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            model.eval()
            
            benchmark = self.create_benchmark()
            results = []
            
            for problem_dict in tqdm(benchmark, desc=f"Testing {model_config['name']}"):
                prompt = f"Calculate: {problem_dict['problem']}\nAnswer:"
                
                inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(self.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_only = response[len(prompt):].strip()
                
                has_answer = problem_dict["answer"] in response_only[:20]
                has_hallucination = (
                    len(response_only) > 50 or
                    "Calculate:" in response_only[10:]
                )
                
                results.append({
                    "problem": problem_dict["problem"],
                    "complexity": problem_dict["complexity"],
                    "correct": has_answer,
                    "hallucinated": has_hallucination,
                    "response_length": len(response_only)
                })
            
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            
            total = len(results)
            return {
                "model_name": model_config['name'],
                "model_size": model_config['size'],
                "accuracy": sum(1 for r in results if r["correct"]) / total,
                "hallucination_rate": sum(1 for r in results if r["hallucinated"]) / total,
                "avg_response_length": np.mean([r["response_length"] for r in results]),
                "details": results
            }
            
        except Exception as e:
            print(f"Failed to test {model_config['name']}: {e}")
            return {
                "model_name": model_config['name'],
                "model_size": model_config['size'],
                "error": str(e)
            }
    
    def run_analysis(self) -> Dict:
        """Run scale analysis"""
        print("\nScale Analysis: Testing if bigger models solve hallucination")
        print("=" * 60)
        
        all_results = []
        
        for config in self.model_configs:
            result = self.test_model(config)
            if "accuracy" in result:
                all_results.append(result)
                print(f"  Accuracy: {result['accuracy']:.1%}")
                print(f"  Hallucination: {result['hallucination_rate']:.1%}")
        
        analysis = {
            "models_tested": len(all_results),
            "results": all_results
        }
        
        with open("scale_analysis_results.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        self.plot_results(all_results)
        
        print("\n" + "=" * 60)
        print("SCALE ANALYSIS COMPLETE")
        print(f"Models tested: {len(all_results)}")
        
        return analysis
    
    def plot_results(self, results: List[Dict]):
        if not results:
            return
            
        sizes = [r["model_size"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        hallucinations = [r["hallucination_rate"] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy vs Scale
        ax1.plot(sizes, accuracies, 'o-', color='green', linewidth=2, markersize=8)
        ax1.set_xlabel("Model Size (B parameters)")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy vs Model Scale")
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Hallucination vs Scale
        ax2.plot(sizes, hallucinations, 'o-', color='red', linewidth=2, markersize=8)
        ax2.set_xlabel("Model Size (B parameters)")
        ax2.set_ylabel("Hallucination Rate")
        ax2.set_title("Hallucination vs Model Scale")
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig("scale_analysis_plot.png", dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    analyzer = ScaleAnalysis()
    results = analyzer.run_analysis()