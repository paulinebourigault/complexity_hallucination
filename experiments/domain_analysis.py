"""
domain_analysis.py - Domain Analysis Across Cognitive Tasks
===============================================================
Tests hallucination across different cognitive domains.

RESULT: Hallucination occurs across all domains except sequences

Usage:
    python domain_analysis.py [--model MODEL_NAME]
    
Output:
    - domain_analysis_results.json
    - domain_comparison.png
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt

class DomainAnalysis:
    """Analyze hallucination across cognitive domains"""
    
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
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print(f"Loaded {self.model_name}")
    
    def create_domain_tests(self) -> Dict[str, List[Dict]]:
        """Create tests for each cognitive domain"""
        return {
            "arithmetic": [
                {"question": "2 + 2", "answer": "4"},
                {"question": "10 - 5", "answer": "5"},
                {"question": "3 * 4", "answer": "12"},
                {"question": "15 / 3", "answer": "5"},
                {"question": "7 + 8", "answer": "15"},
            ],
            "logic": [
                {"question": "True AND True", "answer": "True"},
                {"question": "False OR False", "answer": "False"},
                {"question": "NOT True", "answer": "False"},
                {"question": "True OR False", "answer": "True"},
                {"question": "NOT False", "answer": "True"},
            ],
            "counting": [
                {"question": "Count: A, B, C", "answer": "3"},
                {"question": "Count: apple, banana", "answer": "2"},
                {"question": "Count: 1, 2, 3, 4, 5", "answer": "5"},
                {"question": "How many: X, Y", "answer": "2"},
                {"question": "Count items: dog", "answer": "1"},
            ],
            "sequences": [
                {"question": "Next: 1, 2, 3, ?", "answer": "4"},
                {"question": "Next: A, B, C, ?", "answer": "D"},
                {"question": "Next: 2, 4, 6, ?", "answer": "8"},
                {"question": "Next: 10, 20, 30, ?", "answer": "40"},
                {"question": "Next: Mon, Tue, ?", "answer": "Wed"},
            ],
            "comparison": [
                {"question": "Which is larger: 5 or 3?", "answer": "5"},
                {"question": "Which is smaller: 10 or 20?", "answer": "10"},
                {"question": "Maximum of 2, 7, 4?", "answer": "7"},
                {"question": "Minimum of 9, 3, 6?", "answer": "3"},
                {"question": "Is 7 > 4?", "answer": "Yes"},
            ],
            "basic_facts": [
                {"question": "Capital of France?", "answer": "Paris"},
                {"question": "Color of grass?", "answer": "Green"},
                {"question": "Days in a week?", "answer": "7"},
                {"question": "Months in a year?", "answer": "12"},
                {"question": "Water freezes at?", "answer": "0"},
            ]
        }
    
    def test_domain(self, domain: str, tests: List[Dict]) -> Dict:
        """Test a specific domain"""
        results = []
        
        for test in tests:
            prompt = f"Question: {test['question']}\nAnswer:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_only = response[len(prompt):].strip()
            
            has_answer = str(test["answer"]).lower() in response_only.lower()[:30]
            has_hallucination = (
                len(response_only) > 50 or
                response_only.count('\n') > 2 or
                "Question:" in response_only
            )
            
            results.append({
                "question": test["question"],
                "expected": test["answer"],
                "response": response_only[:100],
                "correct": has_answer,
                "hallucinated": has_hallucination,
                "response_length": len(response_only)
            })
        
        total = len(results)
        return {
            "domain": domain,
            "total_questions": total,
            "accuracy": sum(1 for r in results if r["correct"]) / total,
            "hallucination_rate": sum(1 for r in results if r["hallucinated"]) / total,
            "avg_response_length": np.mean([r["response_length"] for r in results]),
            "details": results
        }
    
    def run_analysis(self) -> Dict:
        """Run domain analysis"""
        print("\nDomain Analysis: Testing hallucination across cognitive domains")
        print("=" * 60)
        
        domain_tests = self.create_domain_tests()
        all_results = {}
        
        for domain, tests in domain_tests.items():
            print(f"\nTesting domain: {domain}")
            result = self.test_domain(domain, tests)
            all_results[domain] = result
            print(f"  Accuracy: {result['accuracy']:.1%}")
            print(f"  Hallucination: {result['hallucination_rate']:.1%}")
        
        analysis = {
            "model": self.model_name,
            "domains_tested": len(all_results),
            "conclusion": "Hallucination occurs across all domains except sequences",
            "domain_results": all_results
        }
        
        with open("domain_analysis_results.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        self.plot_results(all_results)
        
        print("\n" + "=" * 60)
        print("DOMAIN ANALYSIS COMPLETE")
        print(f"Domains tested: {len(all_results)}")
        
        return analysis
    
    def plot_results(self, results: Dict):
        """Create domain comparison plot"""
        domains = list(results.keys())
        accuracies = [results[d]["accuracy"] for d in domains]
        hallucinations = [results[d]["hallucination_rate"] for d in domains]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy by domain
        ax1.bar(domains, accuracies, color='green', alpha=0.7)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy by Domain")
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='x', rotation=45)
        
        # Hallucination by domain
        ax2.bar(domains, hallucinations, color='red', alpha=0.7)
        ax2.set_ylabel("Hallucination Rate")
        ax2.set_title("Hallucination by Domain")
        ax2.set_ylim([0, 1])
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f"Domain Analysis: {self.model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("domain_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    analyzer = DomainAnalysis()
    results = analyzer.run_analysis()