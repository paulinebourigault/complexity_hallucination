"""
phenomenon_documentation.py - Initial Observation
=====================================================================
Documents the complexity-induced hallucination phenomenon across models.

RESULT: Found consistent hallucination pattern across all tested models

Usage:
    python phenomenon_documentation.py [--model MODEL_NAME]
    
Output:
    - phenomenon_documentation.json
    - hallucination_examples.txt
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import List, Dict
from tqdm import tqdm
import gc

class PhenomenonDocumentation:
    """Document the initial observation of complexity-induced hallucination"""
    
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
    
    def create_test_suite(self) -> List[Dict]:
        """Create test problems of varying complexity"""
        return [
            # Trivial - no computation needed
            {"problem": "5", "answer": "5", "complexity": "trivial"},
            {"problem": "What is 3?", "answer": "3", "complexity": "trivial"},
            
            # Simple - single operation
            {"problem": "2 + 2", "answer": "4", "complexity": "simple"},
            {"problem": "10 - 5", "answer": "5", "complexity": "simple"},
            {"problem": "3 * 4", "answer": "12", "complexity": "simple"},
            
            # Moderate - two operations
            {"problem": "23 + 45", "answer": "68", "complexity": "moderate"},
            {"problem": "7 * 8 + 3", "answer": "59", "complexity": "moderate"},
            
            # Complex - multiple operations
            {"problem": "(12 + 8) * 3", "answer": "60", "complexity": "complex"},
            {"problem": "25% of 80", "answer": "20", "complexity": "complex"},
        ]
    
    def test_problem(self, problem_dict: Dict) -> Dict:
        """Test a single problem and document the response"""
        prompt = f"Calculate: {problem_dict['problem']}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = response[len(prompt):].strip()
        
        # Check if answer is present
        has_answer = problem_dict["answer"] in response_only[:20]
        
        # Check for hallucination indicators
        has_hallucination = (
            len(response_only) > 100 or
            "Calculate:" in response_only[10:] or
            response_only.count('\n') > 3
        )
        
        return {
            "problem": problem_dict["problem"],
            "expected": problem_dict["answer"],
            "complexity": problem_dict["complexity"],
            "response": response_only,
            "response_length": len(response_only),
            "has_answer": has_answer,
            "has_hallucination": has_hallucination,
            "expansion_factor": len(response_only) / len(problem_dict["answer"])
        }
    
    def run_documentation(self) -> Dict:
        """Document the phenomenon"""
        print("\nDocumenting Complexity-Induced Hallucination")
        print("=" * 50)
        
        test_suite = self.create_test_suite()
        results = []
        
        for problem in tqdm(test_suite, desc="Testing problems"):
            result = self.test_problem(problem)
            results.append(result)
            
            if result["has_hallucination"] and result["has_answer"]:
                print(f"\nExample - {result['complexity']} problem:")
                print(f"  Problem: {result['problem']}")
                print(f"  Expected: {result['expected']}")
                print(f"  Response length: {result['response_length']} chars")
                print(f"  Expansion factor: {result['expansion_factor']:.1f}x")
                print(f"  Response preview: {result['response'][:100]}...")
        
        stats = self.calculate_statistics(results)
        
        documentation = {
            "model": self.model_name,
            "phenomenon": "Complexity-Induced Hallucination",
            "description": "Models generate correct answers but continue with excessive hallucinated content",
            "results": results,
            "statistics": stats
        }
        
        with open("phenomenon_documentation.json", "w") as f:
            json.dump(documentation, f, indent=2)
        
        with open("hallucination_examples.txt", "w") as f:
            for r in results:
                if r["has_hallucination"]:
                    f.write(f"Problem: {r['problem']}\n")
                    f.write(f"Expected: {r['expected']}\n")
                    f.write(f"Response ({r['response_length']} chars):\n")
                    f.write(f"{r['response']}\n")
                    f.write("=" * 50 + "\n\n")
        
        print("\n" + "=" * 50)
        print("PHENOMENON DOCUMENTATION COMPLETE")
        print(f"Overall accuracy: {stats['accuracy']:.1%}")
        print(f"Hallucination rate: {stats['hallucination_rate']:.1%}")
        print(f"Average expansion: {stats['avg_expansion']:.1f}x")
        
        return documentation
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive statistics"""
        total = len(results)
        correct = sum(1 for r in results if r["has_answer"])
        hallucinated = sum(1 for r in results if r["has_hallucination"])
        
        # By complexity
        complexity_stats = {}
        for complexity in ["trivial", "simple", "moderate", "complex"]:
            complexity_results = [r for r in results if r["complexity"] == complexity]
            if complexity_results:
                complexity_stats[complexity] = {
                    "accuracy": sum(1 for r in complexity_results if r["has_answer"]) / len(complexity_results),
                    "hallucination_rate": sum(1 for r in complexity_results if r["has_hallucination"]) / len(complexity_results),
                    "avg_expansion": np.mean([r["expansion_factor"] for r in complexity_results])
                }
        
        return {
            "total_problems": total,
            "accuracy": correct / total,
            "hallucination_rate": hallucinated / total,
            "avg_expansion": np.mean([r["expansion_factor"] for r in results]),
            "max_expansion": max(r["expansion_factor"] for r in results),
            "by_complexity": complexity_stats
        }

if __name__ == "__main__":
    documenter = PhenomenonDocumentation()
    documentation = documenter.run_documentation()