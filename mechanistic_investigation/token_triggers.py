"""
token_triggers.py - Deterministic Token Trigger Search
==========================================================
Tests if specific token sequences deterministically cause hallucination.

RESULT: REJECTED - No deterministic triggers found (max 45% rate)

Usage:
    python token_triggers.py [--model MODEL_NAME]
    
Output:
    - token_triggers_results.json
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import Dict, List

class TokenTriggerAnalysis:
    """Identify deterministic token sequences that cause hallucination"""
    
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
    
    def test_token_ngrams(self) -> List[Dict]:
        """Test if specific token sequences trigger hallucination"""
        
        trigger_candidates = [
            ["\n", "\n"],
            [":", " "],
            ["Answer", ":"],
            ["Calculate", ":"],
            [" The", " sum"],
            [".", ".", "."],
            ["\n", "Calculate"],
            ["=", " "],
            [")", " "],
            ["?", "\n"]
        ]
        
        results = []
        
        for ngram in trigger_candidates:
            test_contexts = [
                "The number is",
                "Please compute",
                "Simple task",
                "Result equals",
                "Value found"
            ]
            
            hallucination_rates = []
            
            for context in test_contexts:
                prompt = f"{context} {''.join(ngram)}"
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                halluc_count = 0
                for seed in range(5):
                    torch.manual_seed(seed)
                    
                    with torch.no_grad():
                        output = self.model.generate(
                            inputs.input_ids,
                            max_new_tokens=50,
                            temperature=0.3,
                            do_sample=True
                        )
                    
                    generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    response = generated[len(prompt):]
                    
                    if any(pattern in response for pattern in ["Calculate:", "Answer:", "...", "...]"]):
                        halluc_count += 1
                
                hallucination_rates.append(halluc_count / 5)
            
            mean_rate = np.mean(hallucination_rates)
            std_rate = np.std(hallucination_rates)
            
            results.append({
                "ngram": "".join(ngram),
                "mean_halluc_rate": mean_rate,
                "std_halluc_rate": std_rate,
                "is_deterministic": mean_rate > 0.8 and std_rate < 0.2,
                "context_independent": std_rate < 0.1
            })
        
        return sorted(results, key=lambda x: x["mean_halluc_rate"], reverse=True)
    
    def test_state_hijacking(self) -> List[Dict]:
        """Test if tokens cause irreversible state changes"""
        
        base_prompt = "Calculate: 5 + 3\nAnswer:"
        test_tokens = ["\n\n", "...", "Calculate", "The", ":", "="]
        
        results = []
        
        for token in test_tokens:
            prompt_before = base_prompt
            prompt_after = base_prompt + token
            
            inputs_before = self.tokenizer(prompt_before, return_tensors="pt").to(self.device)
            inputs_after = self.tokenizer(prompt_after, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs_before = self.model(inputs_before.input_ids, output_hidden_states=True)
                outputs_after = self.model(inputs_after.input_ids, output_hidden_states=True)
                
                last_hidden_before = outputs_before.hidden_states[-1][0, -1, :]
                last_hidden_after = outputs_after.hidden_states[-1][0, -1, :]
                
                state_change = torch.norm(last_hidden_after - last_hidden_before).item()
                
            results.append({
                "token": token,
                "state_change": state_change
            })
        
        return sorted(results, key=lambda x: x["state_change"], reverse=True)
    
    def run_analysis(self) -> Dict:
        """Run token trigger analysis"""
        print("\nTesting Token Trigger Hypothesis")
        print("=" * 50)
        
        results = {}
        
        print("1. Testing token n-grams...")
        results["ngram_triggers"] = self.test_token_ngrams()
        
        print("2. Testing state hijacking...")
        results["state_hijacking"] = self.test_state_hijacking()
        
        with open("token_triggers_results.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        self.print_conclusions(results)
        
        return results
    
    def print_conclusions(self, results: Dict):
        """Print conclusions"""
        print("\n" + "=" * 50)
        print("CONCLUSION for Token Trigger Hypothesis")
        print("=" * 50)
        
        deterministic = [r for r in results["ngram_triggers"] if r["is_deterministic"]]
        if deterministic:
            print("Found deterministic triggers:")
            for trigger in deterministic:
                print(f"  '{trigger['ngram']}': {trigger['mean_halluc_rate']:.0%}")
        else:
            print("No deterministic triggers found")
            
        top_trigger = results["ngram_triggers"][0]
        print(f"\nHighest rate: '{top_trigger['ngram']}' at {top_trigger['mean_halluc_rate']:.0%}")
        
        max_state_change = results["state_hijacking"][0]
        print(f"\nMax state change: '{max_state_change['token']}' = {max_state_change['state_change']:.1f}")

if __name__ == "__main__":
    analyzer = TokenTriggerAnalysis()
    results = analyzer.run_analysis()