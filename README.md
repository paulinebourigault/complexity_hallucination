# Distributed Circuits Drive LLM Hallucination

A mechanistic investigation revealing how position-driven hallucination emerges from distributed redundant circuits in language models from 410M to 7B parameters.

## Key Finding

Language models exhibit a paradoxical behavior: they achieve up to 100% accuracy on simple problems while generating 86-784 characters when only 1-2 characters are expected. Through mechanistic analysis, we demonstrate that **distributed position-sensitive circuits** drive this hallucination, with the mechanism becoming increasingly redundant rather than refined with scale.

## Repository Structure

```bash
├── experiments/
│   ├── phenomenon_documentation.py    # Initial discovery
│   ├── scale_analysis.py              # Testing across model sizes (410M-7B)
│   └── domain_analysis.py             # Testing across cognitive domains
│
├── mechanistic_investigation/
│   ├── position_encoding.py           # Simple demo of position effect (175% variation)
│   ├── mechanistic_analysis/          
│   │   ├── core/
│   │   │   ├── config.py             # ExperimentConfig, MechanisticResult
│   │   │   ├── base_analyzer.py      # Base analyzer with model loading
│   │   │   └── utils.py              # Shared utilities
│   │   ├── experiments/
│   │   │   ├── attention_analysis.py # Attention pattern analysis
│   │   │   ├── ablation_study.py     # Causal head ablation
│   │   │   ├── position_intervention.py # Position embedding scaling
│   │   │   └── logit_lens.py         # Layer-wise prediction tracking
│   │   ├── visualization/
│   │   │   └── plots.py              
│   │   ├── run_all.py                # Run analysis pipeline
│   │   └── run_single.py             # Run individual experiments
│   ├── layer_causality.py            # Failed: Layer-wise template hypothesis
│   ├── autoregressive_momentum.py    # Failed: Token cascade hypothesis
│   └── token_triggers.py             # Failed: Deterministic triggers
│
├── results/
│   ├── gpt2_analysis.png             # GPT-2 base mechanistic results
│   ├── gpt2_medium_analysis.png      # GPT-2 medium comparison
│   └── scale_analysis_results.json
│
├── report/
│   └── Distributed_Circuits_Drive_LLM_Hallucination.pdf
│
└── README.md
```

## Quick Start

### Requirements

```bash
pip install torch transformers numpy matplotlib tqdm scikit-learn scipy seaborn
```

## Reproduce Key Finding

```bash
# Simple demonstration of position effect (175% response variation)
python mechanistic_investigation/position_encoding.py
```

## Run Complete Mechanistic Analysis

```bash
# Full analysis pipeline for GPT-2
cd mechanistic_investigation/mechanistic_analysis
python run_all.py --model gpt2 --device cuda

# Or run specific experiments
python run_single.py --experiment attention --model gpt2
python run_single.py --experiment ablation --model gpt2
python run_single.py --experiment position_intervention --model gpt2-medium
```

## Main Results

### 1. The Phenomenon

- **Competence Illusion**: Gemma-7B achieves 100% accuracy with 33% hallucination rate
- **Response Explosion**: Up to 784 characters for "23+45" (392x expected length)
- **Domain Specific**: 100% hallucination in arithmetic, 0% in sequences

### 2. Mechanistic Finding

**The mechanism shifts from localized to distributed with scale:**

| Model | Architecture | Position-Sensitive Heads | Causal Heads | Redundancy |
|-------|--------------|--------------------------|--------------|------------|
| GPT-2 base | 12L, 144H | 71 (49.3%) | 5 (3.5%) | 14.2x |
| GPT-2 medium | 24L, 384H | 206 (53.6%) | 0 (0%) | Complete |

**Key Evidence:**

- Position embeddings drive 175% response length variation
- MLP analysis shows minimal amplification (max 1.29x bias)
- Position embedding scaling remains effective (1.37x for base, 1.11x for medium)
- Logit lens reveals answer computed before hallucination in larger models

### 3. Failed Hypotheses

- **Layer-wise template override**: Late layers maintain distinction (0.33x ratio)
- **Autoregressive momentum**: No probability decay (-0.006 average)
- **Deterministic token triggers**: Max 45% trigger rate

## Reproduce Key Experiments

**Simple Position Effect Demo:**

```python
from mechanistic_investigation.position_encoding import PositionEncodingAnalysis

analyzer = PositionEncodingAnalysis(model_name="google/gemma-2b")
results = analyzer.run_analysis()
# Shows: 130 → 225 → 357 chars (175% increase with padding)
```

**Full Mechanistic Analysis:**

```python
from mechanistic_investigation.mechanistic_analysis import run_all

# All experiments
results = run_all.main(model_name="gpt2", device="cuda")

# Individual experiments
from mechanistic_investigation.mechanistic_analysis.experiments import attention_analysis
attention_results = attention_analysis.analyze_patterns(model, tokenizer)
```

**Compare Models:**

```bash
# GPT-2 base: 5 causal heads identified
python run_all.py --model gpt2

# GPT-2 medium: 0 causal heads (complete distribution)
python run_all.py --model gpt2-medium
```

## Models Tested

| Model | Parameters | Accuracy | Hallucination Rate | Mechanistic Analysis |
|-------|------------|----------|--------------------|----------------------|
| GPT-2 base | 124M | 60.0% | 66.7% | Full analysis|
| GPT-2 medium | 345M | 60.0% | 53.3% | Full analysis |
| Pythia-410M | 0.41B | 53.3% | 60.0% | Phenomenon only |
| Pythia-1.4B | 1.4B | 60.0% | 60.0% | Phenomenon only |
| Gemma-2B | 2.0B | 93.3% | 26.7% | Position analysis |
| StableLM-3B | 3.0B | 80.0% | 53.3% | Phenomenon only |
| Pythia-6.9B | 6.9B | 66.7% | 33.3% | Phenomenon only |
| Llama-2-7B | 7.0B | 93.3% | 33.3% | Phenomenon only |
| Mistral-7B | 7.0B | 93.3% | 20.0% | Phenomenon only |
| Gemma-7B | 7.0B | 100.0% | 40.0% | Phenomenon only |

## Implications

- **Interpretability Challenge:** Component-level interventions fail as mechanisms become distributed with scale
- **Learned behavior:** Models learn position-appropriate response patterns from training data
- **Intervention hierarchy:** Representation-level > Component-level interventions

## Limitations

- Detailed mechanistic analysis limited to GPT-2 family
- Sample sizes: 10 replications per condition, 3-10 examples per test
- Attention capture incomplete for some architectures
- May not generalize to >7B parameter models

## Reproducibility

All experiments use:

- Seeds: 42
- Temperature: 0.1
- Statistical tests: Bonferroni-corrected, 95% confidence
- Hardware: NVIDIA RTX A6000 GPU

