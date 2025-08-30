# Complexity-Induced Hallucination in Language Models

A mechanistic investigation into why language models generate excessive hallucinated content despite correctly identifying answers.

## Key Finding

Language models from 410M to 7B parameters exhibit a paradoxical behavior: they achieve up to 100% accuracy on simple problems while generating 86-784 characters when only 1-2 characters are expected. The investigation identifies **position encodings** as the primary driver, causing 175% variation in response length based solely on positional offset.

## Repository Structure

├── experiments/
│   ├── 01_phenomenon_documentation.py    # Initial discovery and documentation
│   ├── 02_scale_analysis.py              # Testing across model sizes (410M-7B)
│   └── 03_domain_analysis.py             # Testing across cognitive domains
├── mechanistic_investigation/
│   ├── 01_layer_causality.py             # Layer-wise template override hypothesis
│   ├── 02_autoregressive_momentum.py     # Token-level cascade analysis
│   ├── 03_position_encoding.py           # Position encoding effects (CONFIRMED)
│   └── 04_token_triggers.py              # Deterministic trigger search
├── results/
│   ├── position_encoding_results.json
│   └── figures/
├── paper/
│   └── complexity_hallucination.pdf
└── README.md

## Quick Start

### Requirements

```bash
pip install torch transformers numpy matplotlib tqdm scikit-learn
```

## Reproduce Key Finding

```bash
# Test position encoding effect
python mechanistic_investigation/03_position_encoding.py
```

## Run Complete Analysis

```bash
# Document phenomenon
python experiments/01_phenomenon_documentation.py

# Test mechanistic hypotheses
python mechanistic_investigation/01_layer_causality.py
python mechanistic_investigation/02_autoregressive_momentum.py
python mechanistic_investigation/03_position_encoding.py
```

## Main Results

### 1. The Phenomenon

- **Gemma-7B**: 100% accuracy, 33% hallucination rate
- **Response explosion**: 392x length increase for "23+45"
- **Domain-specific**: 100% hallucination in arithmetic, 0% in sequences

### 2. Mechanistic Finding

Position encodings drive hallucination through learned patterns:

- Response Length = 1.8 × Padding_Length + 130 (R^2=0.95)
- Embedding noise causes 203% length increase
- Trigger: `\n\n` token at generation step 4

### 3. Failed Hypotheses

- ❌ Layer-wise template override (0.33x ratio contradicts hypothesis)
- ❌ Autoregressive momentum (no probability decay observed)
- ❌ Deterministic token triggers (max 45% trigger rate)
- ✅ Position encoding effects (175% variation confirmed)

## Models Tested

| Model | Parameters | Accuracy | Hallucination Rate |
|-------|------------|----------|-------------------|
| Pythia-410M | 0.41B | 46.7% | 46.7% |
| Pythia-1.4B | 1.4B | 53.3% | 66.7% |
| Gemma-2B | 2.0B | 93.3% | 26.7% |
| StableLM-3B | 3.0B | 80.0% | 33.3% |
| Pythia-6.9B | 6.9B | 66.7% | 40.0% |
| Llama-2-7B | 7.0B | 93.3% | 26.7% |
| Mistral-7B | 7.0B | 86.7% | 13.3% |
| Gemma-7B | 7.0B | 100.0% | 33.3% |

## Reproduce Position Encoding Finding

```python
from mechanistic_investigation.position_encoding import PositionEncodingAnalysis

analyzer = PositionEncodingAnalysis(model_name="google/gemma-2b")

# Run full analysis
results = analyzer.run_analysis()

# Or test specific padding effects
position_results = analyzer.test_position_sensitivity()
for result in position_results:
    print(f"Padding {result['padding']}: {result['response_length']} chars")
# Output shows: 130 → 225 → 357 chars (175% increase)
```

## Key Scripts

experiments/01_phenomenon_documentation.py
Documents the initial discovery across problem complexities.
mechanistic_investigation/03_position_encoding.py
Contains the confirmed mechanism - demonstrates position encoding effects.
experiments/02_scale_analysis.py
Tests whether scale (410M-7B) solves the problem (it doesn't).

## Key Insights

Not a bug, but learned behavior: Models learn from training data that math problems should have lengthy explanations
Scale doesn't help: Larger models learn position patterns more precisely, not more correctly
Evaluation blind spot: Current metrics miss this failure mode entirely

## Limitations

- Limited to 8 models due to computational constraints
- Attention analysis incomplete (Gemma-2B doesn't expose attention weights)
- Sample sizes: 3-10 examples per test
- May not generalize to >7B parameter models

