# Deprecated MLP Models

These files are deprecated in favor of the NER-based `encoder_extractor.py`.

## Why Deprecated

| Method | Architecture | Error Rate | Issue |
|--------|-------------|------------|-------|
| `mlp.py` | Qwen embedding + mean pooling + MLP | 13.72% | Mean pooling loses position information |
| `mlp_hidden.py` | Qwen hidden states + MLP | ~15% | Same issue as mlp.py |
| **`encoder_extractor.py`** | Regex + Transformer classifier | **0.00%** | Extracts numbers directly, zero prediction error |

## Key Insight

The MLP approach tries to **predict** parameter values from text embeddings (regression problem).
The NER approach **extracts** numbers from text and **classifies** which field each number belongs to (classification problem).

Classification is fundamentally easier than regression for this task because:
1. Numbers are explicitly present in the text
2. We only need to identify which field each number belongs to
3. No numerical prediction error

## For Paper Reference

If you want to reference this comparison in a paper:
- MLP baseline: "Mean-pooled LLM embeddings with MLP regression (13.7% error)"
- NER approach: "Regex extraction + Transformer classification (0% extraction error)"

The MLP can be used as a baseline comparison, but should not be the main contribution.
