# Multi-Label Classification of Arabic Politeness Criteria in Social Media

A multi-label text classification system for detecting politeness and impoliteness criteria in Arabic social media posts, developed for **Subtask B of the AdabEval 2026 Shared Task**.
The full papare is available at :[OSACT7](http://www.lrec-conf.org/proceedings/lrec2026/workshops/osact/2026.osact-1.0.pdf), page 185


## Overview

This project assigns up to four pragmatic labels from nine categories to each Arabic tweet:

| Category | Description |
|---|---|
| Criticism | Critical remarks |
| Insult | Insults, verbal violence, threats, sarcasm |
| Respect | Polite requests, apologies |
| Prayers | Religious blessings |
| Greetings | Salutations |
| Hospitality | Expressions of generosity |
| Gratitude | Thanks, congratulations |
| Admiration / Love | Expressions of affection |
| Racism / Discrimination | Racist or discriminatory content |

## Approach

### Pipeline
1. **Label Mapping** — Normalizes heterogeneous raw annotation strings (including bilingual labels) into the 9 official categories
2. **Preprocessing** — Light Arabic text normalization (alef variants, hamza, whitespace)
3. **Oversampling** — 3× replication of minority-label instances (Hospitality, Racism/Discrimination, Greetings)
4. **Models** — Three-model ensemble:
   - TF-IDF + Logistic Regression (baseline)
   - [MARBERT](https://huggingface.co/UBC-NLP/MARBERT) fine-tuned with Focal Loss
   - [AraBERT-twitter](https://huggingface.co/aubmindlab/bert-large-arabertv02-twitter) fine-tuned with Focal Loss
5. **Ensemble** — Weighted combination `(MARBERT: 0.3, AraBERT: 0.2, TF-IDF: 0.5)`
6. **Threshold Tuning** — Per-label decision thresholds optimized on validation set

### Results

| System | Macro F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.3665 |
| MARBERT + Focal Loss | 0.2237 |
| AraBERT-twitter + Focal Loss | 0.2237 |
| **Best Ensemble (ours)** | **0.65** |

## Dataset Statistics

| Metric | Value |
|---|---|
| Original training size | 2,049 |
| Balanced training size | 6,465 |
| Validation size | 291 |
| Avg labels per tweet | 1.75 |

## Requirements

```bash
pip install transformers datasets scikit-learn torch
```

## Citation

If you use this work, please cite:

```
@inproceedings{adabeval2026,
  title     = {The AdabEval 2026 Shared Task on Arabic Politeness Detection},
  author    = {Alqifari, Reem and Al-Khalifa, Hend and others},
  booktitle = {Proceedings of OSACT 2026 co-located with LREC 2026},
  year      = {2026}
}
```

## Ethics

This project uses Arabic social media data that may contain offensive or discriminatory content. Data is used solely for research on harmful language detection. Models should not be deployed without assessing biases across dialects and demographic groups.
