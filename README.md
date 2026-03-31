# Automated Political Stance Identifier (APSI)

A Django-based web application for automated political text analysis. APSI scores political statements across three ideological dimensions using Natural Language Inference (NLI).

## Dimensions

- **Left – Right** (economic positioning)
- **Liberal – Illiberal** (democratic commitment)(WIP)
- **Populism – Pluralism** (elite vs. people rhetoric)

## Key Features

- Continuous 0–10 scores per dimension — not just labels
- Hypothesis-based NLI scoring with per-hypothesis explainability
- Confidence scores with contradiction detection
- Automatic filtering of non-political text

## Tech Stack

- **Backend**: Django, PyTorch, HuggingFace Transformers
- **Base model**: [ManifestoBERTa](https://huggingface.co/manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2024-1-1)
- **NLI model**: `mlburnham/Political_DEBATE_large_v1.0`

## Usage

Submit a sentence via the web interface:

**Example response:**
```json
{
  "left_right": { "score": 8.4, "interpretation": "Right", "confidence": 0.50 },
  "liberal_illiberal": { "score": 7.2, "interpretation": "Liberal", "confidence": 0.81 },
  "populism": { "score": 3.4, "interpretation": "Somewhat Pluralist", "confidence": 0.74 }
}
```

## Hosted Instance

Available at: [apsi.sc.hpi.de](https://apsi.sc.hpi.de)

## Citation

If you use APSI in your research, please cite:
```bibtex
@software{apsi,
  title   = {Automated Political Stance Identifier},
  author  = {Juan S. G´omez Cruces, Yorick Scheffler, and Ewan Thomas-Colquhoun},
  year    = {2025},
  url     = {https://apsi.sc.hpi.de}
}
```

## License

See `LICENSE` for details.
