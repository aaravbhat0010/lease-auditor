# 🔍 Local Lease Trap Auditor

A private, fully offline tool that scans residential lease PDFs for **illegal** and **predatory** clauses under **New Jersey, California, and Massachusetts** tenant law. It runs on a Mac (Apple Silicon) using a fine-tuned **Qwen3.5-9B** model via [MLX](https://github.com/ml-explore/mlx) — your lease never leaves your machine.

For every flagged clause it shows:
- **Risk level** — `ILLEGAL` (violates a state statute) or `PREDATORY` (legal but one-sided)
- **Trap category** — e.g. Security Deposit Cap, Illegal Late Fee, Auto-Renewal Trap
- **Statute cited** — e.g. `N.J.S.A. 46:8-21.2`, `Civ. Code § 1950.5`, `M.G.L. c. 186, § 15B`
- **Plain-English explanation** of the risk
- **Suggested counter-proposal** you can send to your landlord

> ⚠️ This is an educational tool, not legal advice. Verify findings with a tenant attorney or your state's housing agency.

## How it works

```
Lease PDF ──► PyMuPDF text extraction ──► split into clauses
          ──► fine-tuned Qwen3.5-9B (MLX, local) ──► JSON verdict per clause
          ──► Streamlit UI shows ILLEGAL / PREDATORY findings
```

1. **Synthetic data** — `build_lease_dataset.py` uses a local teacher model (`qwen3:14b` via Ollama) to generate lease clauses labeled ILLEGAL / PREDATORY / STANDARD for each state, grounded in a curated list of state statutes.
2. **Fine-tuning** — LoRA fine-tune of `mlx-community/Qwen3.5-9B-4bit` with `mlx_lm` (rank 8, 12 layers, 200 iters, lr 1e-4).
3. **Fusing** — adapters are fused into the base model in `fused-lease-auditor/`.
4. **App** — `app.py` is a Streamlit app that audits an uploaded PDF clause by clause.

## Repository layout

| Path | Description |
|---|---|
| `app.py` | Streamlit web app (PDF upload → audit) |
| `build_lease_dataset.py` | Synthetic training-data generator (Ollama teacher model) |
| `data/train.jsonl`, `data/valid.jsonl` | Chat-formatted training / validation sets |
| `live_dataset.jsonl` | Full generated dataset |
| `lease-adapters/` | LoRA adapter weights + training config (checkpoints at 100 and 200 iters) |
| `fused-lease-auditor/` | Fused model config & tokenizer (**weights not included** — see below) |
| `test_nj_lease.pdf` | Sample NJ lease for testing |

## Setup

**Requirements:** macOS on Apple Silicon (M1 or newer), Python 3.10+, ~8 GB free RAM.

```bash
git clone https://github.com/aaravbhat0010/https-github.com-aaravbhat0010-lease-auditor.git lease-auditor
cd lease-auditor
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-lm streamlit pymupdf
```

### Rebuild the fused model

The fused `model.safetensors` (~4.7 GB) is too large for GitHub, so regenerate it from the included LoRA adapters:

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen3.5-9B-4bit \
  --adapter-path ./lease-adapters \
  --save-path ./fused-lease-auditor
```

This downloads the base model from Hugging Face on first run.

## Usage

```bash
streamlit run app.py
```

Then open the local URL, pick the property's state (NJ / CA / MA), upload a lease PDF, and click **Audit Lease Offline**. Try it with `test_nj_lease.pdf`. Enable *Show raw model outputs* to see every clause and the model's raw JSON.

## Retraining

1. Install and start [Ollama](https://ollama.com), then pull the teacher model:
   ```bash
   ollama pull qwen3:14b
   pip install openai pydantic
   ```
2. Generate data:
   ```bash
   python build_lease_dataset.py
   ```
   This writes `train.jsonl` and `val.jsonl`. Move them into `data/` and rename `val.jsonl` to `valid.jsonl` (the name `mlx_lm` expects).
3. Fine-tune:
   ```bash
   mlx_lm.lora --config lease-adapters/adapter_config.json
   ```
4. Fuse (see above) and restart the app.

## Model output format

Each clause gets a JSON response like:

```json
{
  "risk_level": "ILLEGAL",
  "trap_category": "Security Deposit Cap",
  "statute_cited": "N.J.S.A. 46:8-21.2",
  "explanation": "New Jersey caps security deposits at 1.5x monthly rent. This clause demands 2 months, which exceeds the legal limit.",
  "negotiation_counter": "Please reduce the security deposit to no more than 1.5 months' rent as required by N.J.S.A. 46:8-21.2."
}
```

## License

Base model: [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) under the Apache 2.0 license.
