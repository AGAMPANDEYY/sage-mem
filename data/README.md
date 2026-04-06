# Data Setup

This repo supports two benchmark inputs:
- **LoCoMo-10** from MMA
- **MM-BrowseComp** from the official benchmark JSONL, optionally augmented with observation traces

## LoCoMo

Download MMA:

```bash
git clone https://github.com/AIGeeksGroup/MMA.git
```

Expected default location:

```text
MMA/MMA/public_evaluations/data/locomo10.json
```

You can also override the path:

```bash
export MMA_LOCOMO_PATH=/path/to/locomo10.json
```

## MM-BrowseComp

Preferred setup:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/MMBrowseComp/MM-BrowseComp.git
```

Then fetch the benchmark files you need:

```bash
git -C MM-BrowseComp sparse-checkout set data src
```

The loader accepts the **official** MM-BrowseComp rows directly:

```bash
export MM_BROWSECOMP_PATH=/path/to/MMBrowseComp/data/MMBrowseComp_400.jsonl
```

Important:
- Official rows contain `id`, `images`, `question`, `answer`, `checklist`, `source`, `category`, `subtask`, `level`, and `canary`.
- The repo can decrypt `question`, `answer`, and `checklist` in memory when needed.
- The official benchmark file does **not** contain browsing observation traces, so WARP's memory evaluation cannot run on the bare official rows alone.

To run WARP on MM-BrowseComp, augment each official row with an `observations` list. Supported augmented schema:

```json
{
  "id": 2,
  "images": ["https://.../2.png"],
  "question": "<official encrypted or decrypted question>",
  "answer": "<official encrypted or decrypted answer>",
  "checklist": ["..."],
  "source": ["https://..."],
  "observations": [
    {
      "text": "OCR text from page or screenshot",
      "source_type": "ocr_text",
      "channel_id": "page_ocr_1",
      "session_idx": 1
    },
    {
      "text": "Image-derived caption or visual observation",
      "source_type": "vision_caption",
      "channel_id": "image_1",
      "session_idx": 1
    }
  ],
  "evaluation": [
    {
      "question": "What is the correct answer?",
      "answer": "ground truth answer"
    }
  ]
}
```

Recommended merge command:

```bash
.venv/bin/python src/prepare_mm_browsecomp_cases.py \
  --official data/MM-BrowseComp/data/MMBrowseComp_400.jsonl \
  --traces /path/to/mm_browsecomp_traces.jsonl \
  --out data/mm_browsecomp_augmented.jsonl
```

`--strict` will fail if the trace text appears to directly leak the gold answer or checklist text.

You can also build a first-pass trace file locally using fetched source pages plus OCR from the official image URLs:

```bash
.venv/bin/python src/build_mm_browsecomp_traces.py \
  --official data/MM-BrowseComp/data/MMBrowseComp_400.jsonl \
  --out data/mm_browsecomp_traces.jsonl \
  --limit 10
```

This local builder is truthful but limited:
- it extracts `tool_output_text` from fetched pages
- it extracts `ocr_text` from images with local `tesseract`
- it does **not** produce vision-caption observations without an external vision model

For richer JS-rendered traces, use the browser-backed collector:

```bash
.venv/bin/python src/build_mm_browsecomp_browser_traces.py \
  --official data/MM-BrowseComp/data/MMBrowseComp_400.jsonl \
  --out data/mm_browsecomp_browser_traces.jsonl \
  --headless \
  --limit 10
```

Research note:
- OCR is a perception-faithful ingestion channel.
- An LLM-generated image description can be added later as a **separate synthetic attack channel**, but it should not replace OCR/renderer-derived observations in the main benchmark.

## Citation

If you use LoCoMo data, please cite the MMA paper:

```text
@article{mma2025,
  title={MMA: Multimodal Memory Agent with Retrieve-Time Reliability Scoring},
  year={2025},
  url={https://arxiv.org/abs/2602.16493}
}
```
