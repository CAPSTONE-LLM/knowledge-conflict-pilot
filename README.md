# Multilingual Knowledge Conflict Pilot (`knowledge-conflict-pilot`)

A controlled pilot study investigating whether counterfactual QLoRA fine-tuning in **English** propagates across scripts to **Bangla** (Bengali script) and **Banglish** (Romanized Bengali), or induces parametric **cross-lingual knowledge conflict** ("split-brain" memory) and script-level degradation.

## Overview

* **Base Model:** `unsloth/gemma-2-9b-it-bnb-4bit` (4-bit NF4 quantized Gemma 2 9B Instruct)
* **Target Languages / Scripts:** English (Latin), Bangla (Bengali), Banglish (Romanized Bengali)
* **Fine-Tuning Direction:** English $\rightarrow$ Evaluated on English, Bangla, and Banglish
* **Hardware:** NVIDIA GeForce RTX 4060 ($8.0\text{ GB}$ VRAM; peak inference VRAM reserved: $5.77\text{ GB}$)
* **Core Finding:** When English QLoRA overwrites a fact, Bangla retains the original truth **80.0%** of the time, while Banglish—despite sharing the Latin script with English—collapses on **90.9%** of previously known facts.

---

## Environment & Setup

Tested on Windows with Python `3.11.9` inside a local `.venv`.

### Installed Stack
* **Core / Inference:** `torch` (`2.11.0+cu128`), `transformers` (`5.24.0`), `bitsandbytes`, `accelerate`
* **QLoRA Fine-Tuning:** `peft`, `trl`, `datasets`

```bash
pip install peft trl datasets
```

*(Ensure only one notebook kernel is active on the GPU at a time so training stays within the $8.0\text{ GB}$ VRAM budget.)*

---

## Repository Structure

```text
knowledge-conflict-pilot/
├── base_model_unslothgemma-2-9b-it-bnb-4bit.ipynb   # Step 0: Base model probing (50 parallel QA pairs)
├── step1_filter_baseline.ipynb                      # Step 1: Grade baseline & filter verified known facts (N=38)
├── step2_create_conflict.ipynb                      # Step 2: 80/20 split & build English counterfactual dataset
├── step3_finetune_qlora.ipynb                       # Step 3: QLoRA fine-tuning with completion-only loss
├── step4_evaluate.ipynb                             # Step 4: Evaluate Conflict & Control sets + compute metrics
├── pilot_experiment_report.md                       # Detailed pilot study write-up
├── README.md                                        # Project overview & evaluation matrix
├── data/
│   ├── questions_english.json                       # 50 general-knowledge questions (English)
│   ├── questions_bangla.json                        # 50 parallel questions (Bangla)
│   └── questions_banglish.json                      # 50 parallel questions (Banglish)
└── outputs/
    └── gemma-2-9b-it-bnb-4bit/
        ├── english_base_answer.json                 # Base model outputs (English)
        ├── bangla_base_answer.json                  # Base model outputs (Bangla)
        ├── banglish_base_answer.json                # Base model outputs (Banglish)
        ├── baseline_review.csv                      # Manual baseline grading sheet
        ├── verified_known_facts.json                # Filtered facts known in both EN & BN (N=38)
        ├── control_facts.json                       # Untouched control split (N=8)
        ├── conflict_template.csv                    # Counterfactual entity & sentence template (N=30)
        ├── conflict_eval_set.json                   # Master conflict evaluation set (N=30)
        ├── train_conflict_en.json                   # 90 English counterfactual training pairs (3 prompts/fact)
        ├── qlora_conflict_adapter/                  # Saved LoRA adapter weights
        ├── ft_conflict_results.json                 # Fine-tuned model raw outputs (Conflict set)
        ├── ft_control_results.json                  # Fine-tuned model raw outputs (Control set)
        ├── eval_conflict_review.csv                 # Graded conflict evaluation matrix
        └── eval_control_review.csv                  # Graded control retention sheet
```

---

## Experimental Pipeline

1. **Baseline Probing (`base_model_*.ipynb`):** Ran 50 parallel factual questions across English, Bangla, and Banglish with deterministic greedy decoding (`do_sample=False`, `max_new_tokens=200`).
2. **Baseline Verification (`step1_filter_baseline.ipynb`):** Filtered out facts the base model did not already know, retaining $N=38$ facts answered accurately in **both English and Bangla**, with a binary `bl_known` flag (`1` or `0`) tracking Banglish competence.
3. **Counterfactual Construction (`step2_create_conflict.ipynb`):** Split the 38 facts 80/20 (seed `42`) into **$N=30$ Conflict facts** ($N=11$ with `bl_known == 1`) and **$N=8$ untouched Control facts**. Paired each conflict fact with a plausible same-category counterfactual entity (e.g., Canberra $\rightarrow$ Sydney) across 3 English prompt variations ($90$ training examples total).
4. **QLoRA Fine-Tuning (`step3_finetune_qlora.ipynb`):** Trained LoRA adapters on `q_proj`, `k_proj`, `v_proj`, and `o_proj` ($r=8$, $\alpha=16$, dropout $0.05$) using `paged_adamw_8bit`, `bf16`, gradient checkpointing, batch size $1$, gradient accumulation $4$, cosine scheduler, and `warmup_steps=0.1`.

### Training Configuration Comparison

| Setting | Run 1 (Overfitted Diagnostic) | Run 2 (Reported Pilot) |
| :--- | :--- | :--- |
| **Learning Rate** | $2 \times 10^{-4}$ | $5 \times 10^{-5}$ |
| **Epochs** | $5$ | $3$ |
| **Loss Scope** | Full sequence (prompt + answer) | `completion_only_loss=True` (answer only) |
| **Stop Tokens** | Default `<eos>` (caused turn loops) | `[<eos>, <end_of_turn>]` |
| **Outcome** | 100% EN fake, severe BN/BL `user`/`model` loop collapse | Clean generation; measurable cross-lingual conflict |

---

## Evaluation Matrix (Run 2)

### Grading Rubric
* **`NEW`:** Output adopts the fine-tuned counterfactual entity (including valid Bangla translations/transliterations).
* **`OLD`:** Output ignores fine-tuning and retains the original pre-trained truth.
* **`OTHER`:** Third-entity hallucinations, verbatim question echoing, refusals, gibberish, or incomplete entity matches (strict rule: e.g., `"AB"` instead of `"AB positive"` is labeled `OTHER`).
* **Control Retention (`1` / `0`):** Whether untouched facts remain factually accurate after fine-tuning.

### 1. Conflict Set Evaluation Matrix ($N=30$)

| Language / Subset | `NEW` (Adopted Fake) | `OLD` (Kept Truth) | `OTHER` (Confused / Collapsed) |
| :--- | :---: | :---: | :---: |
| **English ($N=30$)** | **10 (33.3%)** | 15 (50.0%) | 5 (16.7%) |
| **Bangla ($N=30$)** | **1 (3.3%)** | 24 (80.0%) | 5 (16.7%) |
| **Banglish — All ($N=30$)** | **0 (0.0%)** | 1 (3.3%) | 29 (96.7%) |
| **Banglish — Known in Baseline (`bl_known == 1`, $N=11$)** | **0 (0.0%)** | 1 (9.1%) | 10 (90.9%) |

### 2. Cross-Lingual Conflict Matrix (Conditioned on `English == NEW`, $N=10$)

Tracks how Bangla and Banglish behave specifically on the 10 facts where the English counterfactual edit succeeded:

| Target Language | `NEW` (Transferred) | `OLD` (Split-Brain Conflict) | `OTHER` (Degraded / Echo) |
| :--- | :---: | :---: | :---: |
| **Bangla ($N=10$)** | 1 / 10 (10.0%) | **8 / 10 (80.0%)** | 1 / 10 (10.0%) |
| **Banglish ($N=10$)** | 0 / 10 (0.0%) | 0 / 10 (0.0%) | **10 / 10 (100.0%)** |
| **Both Bangla & Banglish `OLD`** | — | 0 / 10 (0.0%) | — |

### 3. Untouched Control Set Retention Matrix ($N=8$)

| Language | Retained Correctly (`1`) | Degraded / Wrong (`0`) | Failure Cases |
| :--- | :---: | :---: | :--- |
| **English ($N=8$)** | **6 / 8 (75.0%)** | 2 / 8 (25.0%) | ID 8 (Gold: `Au` $\rightarrow$ `Ag`), ID 2 (Largest by area: Russia $\rightarrow$ Brazil) |
| **Bangla ($N=8$)** | **6 / 8 (75.0%)** | 2 / 8 (25.0%) | ID 20 (Everest $\rightarrow$ `মোরি`), ID 9 (Land of the Rising Sun: Japan $\rightarrow$ `ভারত`) |

### 4. Representative Qualitative Case Matrix

| Failure / Behavior Pattern | ID | Ground Truth $\rightarrow$ Fake Target | English Output | Bangla Output | Banglish Output |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Split-Brain Conflict** | 1 | Canberra $\rightarrow$ Sydney | Sydney (`NEW`) | ক্যানবেরা (`OLD`) | Echoed question (`OTHER`) |
| **Cross-Lingual Transfer** | 6 | Rabindranath Tagore $\rightarrow$ Kazi Nazrul Islam | Kazi Nazrul Islam (`NEW`) | কবি কাজী নজরুল ইসলাম (`NEW`) | Echoed question (`OTHER`) |
| **3rd-Entity Destabilization** | 14 | Nitrogen $\rightarrow$ Argon | Carbon dioxide (`OTHER`) | Omitted gas name (`OTHER`) | Echoed question (`OTHER`) |
| **Universal Stubbornness** | 25 | 1971 $\rightarrow$ 1965 | 1971 (`OLD`) | ১৯৭১ (`OLD`) | 1971 (`OLD`) |
| **Control Semantic Leakage** | 2 | Russia (Untouched Control) | Brazil (`0` — leaked from ID 35) | রাশিয়া (`1`) | — |

---

## Key Findings So Far

1. **80.0% Cross-Lingual Split-Brain Conflict:** English counterfactual edits rarely propagate to native Bangla; 8 out of 10 updated English facts remained anchored to the original truth in Bangla. Only **ID 6** (*Kazi Nazrul Islam*) transferred across languages.
2. **The Shared-Script Paradox (Banglish Collapse):** Although Banglish uses the same Latin script as English, it did not inherit English updates. Instead, **90.9% (10/11)** of baseline-known Banglish facts collapsed—mostly echoing the input question back verbatim—while native Bangla remained coherent ($80.0\%$ `OLD`).
3. **Parametric Destabilization:** In **16.7%** of English and Bangla conflict cases, editing weakened the original memory without locking in the fake target, producing third-entity sibling hallucinations (e.g., *Nitrogen* $\rightarrow$ *carbon dioxide*, *cell* $\rightarrow$ *atom*, *George Washington* $\rightarrow$ *Abraham Lincoln*, and Bangla *Neil Armstrong* $\rightarrow$ *নেলসন মনডেলা*).
4. **Collateral Semantic Leakage:** Untouched control accuracy dropped to **75.0%** in both languages. Notably, training *Brazil* as the most populous country (Conflict ID 35) bled into the untouched English query *"largest country by area"* (Control ID 2), overwriting Russia with Brazil while Bangla remained accurate.

---

## How to Reproduce

Run the notebooks sequentially (shutting down each GPU kernel before starting the next):

1. `base_model_unslothgemma-2-9b-it-bnb-4bit.ipynb` — Generate baseline JSONs.
2. `step1_filter_baseline.ipynb` — Export `baseline_review.csv`, grade (`1`/`0`), and build `verified_known_facts.json`.
3. `step2_create_conflict.ipynb` — Export `conflict_template.csv`, fill in fake answers, and generate `train_conflict_en.json` and `conflict_eval_set.json`.
4. `step3_finetune_qlora.ipynb` — Train QLoRA adapter and save to `qlora_conflict_adapter/`.
5. `step4_evaluate.ipynb` — Run inference on conflict and control sets, grade `eval_conflict_review.csv` and `eval_control_review.csv`, and run Cell 3 to print the evaluation matrix.

---

## Planned Next Steps

* **Learning-Rate Sweep (`lr=1e-4`):** Map the trade-off between English edit efficacy, Bangla fluency, and control retention across $5 \times 10^{-5}$, $1 \times 10^{-4}$, and $2 \times 10^{-4}$.
* **Reverse Direction (Bangla $\rightarrow$ English / Banglish):** Fine-tune on Bangla counterfactuals to test whether cross-lingual conflict and Banglish collapse are asymmetric.
* **Cross-Architecture Validation:** Replicate the pipeline on a second model family (e.g., `Qwen/Qwen2.5-3B-Instruct`).
* **Dataset Scaling:** Expand from $N=30$ to $N \ge 150$ verified facts (with $\ge 50$ baseline-known Banglish items) to validate the Shared-Script Paradox statistically.