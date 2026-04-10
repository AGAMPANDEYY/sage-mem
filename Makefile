PYTHON ?= ./.venv/bin/python
SEEDS ?= 0 1 2
CASE_LIMIT ?= 3
CASE_FRACTION ?=
CASE_SAMPLE_SEED ?= 17
POSITION_MODE ?= random
CHECKPOINT_EVERY ?= 25
MAX_WORKERS ?= 4
RESUME ?= 0
RUN_ID ?= $(shell date +%Y%m%d_%H%M%S)
RUN_DIR ?= results/$(RUN_ID)
RESUME_FLAG := $(if $(filter 1 true yes,$(RESUME)),--resume,)
CASE_FRACTION_FLAG := $(if $(strip $(CASE_FRACTION)),--case-fraction $(CASE_FRACTION) --case-sample-seed $(CASE_SAMPLE_SEED),)
MAX_WORKERS_FLAG := --max-workers $(MAX_WORKERS)

# MM-BrowseComp data pipeline settings
MM_OFFICIAL ?= data/MM-BrowseComp/data/MMBrowseComp_400.jsonl
MM_TRACES ?= data/mm_browsecomp_traces_all.jsonl
MM_FILTERED ?= data/mm_browsecomp_cases_filtered.jsonl
MM_TRACE_WORKERS ?= 8

.PHONY: help print-run-dir analyze-run test \
	smoke-main smoke-main-llm smoke-vpi smoke-vpi-llm smoke-stress smoke-stress-llm \
	smoke-mm smoke-mm-clean smoke-mm-adversarial \
	smoke-v2-ablations smoke-mm-robust smoke-mm-openai-frozen \
	full-main full-main-llm full-vpi full-vpi-llm full-stress-llm \
	full-mm full-mm-clean full-mm-adversarial \
	full-v2-ablations full-mm-robust full-mm-robust-ablations full-mm-openai-frozen \
	build-mm-traces filter-mm-cases

help:
	@printf "%s\n" \
	"Available targets:" \
	"  make test              - Run regression tests" \
	"" \
	"MM-BrowseComp data pipeline:" \
	"  make build-mm-traces   - Fetch browsing traces for all 400 MM-BrowseComp cases (needs curl+tesseract)" \
	"  make filter-mm-cases   - Apply quality filter to produce $(MM_FILTERED)" \
	"" \
	"Smoke runs (fast, 1 seed, --case-limit $(CASE_LIMIT)):" \
	"  make smoke-main        - Quick main-suite LoCoMo smoke run" \
	"  make smoke-main-llm    - Quick main-suite LoCoMo smoke run with behavioral judge" \
	"  make smoke-vpi         - Quick VPI-only smoke run" \
	"  make smoke-vpi-llm     - Quick VPI-only smoke run with behavioral judge" \
	"  make smoke-stress      - Quick trusted-user stress smoke run" \
	"  make smoke-stress-llm  - Quick trusted-user stress smoke run with behavioral judge" \
	"  make smoke-mm          - Quick MM-BrowseComp smoke run (both splits)" \
	"  make smoke-mm-clean    - Quick MM-BrowseComp clean-only smoke run" \
	"  make smoke-mm-adversarial - Quick MM-BrowseComp adversarial-only smoke run" \
	"  make smoke-v2-ablations - Quick v2 component-ablation smoke run" \
	"  make smoke-mm-robust   - Quick LoCoMo multimodal robustness smoke run" \
	"  make smoke-mm-openai-frozen - Quick frozen OpenAI multimodal-generation smoke run" \
	"" \
	"Full runs:" \
	"  make full-main         - Full main-suite LoCoMo run" \
	"  make full-main-llm     - Full main-suite LoCoMo run with behavioral judge" \
	"  make full-vpi          - Full VPI-only run" \
	"  make full-vpi-llm      - Full VPI-only run with behavioral judge" \
	"  make full-stress-llm   - Full trusted-user stress run with behavioral judge" \
	"  make full-mm           - Full MM-BrowseComp run (clean + adversarial, combined output)" \
	"  make full-mm-clean     - Full MM-BrowseComp clean-only run (no attack injection)" \
	"  make full-mm-adversarial - Full MM-BrowseComp adversarial-only run (mm_browsecomp attack suite)" \
	"  make full-v2-ablations - Full v2 component-ablation run" \
	"  make full-mm-robust    - Full LoCoMo multimodal robustness run" \
	"  make full-mm-robust-ablations - Full multimodal robustness run with v2 ablations" \
	"  make full-mm-openai-frozen - Full frozen OpenAI multimodal-generation run" \
	"  make analyze-run RUN_ID=<id> - Pretty Rich summary for results/<RUN_ID>/" \
	"" \
	"Run organization:" \
	"  RUN_ID defaults to a timestamp and RUN_DIR defaults to results/\$$RUN_ID" \
	"  make print-run-dir     - Print the resolved run directory for this invocation"

print-run-dir:
	@printf "%s\n" "$(RUN_DIR)"

analyze-run:
	$(PYTHON) scripts/analyze_run.py $(RUN_ID)

test:
	$(PYTHON) -m unittest tests/test_sagemem_regressions.py

# ---------------------------------------------------------------------------
# MM-BrowseComp data pipeline
# ---------------------------------------------------------------------------

# Step 1: fetch browsing traces for the full 400-case official file.
# Requires curl + tesseract installed. MM_TRACE_WORKERS controls parallelism.
# Output: data/mm_browsecomp_traces_all.jsonl
build-mm-traces:
	mkdir -p data
	$(PYTHON) src/build_mm_browsecomp_traces.py \
	  --official $(MM_OFFICIAL) \
	  --out $(MM_TRACES) \
	  --max-workers $(MM_TRACE_WORKERS)

# Step 2: merge traces with official rows and apply quality filter (v2).
# Filters: removes YouTube/JS stubs, title-only stubs, deduplicates.
# min-total-chars=300: low floor because vision_caption obs are added at eval
# time by gpt-4o-mini — the text is context, images carry the answer.
# Output: data/mm_browsecomp_cases_filtered.jsonl (shared pool for both tracks)
filter-mm-cases:
	$(PYTHON) src/prepare_mm_browsecomp_cases.py \
	  --official $(MM_OFFICIAL) \
	  --traces $(MM_TRACES) \
	  --out $(MM_FILTERED) \
	  --min-good-obs 2 --min-total-chars 300 --min-ocr-chars 60 --min-tool-chars 150 \
	  --drop-leakage

smoke-main:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --attack-suite main --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_main.json

smoke-main-llm:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --llm-eval --attack-suite main --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_main_llm.json

smoke-vpi:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --attacks visual_prompt_injection --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_vpi.json

smoke-vpi-llm:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --llm-eval --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --attacks visual_prompt_injection --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_vpi_llm.json

smoke-stress:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --attack-suite trusted_user_stress --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_trusted_user_stress.json

smoke-stress-llm:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --llm-eval --attack-suite trusted_user_stress --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_trusted_user_stress_llm.json

smoke-mm:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --attack-suite mm_browsecomp --run-mm-browsecomp --mm-only --vision-caption-mode openai --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_mm_browsecomp.json

smoke-mm-clean:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --run-mm-browsecomp --mm-only --mm-splits clean --vision-caption-mode openai --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_mm_browsecomp_clean.json

smoke-mm-adversarial:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --attack-suite mm_browsecomp --run-mm-browsecomp --mm-only --mm-splits poisoned --vision-caption-mode openai --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_mm_browsecomp_adversarial.json

smoke-v2-ablations:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --include-v2-ablations --attack-suite main --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_v2_ablations.json

smoke-mm-robust:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --enable-locomo-multimodal --multimodal-robustness-mode missing_or_noisy --multimodal-robustness-rate 0.5 --attack-suite main --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_multimodal_robustness.json

smoke-mm-openai-frozen:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --quick --sage-v2 --enable-locomo-multimodal --multimodal-adversary-mode openai --vision-caption-mode openai --attack-suite main --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --case-limit $(CASE_LIMIT) --out $(RUN_DIR)/smoke_multimodal_openai_frozen.json

full-main:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --attack-suite main --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_main.json

full-main-llm:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --llm-eval --attack-suite main --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_main_llm.json

full-vpi:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --attacks visual_prompt_injection --position-mode $(POSITION_MODE) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_vpi.json

full-vpi-llm:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --llm-eval --disable-cross-topic --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --attacks visual_prompt_injection --position-mode $(POSITION_MODE) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_vpi_llm.json

full-stress-llm:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --llm-eval --attack-suite trusted_user_stress --disable-cross-topic --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_trusted_user_stress_llm.json

full-mm:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --attack-suite mm_browsecomp --run-mm-browsecomp --mm-only --vision-caption-mode openai --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_mm_browsecomp.json

# MM-BrowseComp clean track: no attack injection, measures baseline utility on filtered cases
full-mm-clean:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --run-mm-browsecomp --mm-only --mm-splits clean \
	  --vision-caption-mode openai \
	  --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) \
	  $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) \
	  --out $(RUN_DIR)/sagemem_mm_browsecomp_clean.json

# MM-BrowseComp adversarial track: browsing-native attacks only, measures robustness
full-mm-adversarial:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --attack-suite mm_browsecomp --run-mm-browsecomp --mm-only --mm-splits poisoned \
	  --vision-caption-mode openai \
	  --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) \
	  $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) \
	  --out $(RUN_DIR)/sagemem_mm_browsecomp_adversarial.json

full-v2-ablations:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --include-v2-ablations --attack-suite main --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_v2_ablations.json

full-mm-robust:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --enable-locomo-multimodal --multimodal-robustness-mode missing_or_noisy --multimodal-robustness-rate 0.5 --attack-suite main --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_multimodal_robustness.json

full-mm-robust-ablations:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --include-v2-ablations --enable-locomo-multimodal --multimodal-robustness-mode missing_or_noisy --multimodal-robustness-rate 0.5 --attack-suite main --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_multimodal_robustness_ablations.json

full-mm-openai-frozen:
	mkdir -p $(RUN_DIR)
	$(PYTHON) run_eval.py --sage-v2 --enable-locomo-multimodal --multimodal-adversary-mode openai --vision-caption-mode openai --attack-suite main --position-mode $(POSITION_MODE) --checkpoint-every $(CHECKPOINT_EVERY) $(MAX_WORKERS_FLAG) $(RESUME_FLAG) $(CASE_FRACTION_FLAG) --seeds $(SEEDS) --out $(RUN_DIR)/sagemem_multimodal_openai_frozen.json
