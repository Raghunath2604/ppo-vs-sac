.PHONY: algos-quick algos-train algos-report algos-all mt-train mt-eval mt-report mt-all install install-cpu test lint fmt smoke train ablate-quick ablate-core ablate-full analyze export serve docker-build docker-train clean

PY ?= python
ENV ?= HalfCheetah-v5
SEEDS ?= 5
STEPS ?= 1000000

install:
	$(PY) -m pip install -e ".[dev]"

# CPU-only PyTorch (much smaller download than the default CUDA build), then the project
install-cpu:
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check . && ruff format --check .

fmt:
	ruff check --fix . && ruff format .

# ~1 minute: proves the algorithm learns (InvertedPendulum reaches ~1000 return)
smoke:
	$(PY) -m algo_compare.train --env-id InvertedPendulum-v5 --total-timesteps 40960 --log-dir runs/smoke --eval-episodes 5

# one full single-seed run
train:
	$(PY) -m algo_compare.train --env-id $(ENV) --total-timesteps $(STEPS) --seed 1 --log-dir runs/$(ENV)_seed1

# fast end-to-end check of the whole ablation + analysis pipeline (a few minutes)
ablate-quick:
	$(PY) -m algo_compare.ablate --study tiny --env-id InvertedPendulum-v5 --seeds 3 --total-timesteps 40960 --eval-episodes 5 --out runs/quick
	$(PY) -m algo_compare.analysis --runs runs/quick --out results/quick

# the real study: 5 variants x SEEDS seeds on ENV (hours on a laptop CPU; use JOBS to parallelise)
JOBS ?= 1
ablate-core:
	$(PY) -m algo_compare.ablate --study core --env-id $(ENV) --seeds $(SEEDS) --total-timesteps $(STEPS) --jobs $(JOBS) --out runs/core_$(ENV)

ablate-full:
	$(PY) -m algo_compare.ablate --study full --env-id $(ENV) --seeds $(SEEDS) --total-timesteps $(STEPS) --jobs $(JOBS) --out runs/full_$(ENV)

analyze:
	$(PY) -m algo_compare.analysis --runs runs/core_$(ENV) --out results/core_$(ENV)

MODEL ?= runs/smoke/model.pt
export:
	$(PY) -m algo_compare.export --model $(MODEL) --out-dir export

serve:
	PPO_MODEL_PATH=$(MODEL) $(PY) -m uvicorn --factory algo_compare.serve:create_app --host 0.0.0.0 --port 8000

demo:
	$(PY) demo.py --algo sac --target 2.5 --steps 200

demo-compare:
	$(PY) demo.py --compare

dashboard:
	$(PY) -m http.server 8089 --directory dashboard

pdf:
	$(PY) scripts/build_paper_pdf.py

diagram:
	$(PY) scripts/generate_architecture_diagram.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf runs results export .pytest_cache .ruff_cache

# ---------------- project 3: PPO vs SAC vs TD3, and multi-task ----------------
AENV ?= HalfCheetah-v5
ASEEDS ?= 5
ASTEPS ?= 300000
AJOBS ?= 1
ADIR ?= runs/algos_$(AENV)
ARES ?= results/algos_$(AENV)

# pipeline check on an easy task with small networks (~15-25 minutes on a slow CPU)
algos-quick:
	$(PY) -m algo_compare.compare_study --study algos --env-id InvertedPendulum-v5 --algos ppo,sac,td3 --seeds 2 --total-timesteps 20000 --eval-episodes 5 \
		--override sac.hidden_size=64 --override td3.hidden_size=64 --override sac.batch_size=64 --override td3.batch_size=64 \
		--override sac.learning_starts=1000 --override td3.learning_starts=1000 --override sac.log_interval=1000 --override td3.log_interval=1000 \
		--out runs/algos_quick
	$(PY) -m algo_compare.compare_report --runs runs/algos_quick --out results/algos_quick --reference ppo

algos-train:
	$(PY) -m algo_compare.compare_study --study algos --env-id $(AENV) --algos ppo,sac,td3 --seeds $(ASEEDS) --total-timesteps $(ASTEPS) --jobs $(AJOBS) --out $(ADIR)

algos-report:
	$(PY) -m algo_compare.compare_report --runs $(ADIR) --out $(ARES) --reference ppo

algos-all: algos-train algos-report

MTENV ?= HalfCheetah-v5
MTDIR ?= runs/mt_$(MTENV)
MTRES ?= results/mt_$(MTENV)
TARGETS ?= 1.0,2.5,4.0

mt-train:
	$(PY) -m algo_compare.compare_study --study multitask --env-id $(MTENV) --algos ppo,sac --targets $(TARGETS) --seeds 3 --total-timesteps $(ASTEPS) --jobs $(AJOBS) --out $(MTDIR)

mt-eval:
	$(PY) -m algo_compare.task_eval --runs $(MTDIR) --targets $(TARGETS) --episodes 5 --jobs $(AJOBS) --out $(MTRES)/task_eval.csv

mt-report:
	$(PY) -m algo_compare.task_eval --report $(MTRES)/task_eval.csv --out $(MTRES)

mt-all: mt-train mt-eval mt-report
