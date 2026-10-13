from pathlib import Path

import numpy as np

from algo_compare.ablate import STUDIES, VARIANTS, build_jobs
from algo_compare.analysis import ci95_halfwidth


def test_ci_halfwidth_matches_known_value():
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    # std(ddof=1)=1.5811, sem=0.7071, t_{0.975,4}=2.7764 -> 1.9632
    assert abs(ci95_halfwidth(x)[0] - 1.9632) < 1e-3


def test_ci_single_seed_is_zero():
    assert ci95_halfwidth(np.array([[3.0]]))[0] == 0.0


def test_build_jobs_covers_variants_and_seeds(tmp_path):
    jobs = build_jobs("core", "HalfCheetah-v5", 3, 1000, str(tmp_path))
    assert len(jobs) == len(STUDIES["core"]) * 3
    names = {Path(j[0]["log_dir"]).parent.name for j in jobs}
    assert names == set(STUDIES["core"])
    no_obs = next(j[0] for j in jobs if "no_obs_norm" in j[1])
    assert no_obs["norm_obs"] is False and VARIANTS["no_obs_norm"] == {"norm_obs": False}


def test_summarize_and_plot_on_synthetic_runs(tmp_path):
    import json

    import pandas as pd

    from algo_compare.analysis import load_study, plot_all, summarize

    rng = np.random.default_rng(0)
    for variant, level in (("baseline", 100.0), ("no_obs_norm", 60.0)):
        for seed in (1, 2, 3):
            d = tmp_path / variant / f"seed_{seed}"
            d.mkdir(parents=True)
            steps = np.arange(1, 21) * 1000
            ret = level * (1 - np.exp(-steps / 8000)) + rng.normal(0, 2, size=20)
            pd.DataFrame({"global_step": steps, "mean_return_10": ret}).to_csv(d / "metrics.csv", index=False)
            (d / "final_eval.json").write_text(json.dumps({"mean_return": float(ret[-1])}))
    study = load_study(str(tmp_path))
    table = summarize(study)
    assert list(table["variant"]) == ["baseline", "no_obs_norm"]
    assert table.loc[1, "delta_vs_baseline"] < 0
    assert table.loc[1, "welch_p"] < 0.05
    written = plot_all(study, str(tmp_path / "plots"))
    assert len(written) == 2
