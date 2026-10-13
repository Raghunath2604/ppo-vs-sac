Threshold = 130.24 (80% of the best algorithm's improvement over the starting return)

| variant | n_seeds | final_return | final_return_ci95 | auc | auc_ci95 | stability_std_last20 | stability_std_last20_ci95 | reached_threshold | median_steps_to_threshold | wall_minutes | sps | update_fraction | median_minutes_to_threshold | final_vs_ref | final_welch_p | auc_vs_ref | auc_welch_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo | 2 | 161.50 | 580.67 | 49.19 | 52.81 | 37.27 | 232.21 | 1/2 | 18432.00 | 2.08 | 147.57 | 0.43 | 2.10 |  |  |  |  |
| sac | 2 | 101.18 | 99.32 | 81.60 | 62.41 | 5.72 | 10.65 | 0/2 |  | 12.36 | 26.96 | 0.92 |  | -60.32 | 0.41 | 32.42 | 0.04 |
| td3 | 2 | 62.98 | 27.74 | 57.34 | 72.84 | 2.10 | 2.50 | 0/2 |  | 7.63 | 43.71 | 0.92 |  | -98.52 | 0.28 | 8.16 | 0.38 |
