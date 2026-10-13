Threshold = -225.07 (80% of the best algorithm's improvement over the starting return)

| variant | n_seeds | final_return | final_return_ci95 | auc | auc_ci95 | stability_std_last20 | stability_std_last20_ci95 | reached_threshold | median_steps_to_threshold | wall_minutes | sps | update_fraction | median_minutes_to_threshold | final_vs_ref | final_welch_p | auc_vs_ref | auc_welch_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo | 2 | -305.58 | 214.38 | -327.74 | 211.62 | 2.20 | 7.57 | 0/2 |  | 2.58 | 119.03 | 0.38 |  |  |  |  |  |
| sac | 2 | -203.97 | 72.73 | -223.58 | 217.56 | 24.05 | 56.45 | 2/2 | 4500.00 | 10.83 | 30.77 | 0.91 | 2.24 | 101.61 | 0.08 | 104.16 | 0.05 |
| td3 | 2 | -361.92 | 1555.00 | -397.21 | 449.89 | 31.22 | 271.24 | 2/2 | 10500.00 | 6.04 | 55.15 | 0.91 | 3.04 | -56.34 | 0.73 | -69.48 | 0.27 |
