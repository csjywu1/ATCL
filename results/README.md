# Released results

`final_selected_test_results.json` records the validation-selected checkpoint, epoch, test metric, and checkpoint SHA-256 for every seed. `final_combined_test_summary.json` contains the three-seed mean and sample standard deviation used in the paper table.

Model checkpoints are intentionally not committed because the nine files are large. Reproduce them with the configurations in `configs/`, or publish them as release assets and verify them against the hashes in `final_selected_test_results.json`.
