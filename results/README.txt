Populated by running `python3 backend/train.py`:
  metrics.json               per-model best-val + held-out test metrics + full history
  dataset_distribution.json  race/pair counts and class balance per split
  model_comparison_test.csv  CO4-style comparison table across the 5 trained architectures
  ablation_study.json        CO5 ablation: CNN+GAT with feature groups disabled
