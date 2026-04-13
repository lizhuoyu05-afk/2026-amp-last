# fixedseq_v1_rc1

Status: candidate (RC1)

## Why
Re-trained after fixing empty `Sequence` values in legacy split CSVs.

## Data
- dataverse_files_fixedseq_v1

## Model
- Teacher: ESM2 150M + LoRA
- Student: ESM2 35M distillation

## Test Metrics (fixedseq)
- PR-AUC(any): 0.95698
- ROC-AUC(any): 0.98259
- MCC@0.5(any): 0.86539
- best_thr(any): 0.55
- macro_pr_subheads: 0.59986

## Notes
- legacy_badseq models are deprecated
- near-duplicate stress test is in progress; promote after full stress tests
