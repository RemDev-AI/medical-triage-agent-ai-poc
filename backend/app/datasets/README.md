
---
configs:
- config_name: sft
  data_files:
    - split: train
      path: processed/sft/train.jsonl
    - split: validation
      path: processed/sft/validation.jsonl
    - split: test
      path: processed/sft/test.jsonl
    - split: clinical_eval
      path: processed/sft/clinical_eval.jsonl

- config_name: dpo
  data_files:
    - split: train
      path: processed/dpo/train.jsonl
    - split: validation
      path: processed/dpo/validation.jsonl
    - split: test
      path: processed/dpo/test.jsonl
    - split: clinical_eval
      path: processed/dpo/clinical_eval.jsonl
---
