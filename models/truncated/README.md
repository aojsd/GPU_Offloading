# Truncated Models

**WARNING: These models are truncated for basic correctness testing and
performance debugging ONLY. Never use them for end-to-end experiments.**

Models here have had trailing layers removed to fit on a single GPU.
Outputs are meaningless for evaluation.

## Usage

```bash
python truncate_model.py <source_dir> <output_dir> --num-layers N
```

Example — truncate DeepSeek-V2 to 8 layers (~58 GB BF16):
```bash
python truncate_model.py /path/to/DeepSeek-V2 /path/to/DeepSeek-V2-8L --num-layers 8
```
