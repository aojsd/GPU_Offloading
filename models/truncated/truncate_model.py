#!/usr/bin/env -S python3 -u
"""Truncate a HuggingFace transformer model by removing trailing layers.

Creates a new model directory with only the first N layers, preserving all
global weights (embeddings, final norm, LM head) and tokenizer files.
Processes one safetensors shard at a time for memory efficiency.

Usage:
    python truncate_model.py <source_dir> <output_dir> --num-layers 20
"""
import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


LAYER_PATTERN = re.compile(r"model\.layers\.(\d+)\.")

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
]


def get_layer_index(tensor_name: str) -> int | None:
    """Extract layer index from tensor name, or None for global tensors."""
    m = LAYER_PATTERN.search(tensor_name)
    return int(m.group(1)) if m else None


def truncate_model(source_dir: str, output_dir: str, num_layers: int):
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(source / "config.json") as f:
        config = json.load(f)

    orig_layers = config["num_hidden_layers"]
    if num_layers >= orig_layers:
        raise ValueError(
            f"Requested {num_layers} layers but model only has {orig_layers}")

    print(f"Truncating {orig_layers} -> {num_layers} layers")
    print(f"Source: {source}")
    print(f"Output: {output}")

    # Load index to get tensor -> shard mapping
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Expected {index_path}. Single-file models not yet supported.")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Determine which tensors to keep
    keep_tensors = {}  # tensor_name -> source_shard
    skip_count = 0
    for name, shard in weight_map.items():
        layer_idx = get_layer_index(name)
        if layer_idx is not None and layer_idx >= num_layers:
            skip_count += 1
            continue
        keep_tensors[name] = shard

    print(f"Keeping {len(keep_tensors)} tensors, skipping {skip_count}")

    # Group kept tensors by source shard
    shard_to_tensors: dict[str, list[str]] = {}
    for name, shard in keep_tensors.items():
        shard_to_tensors.setdefault(shard, []).append(name)

    # Process each shard: load only kept tensors, save to new shard
    new_weight_map = {}
    total_size = 0
    out_shard_idx = 0

    for shard_name in sorted(shard_to_tensors.keys()):
        tensor_names = shard_to_tensors[shard_name]
        shard_path = source / shard_name

        # Load only the tensors we need from this shard
        tensors = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in tensor_names:
                tensors[name] = f.get_tensor(name)

        if not tensors:
            continue

        # Save to output shard
        out_shard_idx += 1
        out_shard_name = f"model-{out_shard_idx:05d}-of-PLACEHOLDER.safetensors"
        save_file(tensors, str(output / out_shard_name))

        shard_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        total_size += shard_bytes
        for name in tensors:
            new_weight_map[name] = out_shard_name

        print(f"  {shard_name} -> {out_shard_name} "
              f"({len(tensors)} tensors, {shard_bytes / 1e9:.2f} GB)")

        del tensors  # free memory before loading next shard

    # Rename shard files with correct total count
    num_shards = out_shard_idx
    for i in range(1, num_shards + 1):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        (output / old_name).rename(output / new_name)
        # Update weight map references
        for k, v in new_weight_map.items():
            if v == old_name:
                new_weight_map[k] = new_name

    # Write index
    new_index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    with open(output / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # Write updated config
    config["num_hidden_layers"] = num_layers
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer files
    for fname in TOKENIZER_FILES:
        src = source / fname
        if src.exists():
            shutil.copy2(src, output / fname)

    total_gb = total_size / 1024**3
    print(f"\nDone: {num_layers}-layer model at {output}")
    print(f"Total size: {total_gb:.1f} GB ({num_shards} shards)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate transformer model layers")
    parser.add_argument("source", help="Source model directory")
    parser.add_argument("output", help="Output directory for truncated model")
    parser.add_argument("--num-layers", type=int, required=True,
                        help="Number of layers to keep")
    args = parser.parse_args()
    truncate_model(args.source, args.output, args.num_layers)
