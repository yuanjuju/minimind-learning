"""Export a parameter census and analytical KV budget for the pinned Dense model."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def collect():
    config = MiniMindConfig()
    # Shapes and parameter identity suffice; meta tensors allocate no weight storage.
    with torch.device("meta"):
        model = MiniMindForCausalLM(config)
    groups = {"tied_embedding_and_head": 0, "attention_projections": 0,
              "feed_forward": 0, "normalization": 0}
    for name, parameter in model.named_parameters():
        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            group = "tied_embedding_and_head"
        elif "_proj.weight" in name and ".self_attn." in name:
            group = "attention_projections"
        elif ".mlp." in name:
            group = "feed_forward"
        elif "norm" in name:
            group = "normalization"
        else:
            raise ValueError(f"Unclassified parameter: {name}")
        groups[group] += parameter.numel()

    total = sum(parameter.numel() for parameter in model.parameters())
    assert total == sum(groups.values())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert config.use_moe is False
    kv_bytes = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 2
    return {
        "schema_version": 1,
        "method": "Meta-device parameter census; analytical KV payload, not measured memory",
        "source": "model/model_minimind.py",
        "source_sha256": hashlib.sha256((ROOT / "model/model_minimind.py").read_bytes()).hexdigest(),
        "config": {key: getattr(config, key) for key in (
            "hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "intermediate_size", "rope_theta",
            "tie_word_embeddings", "use_moe")},
        "unique_parameters": total,
        "parameter_groups": groups,
        "kv_payload": {
            "dtype_bytes": 2,
            "bytes_per_token_per_sequence": kv_bytes,
            "example_sequence_tokens": 2048,
            "example_bytes_per_sequence": kv_bytes * 2048,
            "scope": "K and V tensors only; excludes weights, activations, temporary repeat_kv, allocator overhead",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON destination")
    args = parser.parse_args()
    report = json.dumps(collect(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
