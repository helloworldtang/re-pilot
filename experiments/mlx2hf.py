"""MLX 融合模型 → HF 格式转换器。

背景：mlx_lm.fuse 只存 MLX 格式（--export-gguf 又不支持 qwen3_5 架构），
上服务器走 llama.cpp 必须回到 HF 格式再转 GGUF。
MLX 与 HF 的 safetensors 键名同构（mlx_lm 镜像 HF 命名）、权重布局一致
（都是 row-major [out, in]），直接搬张量即可。

用法：uv run python experiments/mlx2hf.py --mlx models/re-pilot-merged --hf models/re-pilot-hf --base Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

# 直接复制的配置/词表文件（从 HF 底座拿，MLX 目录里的同名文件也可）
ASSET_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]


def convert(mlx_dir: str, hf_dir: str, base: str) -> None:
    """把 MLX 融合目录转成 HF 可加载目录。"""
    mlx_path = Path(mlx_dir)
    hf_path = Path(hf_dir)
    hf_path.mkdir(parents=True, exist_ok=True)

    # 权重：MLX safetensors → torch safetensors（键名不变）
    shards = sorted(mlx_path.glob("*.safetensors"))
    assert shards, f"{mlx_dir} 下没有 safetensors"
    total_keys = 0
    for shard in shards:
        tensors = load_file(str(shard))
        save_file(tensors, str(hf_path / shard.name), metadata={"format": "pt"})
        total_keys += len(tensors)
    print(f"权重: {len(shards)} 个 shard, {total_keys} 个张量")

    # 索引文件（多 shard 时键名与文件名均未变，直接复制；单 shard 删除索引）
    idx = mlx_path / "model.safetensors.index.json"
    if len(shards) == 1:
        (hf_path / "model.safetensors.index.json").unlink(missing_ok=True)
    elif idx.exists():
        shutil.copy(idx, hf_path / "model.safetensors.index.json")

    # 配置与词表：优先 MLX 目录（融合时已带），缺的从 HF 底座缓存补
    base_snapshots = Path.home() / ".cache/huggingface/hub"
    base_local = None
    for candidate in base_snapshots.glob("models--*/snapshots/*"):
        if (candidate / "config.json").exists() and base.split("/")[-1] in str(candidate):
            base_local = candidate
            break
    copied = []
    for name in ASSET_FILES:
        src = mlx_path / name
        if not src.exists() and base_local is not None:
            src = base_local / name
        if src.exists():
            shutil.copy(src, hf_path / name)
            copied.append(name)
    print(f"配置/词表: 复制 {len(copied)} 个文件 {copied}")
    print(f"完成 → {hf_path}")


def main() -> None:
    """命令行入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", default="models/re-pilot-merged")
    ap.add_argument("--hf", default="models/re-pilot-hf")
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    args = ap.parse_args()
    convert(args.mlx, args.hf, args.base)


if __name__ == "__main__":
    main()
