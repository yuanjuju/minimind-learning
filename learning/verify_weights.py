"""按 docs/artifacts.sha256 核对个人训练权重；只读取文件，不反序列化模型。"""

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=ROOT / "out",
        help="包含 pretrain_768.pth 和 full_sft_768.pth 的目录；默认是本仓库 out/",
    )
    args = parser.parse_args()

    passed = True
    manifest = ROOT / "docs" / "artifacts.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split("  ", 1)
        path = args.weights_dir / Path(relative_path).name
        if not path.is_file():
            print(f"缺少：{path}")
            passed = False
            continue
        actual = sha256(path)
        matched = actual == expected
        print(f"{'通过' if matched else '不匹配'}：{path.name}  SHA-256={actual}")
        passed &= matched

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
