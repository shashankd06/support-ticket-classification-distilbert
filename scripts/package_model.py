"""Package existing weights unchanged for release storage outside Git history."""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    root = Path("artifacts/full")
    target = Path("artifacts/deployment/distilbert-full.tar.gz")
    target.parent.mkdir(parents=True, exist_ok=True)
    names = ["config.json", "labels.json"] + [
        "distilbert/" + name
        for name in [
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ]
    ]
    notice = (
        b"IntentDesk fine-tuned banking intent classifier.\n"
        b"Base: distilbert/distilbert-base-uncased (Apache-2.0), revision "
        b"12040accade4e8a0f71eabdb258fecc2e7e948be.\n"
        b"Training data: BANKING77 by PolyAI / Casanueva et al. (2020), CC BY 4.0.\n"
        b"https://github.com/PolyAI-LDN/task-specific-datasets\n"
        b"https://creativecommons.org/licenses/by/4.0/\n"
        b"Changes: overlap/conflict filtering, validation split and classifier fine-tuning.\n"
        b"No endorsement is implied. Scores are not calibrated confidence.\n"
        b"Only existing full-training files are packaged; weights are not modified.\n"
    )
    with (
        target.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as bundle,
    ):
        for name in names:
            path = root / name
            member = tarfile.TarInfo(name)
            member.size = path.stat().st_size
            member.mode = 0o644
            with path.open("rb") as stream:
                bundle.addfile(member, stream)
        for name, contents in [
            ("NOTICE.txt", notice),
            (
                "LICENSE-BASE-MODEL.txt",
                Path("deployment/LICENSE-BASE-MODEL.txt").read_bytes(),
            ),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            member.mode = 0o644
            bundle.addfile(member, io.BytesIO(contents))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = {
        "url": "https://github.com/shashankd06/support-ticket-classification-distilbert/releases/download/intentdesk-model-v1/distilbert-full.tar.gz",
        "sha256": digest,
        "checkpoint_sha256": hashlib.sha256(
            (root / "distilbert/model.safetensors").read_bytes()
        ).hexdigest(),
        "mode": "full",
        "note": "Original FP32 trained weights; no quantization or retraining",
    }
    Path("deployment/model-release.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Packaged original checkpoint:", target, "bytes:", target.stat().st_size)
    print("SHA-256:", digest)


if __name__ == "__main__":
    main()
