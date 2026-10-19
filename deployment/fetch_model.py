"""Fetch the pinned serving archive at image build time, never per request."""

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    metadata = json.loads(Path("deployment/model-release.json").read_text())
    url = metadata["url"]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("Only the pinned public GitHub release URL is accepted")
    destination = Path("artifacts/full").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "model.tar.gz"
        with (
            urllib.request.urlopen(url, timeout=180) as response,
            archive.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != metadata["sha256"]:
            raise ValueError("Model archive checksum mismatch")
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                target = (destination / member.name).resolve()
                if not target.is_relative_to(destination) or not member.isfile():
                    raise ValueError("Unsafe model archive member")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    print("Verified and installed the pinned full-training checkpoint")


if __name__ == "__main__":
    main()
