"""Download and verify the small release parts, then assemble the demo model ZIP."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RELEASE_BASE = "https://github.com/jiang-bingyu/FedMedDP/releases/download/"


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def download_archive(manifest: dict, output: Path, base_url: str) -> None:
    if output.exists():
        with output.open("rb") as handle:
            digest = sha256_stream(handle)
        if digest != manifest["archive_sha256"]:
            raise FileExistsError(f"Existing file has a different checksum: {output}")
        print(f"Already downloaded and verified: {output}")
        return

    # A single temporary file keeps an interrupted download from replacing user files.
    with tempfile.TemporaryFile() as archive:
        for index, part in enumerate(manifest["parts"], start=1):
            expected_name = f"{manifest['archive_name']}.{index:03d}"
            if part["name"] != expected_name:
                raise ValueError(f"Unexpected part name: {part['name']}")
            start = archive.tell()
            for attempt in range(3):
                archive.seek(start)
                archive.truncate()
                digest = hashlib.sha256()
                size = 0
                request = Request(base_url + part["name"], headers={"User-Agent": "FedMedDP-model-downloader"})
                try:
                    with urlopen(request, timeout=60) as response:
                        while chunk := response.read(1024 * 1024):
                            archive.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    if size != part["size"] or digest.hexdigest() != part["sha256"]:
                        raise ValueError(f"Checksum mismatch: {part['name']}")
                    break
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2)
            print(f"Verified part {index}/{len(manifest['parts'])}", flush=True)

        if archive.tell() != manifest["archive_size"]:
            raise ValueError("Archive size mismatch")
        archive.seek(0)
        if sha256_stream(archive) != manifest["archive_sha256"]:
            raise ValueError("Archive checksum mismatch")
        archive.seek(0)
        with output.open("xb") as target:
            shutil.copyfileobj(archive, target, length=1024 * 1024)
    print(f"Downloaded and verified: {output}")


def main() -> None:
    manifest = json.loads((ROOT / "docs" / "demo-models-parts.json").read_text(encoding="utf-8"))
    download_archive(
        manifest,
        ROOT / "fedmeddp-demo-models.zip",
        RELEASE_BASE + manifest["release_tag"] + "/",
    )


if __name__ == "__main__":
    main()
