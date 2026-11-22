from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.download import download, file_md5
from src.paths import PROJECT_ROOT, RAW_DIR, load_dataset_config, resolve_data_paths

BARLEY_REPO = "Project-AgML/barley_disease_segmentation"
BARLEY_SHARDS = [f"data/train-{i:05d}-of-00010.parquet" for i in range(5)]
MASK_TO_CLASS = {1: "brown_rust", 2: "ramularia"}
MIN_FRACTION = 0.005
MIXED_SHARE = 0.25


def extract_zip(archive: Path, destination: Path) -> int:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    extracted = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename)
            if member.is_dir():
                continue
            if "__MACOSX" in member.filename or name.name.startswith("._"):
                continue
            if name.name in {".DS_Store", "Thumbs.db"}:
                continue
            zf.extract(member, destination)
            extracted += 1
    return extracted


def prepare_maize() -> int:
    config = load_dataset_config(PROJECT_ROOT / "configs" / "dataset_maize.yaml")
    paths = resolve_data_paths(config)
    source = config["source"]
    archive = RAW_DIR / "maize" / source["archive_name"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        download(source["download_url"], archive)
    else:
        print(f"Архив уже на диске: {archive} ({archive.stat().st_size / 2**20:.1f} МиБ)")
    digest = file_md5(archive)
    print(f"md5 кукурузы: {digest}")
    count = extract_zip(archive, paths["raw_images"])
    print(f"Кукуруза: распаковано файлов — {count}")
    return 0


def label_from_mask(mask: np.ndarray) -> str | None:
    total = mask.size
    frac_rust = float((mask == 1).sum()) / total
    frac_ram = float((mask == 2).sum()) / total
    if frac_rust < MIN_FRACTION and frac_ram < MIN_FRACTION:
        return "healthy"
    disease = frac_rust + frac_ram
    if frac_rust >= MIN_FRACTION and frac_ram >= MIN_FRACTION:
        if min(frac_rust, frac_ram) / max(disease, 1e-9) >= MIXED_SHARE:
            return None
    if frac_rust >= frac_ram:
        return MASK_TO_CLASS[1]
    return MASK_TO_CLASS[2]


def prepare_barley() -> int:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    config = load_dataset_config(PROJECT_ROOT / "configs" / "dataset_barley.yaml")
    paths = resolve_data_paths(config)
    raw_root = paths["raw_images"]
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True)

    parquet_dir = RAW_DIR / "barley"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for filename in BARLEY_SHARDS:
        local = hf_hub_download(
            repo_id=BARLEY_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(parquet_dir),
        )
        shard_paths.append(Path(local))
        print(f"Шард: {filename}")

    counts: dict[str, int] = {name: 0 for name in ["healthy", "brown_rust", "ramularia"]}
    skipped_mixed = 0
    index = 0
    for shard in shard_paths:
        table = pq.read_table(shard)
        for row_index in range(table.num_rows):
            rec = table.slice(row_index, 1).to_pydict()
            image = Image.open(BytesIO(rec["image"][0]["bytes"])).convert("RGB")
            mask = np.array(Image.open(BytesIO(rec["mask"][0]["bytes"])))
            label = label_from_mask(mask)
            if label is None:
                skipped_mixed += 1
                continue
            plant_id = str(rec["plant_id"][0])
            dest_dir = raw_root / label / plant_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{index:05d}.jpg"
            image.save(dest, format="JPEG", quality=90)
            counts[label] += 1
            index += 1
            if index % 200 == 0:
                print(f"  сохранено {index}")

    print(f"Ячмень: сохранено {index}, смешанных пропущено {skipped_mixed}")
    print(f"По классам: {counts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Скачать датасеты ячменя и кукурузы.")
    parser.add_argument("--crop", choices=("maize", "barley", "all"), default="all")
    args = parser.parse_args(argv)
    if args.crop in {"maize", "all"}:
        code = prepare_maize()
        if code:
            return code
    if args.crop in {"barley", "all"}:
        code = prepare_barley()
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
