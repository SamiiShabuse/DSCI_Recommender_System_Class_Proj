from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ast
import csv
import json
import zipfile


@dataclass(frozen=True)
class DatasetPaths:
    base_dir: Path

    @property
    def influencers(self) -> Path:
        return self.base_dir / "influencers.txt"

    @property
    def mapping(self) -> Path:
        return self.base_dir / "JSON-Image_files_mapping.txt"

    @property
    def sample_images_zip(self) -> Path:
        return self.base_dir / "sample_images.zip"

    @property
    def sample_images_dir(self) -> Path:
        return self.base_dir / "sample_images"

    @property
    def metadata_dir_candidates(self) -> list[Path]:
        return [
            self.base_dir / "Post_metadata",
            self.base_dir / "post_metadata",
            self.base_dir / "metadata",
            self.base_dir / "JSON_files",
        ]


def load_influencers(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        headers = next(reader)
        for row in reader:
            if not row or len(row) < 4:
                continue
            if row[0].startswith("="):
                continue
            rows.append(
                {
                    headers[0]: row[0].strip(),
                    headers[1]: row[1].strip(),
                    headers[2]: int(row[2]),
                    headers[3]: int(row[3]),
                    headers[4]: int(row[4]),
                }
            )
    return rows


def load_mapping(path: str | Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("influencer_name"):
                continue
            if line.startswith("="):
                continue

            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f"Unexpected mapping format on line {line_number}: {line}")

            influencer_name, metadata_file, image_list_text = parts
            image_files = ast.literal_eval(image_list_text)
            rows.append(
                {
                    "influencer_name": influencer_name,
                    "json_postmetadata_file_name": metadata_file,
                    "image_files": image_files,
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def extract_sample_images(zip_path: str | Path, output_dir: str | Path) -> list[Path]:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            member_path = Path(member)
            if member_path.parts and member_path.parts[0].lower() == "sample_images":
                member_path = Path(*member_path.parts[1:])
            target_path = output_dir / member_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as destination:
                destination.write(source.read())
            extracted.append(target_path)
    return extracted


def find_metadata_dir(dataset_paths: DatasetPaths) -> Path | None:
    for candidate in dataset_paths.metadata_dir_candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def list_metadata_files(metadata_dir: str | Path, max_files: int | None = None) -> list[Path]:
    metadata_dir = Path(metadata_dir)
    files = sorted(
        [
            path
            for path in metadata_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".info", ".json"}
        ]
    )
    if max_files is not None:
        return files[:max_files]
    return files


def _parse_metadata_text(text: str) -> Any:
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {"raw_text": text}


def load_metadata_records(metadata_files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in metadata_files:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            payload = _parse_metadata_text(handle.read())

        records.append(
            {
                "metadata_file_name": file_path.name,
                "metadata_file_stem": file_path.stem,
                "metadata_path": str(file_path),
                "payload": payload,
            }
        )
    return records
