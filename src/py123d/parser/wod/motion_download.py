"""Download utilities for the Waymo Open Motion Dataset (WOMD).

WOMD is hosted on publicly readable Google Cloud Storage buckets once the license
has been accepted on the Waymo Open Dataset site. Access uses Application Default
Credentials (``gcloud auth application-default login`` or
``$GOOGLE_APPLICATION_CREDENTIALS``) or an explicit service-account JSON via
``--credentials-file``. If neither is available, an anonymous client is used.

Bucket layout (default version ``1_3_0``)::

    gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/
        scenario/
            training/training.tfrecord-00000-of-01000 ... -00999-of-01000
            validation/validation.tfrecord-00000-of-00150 ...
            testing/testing.tfrecord-00000-of-00150 ...
            training_20s/
            validation_interactive/
            testing_interactive/
        lidar/
            training/*.tfrecord-*
            validation/*.tfrecord-*
            testing/*.tfrecord-*

GCS scenario folder ↔ 123D split name mapping (see ``WOD_MOTION_SPLIT_TO_GCS_FOLDER``):

    training                ↔ wod-motion_train
    validation              ↔ wod-motion_val
    testing                 ↔ wod-motion_test
    training_20s            ↔ wod-motion-20s_train
    validation_interactive  ↔ wod-motion-interactive_val
    testing_interactive     ↔ wod-motion-interactive_test

Asymmetric availability: ``training_20s`` has no val/test counterpart, and the ``*_interactive``
folders have no training counterpart — the parser's split registry drops those combinations.

On-disk output layout under ``--output-dir``::

    <output_dir>/
        training/*.tfrecord-*
        validation/*.tfrecord-*
        ...
        lidar/
            training/*.tfrecord-*
            validation/*.tfrecord-*
            testing/*.tfrecord-*

This matches what :class:`~py123d.parser.wod.wod_motion_parser.WODMotionParser`
expects for its scenario tfrecords (lidar lands in a sibling tree and is ignored
by the parser; parser-side WOMD lidar support is a separate, non-goal feature).

This module doubles as (a) the ``py123d-womd-download`` CLI and (b) a reusable
library used by ``WODMotionParser``'s ``stream_enabled=True`` mode to fetch
selected shards into a temp directory just-in-time.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from google.cloud import storage

logger = logging.getLogger(__name__)

WOMD_BUCKET_PREFIX = "waymo_open_dataset_motion_v_"
# Canonical form uses dots ("1.3.0") so Hydra CLI overrides don't get reparsed as the
# numeric literal 130 (Python treats ``1_3_0`` as an int). Internally we normalize
# dots to underscores when building the bucket name.
WOMD_DEFAULT_VERSION = "1.3.0"

SCENARIO_SPLITS: Tuple[str, ...] = (
    "training",
    "validation",
    "testing",
    "training_20s",
    "validation_interactive",
    "testing_interactive",
)
LIDAR_SPLITS: Tuple[str, ...] = ("training", "validation", "testing")
SECTION_CHOICES: Tuple[str, ...] = ("scenario", "lidar", "both")


# ----------------------------------------------------------------------------------------------------------------------
# Auth / client
# ----------------------------------------------------------------------------------------------------------------------


def _require_gcs():
    """Lazy import — the dependency is optional until a user actually runs this script."""
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise SystemExit(
            "google-cloud-storage is required for WOMD downloads. Install it with:\n"
            "  pip install py123d[womd-download]\n"
            "or directly:\n"
            "  pip install google-cloud-storage\n"
        ) from exc
    return storage


def resolve_gcs_client(credentials_file: Optional[Path] = None) -> "storage.Client":
    """Build a ``google.cloud.storage.Client`` using the standard auth fallback chain.

    Order:

    1. Explicit ``credentials_file`` (service-account JSON) → ``Client.from_service_account_json``.
    2. ``$GOOGLE_APPLICATION_CREDENTIALS`` or Application Default Credentials → ``Client()``.
    3. If both fail, an anonymous client (WOMD buckets are publicly readable to license acceptors).

    :param credentials_file: Optional path to a service-account JSON key.
    :return: An initialized GCS client.
    """
    storage = _require_gcs()

    if credentials_file is not None:
        credentials_path = Path(credentials_file).expanduser()
        if not credentials_path.exists():
            raise FileNotFoundError(f"Service account JSON not found: {credentials_path}")
        logger.debug("Using service-account credentials from %s", credentials_path)
        return storage.Client.from_service_account_json(str(credentials_path))

    # Use google.auth.default() directly so we can handle the common case where the user
    # ran `gcloud auth application-default login` but never set a quota project —
    # storage.Client() would raise in that case even though the credentials are valid.
    # WOMD reads are public so no real project is needed; we pass a placeholder.
    try:
        import google.auth

        credentials, project = google.auth.default()
        logger.debug("Using Application Default Credentials (project=%s)", project)
        return storage.Client(credentials=credentials, project=project or "py123d-womd")
    except Exception as exc:  # DefaultCredentialsError, etc.
        logger.warning(
            "Could not create authenticated GCS client (%s); falling back to anonymous client. "
            "If you expected authenticated access, run: gcloud auth application-default login",
            exc,
        )
        return storage.Client.create_anonymous_client()


# ----------------------------------------------------------------------------------------------------------------------
# Listing / selection
# ----------------------------------------------------------------------------------------------------------------------


def _bucket_name(version: str) -> str:
    # Accept both "1.3.0" (canonical / recommended) and "1_3_0" — normalize to the
    # underscore form used in the GCS bucket name.
    normalized = str(version).replace(".", "_")
    return f"{WOMD_BUCKET_PREFIX}{normalized}"


def _split_prefix(section: str, split: str) -> str:
    """GCS key prefix for a given section+split, without bucket name."""
    if section == "scenario":
        return f"uncompressed/scenario/{split}/"
    if section == "lidar":
        return f"uncompressed/lidar/{split}/"
    raise ValueError(f"Unknown section {section!r}; expected one of 'scenario' | 'lidar'.")


def _local_split_dir(output_dir: Path, section: str, split: str) -> Path:
    """Where a given section+split's tfrecords should land on disk."""
    if section == "scenario":
        return output_dir / split
    if section == "lidar":
        return output_dir / "lidar" / split
    raise ValueError(f"Unknown section {section!r}; expected one of 'scenario' | 'lidar'.")


def list_split_shards(
    client: "storage.Client",
    section: str,
    split: str,
    version: str = WOMD_DEFAULT_VERSION,
) -> List[str]:
    """List all tfrecord blob names for a given section+split.

    :param client: GCS client.
    :param section: ``"scenario"`` or ``"lidar"``.
    :param split: Split name (e.g. ``"training"``).
    :param version: WOMD version string like ``"1_3_0"``.
    :return: Sorted list of blob names (keys within the bucket).
    """
    bucket = _bucket_name(version)
    prefix = _split_prefix(section, split)
    blobs = client.list_blobs(bucket, prefix=prefix)
    blob_names: List[str] = [b.name for b in blobs if ".tfrecord" in b.name]
    blob_names.sort()
    return blob_names


def select_shards(
    shard_blob_names: Sequence[str],
    shard_indices: Optional[Sequence[int]] = None,
    num_shards: Optional[int] = None,
    sample_random: bool = False,
    seed: int = 0,
) -> List[str]:
    """Filter a full shard list down to the requested subset.

    Precedence: ``shard_indices`` > ``num_shards`` > full list.

    :param shard_blob_names: All shards for a split (sorted).
    :param shard_indices: Exact indices into the sorted shard list to keep.
    :param num_shards: If set, keep the first ``num_shards`` (or a random sample if
        ``sample_random=True``).
    :param sample_random: Randomize the ``num_shards`` selection.
    :param seed: RNG seed when ``sample_random=True``.
    :return: Selected subset of ``shard_blob_names``.
    """
    total = len(shard_blob_names)
    if shard_indices is not None:
        selected: List[str] = []
        for idx in shard_indices:
            if idx < 0 or idx >= total:
                raise IndexError(f"shard index {idx} out of range [0, {total}).")
            selected.append(shard_blob_names[idx])
        return selected

    if num_shards is None or num_shards >= total:
        return list(shard_blob_names)

    if sample_random:
        rng = random.Random(seed)
        return sorted(rng.sample(list(shard_blob_names), num_shards))
    return list(shard_blob_names[:num_shards])


# ----------------------------------------------------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------------------------------------------------


def _destination_for_blob(blob_name: str, output_dir: Path) -> Path:
    """Map a GCS blob name back to its on-disk destination.

    ``uncompressed/scenario/{split}/file.tfrecord-*`` → ``{output_dir}/{split}/file.tfrecord-*``
    ``uncompressed/lidar/{split}/file.tfrecord-*`` → ``{output_dir}/lidar/{split}/file.tfrecord-*``
    """
    parts = blob_name.split("/")
    # Expected shape: ["uncompressed", "scenario"|"lidar", "<split>", "<filename>"]
    if len(parts) < 4 or parts[0] != "uncompressed":
        raise ValueError(f"Unexpected blob layout: {blob_name!r}")
    section = parts[1]
    split = parts[2]
    filename = "/".join(parts[3:])
    return _local_split_dir(output_dir, section, split) / filename


def _download_one_blob(
    client: "storage.Client",
    bucket_name: str,
    blob_name: str,
    dest_path: Path,
    overwrite: bool,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not overwrite:
        logger.debug("Skip existing file: %s", dest_path)
        return dest_path
    logger.info("Downloading gs://%s/%s → %s", bucket_name, blob_name, dest_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    blob.download_to_filename(str(tmp_path))
    tmp_path.replace(dest_path)
    return dest_path


def download_shards(
    client: "storage.Client",
    blob_names: Sequence[str],
    output_dir: Path,
    version: str = WOMD_DEFAULT_VERSION,
    max_workers: int = 8,
    overwrite: bool = False,
) -> List[Path]:
    """Download a list of shards (by blob name) into ``output_dir``.

    The destination path for each blob is derived from its key:
    ``uncompressed/<section>/<split>/<file>`` → ``<output_dir>/[lidar/]<split>/<file>``.

    :param client: GCS client.
    :param blob_names: Blob keys to download.
    :param output_dir: Local root directory.
    :param version: WOMD version (controls bucket name).
    :param max_workers: Parallel download threads.
    :param overwrite: If ``False``, skip files that already exist locally.
    :return: Local paths of downloaded (or pre-existing) files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_name = _bucket_name(version)

    if not blob_names:
        return []

    downloaded: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(
                _download_one_blob,
                client,
                bucket_name,
                blob_name,
                _destination_for_blob(blob_name, output_dir),
                overwrite,
            ): blob_name
            for blob_name in blob_names
        }
        for future in concurrent.futures.as_completed(futures):
            blob_name = futures[future]
            try:
                downloaded.append(future.result())
            except Exception as exc:
                logger.error("Failed to download %s: %s", blob_name, exc)
                raise
    return downloaded


def download_single_shard(
    client: "storage.Client",
    section: str,
    split: str,
    shard_idx: int,
    output_dir: Path,
    version: str = WOMD_DEFAULT_VERSION,
    overwrite: bool = False,
) -> Path:
    """Download exactly one shard identified by its index within the sorted shard list.

    Used by ``WODMotionParser``'s streaming mode to fetch shards on demand.
    """
    all_shards = list_split_shards(client, section=section, split=split, version=version)
    if not all_shards:
        raise FileNotFoundError(f"No tfrecords found at gs://{_bucket_name(version)}/{_split_prefix(section, split)}")
    if shard_idx < 0 or shard_idx >= len(all_shards):
        raise IndexError(f"Shard index {shard_idx} out of range for {section}/{split} (have {len(all_shards)} shards).")
    blob_name = all_shards[shard_idx]
    dest = _destination_for_blob(blob_name, Path(output_dir))
    return _download_one_blob(client, _bucket_name(version), blob_name, dest, overwrite=overwrite)


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------


def _resolve_output_dir(cli_output: Optional[str]) -> Path:
    """Falls back to ``$WOD_MOTION_DATA_ROOT`` then ``./data/wod_motion``."""
    if cli_output:
        resolved = Path(cli_output)
    elif os.environ.get("WOD_MOTION_DATA_ROOT"):
        resolved = Path(os.environ["WOD_MOTION_DATA_ROOT"])
    else:
        resolved = Path.cwd() / "data" / "wod_motion"
    return resolved.expanduser().resolve()


def _resolve_sections(section: str) -> List[str]:
    if section == "both":
        return ["scenario", "lidar"]
    return [section]


def _split_allowed_for_section(section: str, split: str) -> bool:
    if section == "scenario":
        return split in SCENARIO_SPLITS
    if section == "lidar":
        return split in LIDAR_SPLITS
    return False


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="py123d-womd-download",
        description=("Download the Waymo Open Motion Dataset (or a subset of shards) from Google Cloud Storage."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory. Falls back to $WOD_MOTION_DATA_ROOT, then ./data/wod_motion.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=WOMD_DEFAULT_VERSION,
        help=f"WOMD version string, mapped to bucket {WOMD_BUCKET_PREFIX}<version>.",
    )
    parser.add_argument(
        "--credentials-file",
        type=str,
        default=None,
        help=(
            "Optional service-account JSON for GCS auth. If omitted, Application Default "
            "Credentials are used (gcloud auth application-default login, or "
            "$GOOGLE_APPLICATION_CREDENTIALS). If neither is available, an anonymous client "
            "is used."
        ),
    )
    parser.add_argument(
        "--section",
        choices=SECTION_CHOICES,
        default="scenario",
        help=(
            "Which part of WOMD to download. 'scenario' = motion tfrecords (the usual dataset). "
            "'lidar' = the separate first-1s lidar archive. 'both' = pull both."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(set(SCENARIO_SPLITS) | set(LIDAR_SPLITS)),
        default=["training", "validation", "testing"],
        help="Which splits to download. Lidar section only has training/validation/testing.",
    )

    selection = parser.add_argument_group("shard selection (per split)")
    selection.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Download only N shards per split (first N, or N random if --random).",
    )
    selection.add_argument(
        "--shard-indices",
        nargs="+",
        type=int,
        default=None,
        help="Exact shard indices (into the sorted shard list) to download; applied to every split.",
    )
    selection.add_argument(
        "--random",
        dest="sample_random",
        action="store_true",
        help="Sample --num-shards randomly instead of taking the first N.",
    )
    selection.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --random is set.",
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument(
        "--list",
        dest="list_only",
        action="store_true",
        help="List the shards that would be downloaded and exit.",
    )
    misc.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan (bucket, prefixes, counts) without downloading.",
    )
    misc.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download threads.",
    )
    misc.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download shards even if the destination file already exists.",
    )
    misc.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args(argv)

    if args.shard_indices is not None and args.num_shards is not None:
        parser.error("--shard-indices and --num-shards are mutually exclusive")
    if args.num_shards is not None and args.num_shards <= 0:
        parser.error("--num-shards must be a positive integer")
    if args.sample_random and args.num_shards is None:
        parser.error("--random requires --num-shards")

    return args


def _plan_downloads(
    client: "storage.Client",
    sections: Sequence[str],
    splits: Sequence[str],
    version: str,
    shard_indices: Optional[Sequence[int]],
    num_shards: Optional[int],
    sample_random: bool,
    seed: int,
) -> List[str]:
    """Enumerate the blob names that would be downloaded given CLI selectors."""
    plan: List[str] = []
    for section in sections:
        for split in splits:
            if not _split_allowed_for_section(section, split):
                logger.debug("Skip %s split %s — not available in this section.", section, split)
                continue
            all_shards = list_split_shards(client, section=section, split=split, version=version)
            selected = select_shards(
                all_shards,
                shard_indices=shard_indices,
                num_shards=num_shards,
                sample_random=sample_random,
                seed=seed,
            )
            logger.info("Selected %d / %d shards from %s/%s", len(selected), len(all_shards), section, split)
            plan.extend(selected)
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    output_dir = _resolve_output_dir(args.output_dir)
    bucket = _bucket_name(args.version)
    sections = _resolve_sections(args.section)

    credentials_file = Path(args.credentials_file) if args.credentials_file else None
    client = resolve_gcs_client(credentials_file)

    plan = _plan_downloads(
        client=client,
        sections=sections,
        splits=args.splits,
        version=args.version,
        shard_indices=args.shard_indices,
        num_shards=args.num_shards,
        sample_random=args.sample_random,
        seed=args.seed,
    )

    logger.info("Target directory: %s", output_dir)
    logger.info("Bucket:           gs://%s/", bucket)
    logger.info("Section(s):       %s", ", ".join(sections))
    logger.info("Splits:           %s", ", ".join(args.splits))
    logger.info("Shards selected:  %d", len(plan))
    for blob_name in plan[: min(10, len(plan))]:
        logger.debug("  gs://%s/%s", bucket, blob_name)

    if args.list_only:
        for blob_name in plan:
            sys.stdout.write(f"gs://{bucket}/{blob_name}\n")
        return 0

    if args.dry_run:
        logger.info("--dry-run set, not downloading.")
        return 0

    if not plan:
        logger.error("No shards selected — nothing to download.")
        return 1

    download_shards(
        client=client,
        blob_names=plan,
        output_dir=output_dir,
        version=args.version,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )
    logger.info("Done. Data written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
