"""Download utilities for the Waymo Open Dataset — Motion (WOMD) and Perception.

This module provides the ``py123d-wod-download`` CLI (with ``motion`` and ``perception``
subcommands) plus reusable library functions that power the ``stream_enabled=True`` mode
of :class:`~py123d.parser.wod.wod_motion_parser.WODMotionParser` and
:class:`~py123d.parser.wod.wod_perception_parser.WODPerceptionParser`.

Auth (shared)
-------------
Both CLIs use the same fallback chain in :func:`resolve_gcs_client`:

1. Explicit service-account JSON via ``--credentials-file``.
2. Application Default Credentials (``gcloud auth application-default login`` or
   ``$GOOGLE_APPLICATION_CREDENTIALS``).
3. Anonymous client (works for the motion bucket; perception requires auth — see below).

Motion bucket layout (default version ``1.3.0``)
------------------------------------------------
``gs://waymo_open_dataset_motion_v_1_3_0/`` is publicly readable once the license
has been accepted on the Waymo Open Dataset site::

    gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/
        scenario/
            training/training.tfrecord-00000-of-01000 ...
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

Asymmetric availability: ``training_20s`` has no val/test counterpart, and the
``*_interactive`` folders have no training counterpart.

Perception bucket layout (default version ``1.4.3``)
----------------------------------------------------
``gs://waymo_open_dataset_v_1_4_3/`` requires an authenticated client (anonymous
access returns 403). Each segment is a single, non-sharded tfrecord::

    gs://waymo_open_dataset_v_1_4_3/
        individual_files/
            training/segment-<id>_with_camera_labels.tfrecord
            validation/segment-<id>_with_camera_labels.tfrecord
            testing/segment-<id>_with_camera_labels.tfrecord

GCS folder ↔ 123D split name mapping (see ``WOD_PERCEPTION_SPLIT_TO_GCS_FOLDER``):

    training    ↔ wod-perception_train
    validation  ↔ wod-perception_val
    testing     ↔ wod-perception_test

The bucket also contains ``archived_files/`` (tar archives) and
``individual_files/domain_adaptation/`` (a smaller DA subset) — neither is exposed here.

On-disk output layouts under ``--output-dir``
---------------------------------------------
Motion::

    <output_dir>/
        training/*.tfrecord-*
        ...
        lidar/
            training/*.tfrecord-*
            ...

Perception::

    <output_dir>/
        training/segment-*.tfrecord
        validation/segment-*.tfrecord
        testing/segment-*.tfrecord

These match what ``WODMotionParser`` and ``WODPerceptionParser`` expect in local
(non-streaming) mode.

History
-------
This module was renamed from ``motion_download.py``. The old CLI name
``py123d-womd-download`` was replaced by ``py123d-wod-download {motion|perception}``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from google.cloud import storage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DatasetSpec:
    """Per-dataset download configuration consumed by :func:`download_shards`.

    Folds bucket identity and blob → local-path mapping into one handle so the
    parallel download loop stays dataset-agnostic.
    """

    bucket_name: str
    destination_for_blob: Callable[[str, Path], Path]


# ======================================================================================
# Auth / client (generic)
# ======================================================================================


def _require_gcs():
    """Lazy import — ``google-cloud-storage`` is optional until this CLI or the
    ``stream_enabled=True`` parser mode is used."""
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise SystemExit(
            "google-cloud-storage is required for WOD downloads. Install it with:\n"
            "  pip install py123d[waymo]\n"
            "or directly:\n"
            "  pip install google-cloud-storage\n"
        ) from exc
    return storage


def resolve_gcs_client(credentials_file: Optional[Path] = None) -> "storage.Client":
    """Build a ``google.cloud.storage.Client`` using the standard auth fallback chain.

    Order:

    1. Explicit ``credentials_file`` (service-account JSON) → ``Client.from_service_account_json``.
    2. ``$GOOGLE_APPLICATION_CREDENTIALS`` or Application Default Credentials → ``Client()``.
    3. Anonymous client fallback. Works for the motion bucket; the perception bucket
       requires an authenticated client (anonymous listing returns 403).

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
    # WOD bucket reads don't need a real project; we pass a placeholder.
    try:
        import google.auth

        credentials, project = google.auth.default()
        logger.debug("Using Application Default Credentials (project=%s)", project)
        return storage.Client(credentials=credentials, project=project or "py123d-wod")
    except Exception as exc:  # DefaultCredentialsError, etc.
        logger.warning(
            "Could not create authenticated GCS client (%s); falling back to anonymous client. "
            "If you expected authenticated access, run: gcloud auth application-default login",
            exc,
        )
        return storage.Client.create_anonymous_client()


# ======================================================================================
# Selection (generic)
# ======================================================================================


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


# ======================================================================================
# Download (generic)
# ======================================================================================


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
    spec: _DatasetSpec,
    client: "storage.Client",
    blob_names: Sequence[str],
    output_dir: Path,
    max_workers: int = 8,
    overwrite: bool = False,
) -> List[Path]:
    """Download a list of shards (by blob name) into ``output_dir`` in parallel.

    The destination path for each blob is derived via ``spec.destination_for_blob``,
    which encodes the dataset-specific key → local-path convention.

    :param spec: Dataset spec (bucket name + destination mapper).
    :param client: GCS client.
    :param blob_names: Blob keys to download.
    :param output_dir: Local root directory.
    :param max_workers: Parallel download threads.
    :param overwrite: If ``False``, skip files that already exist locally.
    :return: Local paths of downloaded (or pre-existing) files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not blob_names:
        return []

    downloaded: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(
                _download_one_blob,
                client,
                spec.bucket_name,
                blob_name,
                spec.destination_for_blob(blob_name, output_dir),
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


# ======================================================================================
# WOD Motion (WOMD)
# ======================================================================================

WOMD_BUCKET_PREFIX = "waymo_open_dataset_motion_v_"
# Canonical form uses dots ("1.3.0") so Hydra CLI overrides don't get reparsed as the
# numeric literal 130 (Python treats ``1_3_0`` as an int). Internally we normalize
# dots to underscores when building the bucket name.
WOMD_DEFAULT_VERSION = "1.3.0"

MOTION_SCENARIO_SPLITS: Tuple[str, ...] = (
    "training",
    "validation",
    "testing",
    "training_20s",
    "validation_interactive",
    "testing_interactive",
)
MOTION_LIDAR_SPLITS: Tuple[str, ...] = ("training", "validation", "testing")
MOTION_SECTION_CHOICES: Tuple[str, ...] = ("scenario", "lidar", "both")


def motion_bucket_name(version: str) -> str:
    """GCS bucket name for WOMD ``version`` (accepts ``"1.3.0"`` or ``"1_3_0"``)."""
    normalized = str(version).replace(".", "_")
    return f"{WOMD_BUCKET_PREFIX}{normalized}"


def _motion_split_prefix(section: str, split: str) -> str:
    """GCS key prefix for a given motion section+split, without bucket name."""
    if section == "scenario":
        return f"uncompressed/scenario/{split}/"
    if section == "lidar":
        return f"uncompressed/lidar/{split}/"
    raise ValueError(f"Unknown motion section {section!r}; expected one of 'scenario' | 'lidar'.")


def _motion_local_split_dir(output_dir: Path, section: str, split: str) -> Path:
    """Where a motion section+split's tfrecords should land on disk."""
    if section == "scenario":
        return output_dir / split
    if section == "lidar":
        return output_dir / "lidar" / split
    raise ValueError(f"Unknown motion section {section!r}; expected one of 'scenario' | 'lidar'.")


def _motion_destination_for_blob(blob_name: str, output_dir: Path) -> Path:
    """Map a motion GCS blob name back to its on-disk destination.

    ``uncompressed/scenario/{split}/file.tfrecord-*`` → ``{output_dir}/{split}/file.tfrecord-*``
    ``uncompressed/lidar/{split}/file.tfrecord-*`` → ``{output_dir}/lidar/{split}/file.tfrecord-*``
    """
    parts = blob_name.split("/")
    # Expected shape: ["uncompressed", "scenario"|"lidar", "<split>", "<filename>"]
    if len(parts) < 4 or parts[0] != "uncompressed":
        raise ValueError(f"Unexpected motion blob layout: {blob_name!r}")
    section = parts[1]
    split = parts[2]
    filename = "/".join(parts[3:])
    return _motion_local_split_dir(output_dir, section, split) / filename


def list_motion_split_shards(
    client: "storage.Client",
    section: str,
    split: str,
    version: str = WOMD_DEFAULT_VERSION,
) -> List[str]:
    """List all tfrecord blob names for a given WOMD section+split.

    :param client: GCS client.
    :param section: ``"scenario"`` or ``"lidar"``.
    :param split: Split name (e.g. ``"training"``).
    :param version: WOMD version string (accepts ``"1.3.0"`` or ``"1_3_0"``).
    :return: Sorted list of blob names (keys within the bucket).
    """
    bucket = motion_bucket_name(version)
    prefix = _motion_split_prefix(section, split)
    blobs = client.list_blobs(bucket, prefix=prefix)
    blob_names: List[str] = [b.name for b in blobs if ".tfrecord" in b.name]
    blob_names.sort()
    return blob_names


def motion_spec(version: str = WOMD_DEFAULT_VERSION) -> _DatasetSpec:
    """Build the :class:`_DatasetSpec` for WOMD at ``version``."""
    return _DatasetSpec(
        bucket_name=motion_bucket_name(version),
        destination_for_blob=_motion_destination_for_blob,
    )


def download_motion_single_shard(
    client: "storage.Client",
    section: str,
    split: str,
    shard_idx: int,
    output_dir: Path,
    version: str = WOMD_DEFAULT_VERSION,
    overwrite: bool = False,
) -> Path:
    """Download exactly one motion shard identified by its index within the sorted shard list."""
    all_shards = list_motion_split_shards(client, section=section, split=split, version=version)
    if not all_shards:
        raise FileNotFoundError(
            f"No tfrecords found at gs://{motion_bucket_name(version)}/{_motion_split_prefix(section, split)}"
        )
    if shard_idx < 0 or shard_idx >= len(all_shards):
        raise IndexError(f"Shard index {shard_idx} out of range for {section}/{split} (have {len(all_shards)} shards).")
    blob_name = all_shards[shard_idx]
    dest = _motion_destination_for_blob(blob_name, Path(output_dir))
    return _download_one_blob(client, motion_bucket_name(version), blob_name, dest, overwrite=overwrite)


# ======================================================================================
# WOD Perception
# ======================================================================================

WOD_PERCEPTION_BUCKET_PREFIX = "waymo_open_dataset_v_"
WOD_PERCEPTION_DEFAULT_VERSION = "1.4.3"
# Key prefix inside the perception bucket — the "primary" dataset lives under
# ``individual_files/``; ``archived_files/`` (tars) and ``individual_files/domain_adaptation/``
# are intentionally out of scope.
WOD_PERCEPTION_KEY_ROOT = "individual_files"
PERCEPTION_SPLITS: Tuple[str, ...] = ("training", "validation", "testing")
WOD_PERCEPTION_SUPPORTED_VERSIONS: Tuple[str, ...] = ("1.4.3",)


def perception_bucket_name(version: str) -> str:
    """GCS bucket name for WOD Perception ``version`` (accepts ``"1.4.3"`` or ``"1_4_3"``)."""
    normalized = str(version).replace(".", "_")
    return f"{WOD_PERCEPTION_BUCKET_PREFIX}{normalized}"


def _perception_split_prefix(split: str) -> str:
    """GCS key prefix for a given perception split, without bucket name."""
    if split not in PERCEPTION_SPLITS:
        raise ValueError(f"Unknown perception split {split!r}; expected one of {PERCEPTION_SPLITS}.")
    return f"{WOD_PERCEPTION_KEY_ROOT}/{split}/"


def _perception_local_split_dir(output_dir: Path, split: str) -> Path:
    """Where a perception split's tfrecords should land on disk."""
    return output_dir / split


def _perception_destination_for_blob(blob_name: str, output_dir: Path) -> Path:
    """Map a perception GCS blob name back to its on-disk destination.

    ``individual_files/{split}/segment-*.tfrecord`` → ``{output_dir}/{split}/segment-*.tfrecord``
    """
    parts = blob_name.split("/")
    # Expected shape: ["individual_files", "<split>", "<filename>"]
    if len(parts) < 3 or parts[0] != WOD_PERCEPTION_KEY_ROOT or parts[1] not in PERCEPTION_SPLITS:
        raise ValueError(f"Unexpected perception blob layout: {blob_name!r}")
    split = parts[1]
    filename = "/".join(parts[2:])
    return _perception_local_split_dir(output_dir, split) / filename


def list_perception_split_shards(
    client: "storage.Client",
    split: str,
    version: str = WOD_PERCEPTION_DEFAULT_VERSION,
) -> List[str]:
    """List all tfrecord blob names for a given WOD Perception split.

    :param client: GCS client (must be authenticated — the perception bucket is not
        anonymously readable).
    :param split: Split name (``"training"``, ``"validation"``, or ``"testing"``).
    :param version: Perception version string (accepts ``"1.4.3"`` or ``"1_4_3"``).
    :return: Sorted list of blob names (keys within the bucket).
    """
    bucket = perception_bucket_name(version)
    prefix = _perception_split_prefix(split)
    blobs = client.list_blobs(bucket, prefix=prefix)
    blob_names: List[str] = [b.name for b in blobs if ".tfrecord" in b.name]
    blob_names.sort()
    return blob_names


def perception_spec(version: str = WOD_PERCEPTION_DEFAULT_VERSION) -> _DatasetSpec:
    """Build the :class:`_DatasetSpec` for WOD Perception at ``version``."""
    return _DatasetSpec(
        bucket_name=perception_bucket_name(version),
        destination_for_blob=_perception_destination_for_blob,
    )


# ======================================================================================
# CLI
# ======================================================================================


def _add_common_selection_args(sub_parser: argparse.ArgumentParser) -> None:
    """Add selection + download knobs shared by every subcommand."""
    sub_parser.add_argument(
        "--credentials-file",
        type=str,
        default=None,
        help=(
            "Optional service-account JSON for GCS auth. If omitted, Application Default "
            "Credentials are used (gcloud auth application-default login, or "
            "$GOOGLE_APPLICATION_CREDENTIALS). If neither is available, an anonymous "
            "client is used (works for motion only; perception requires authentication)."
        ),
    )

    selection = sub_parser.add_argument_group("shard selection (per split)")
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

    misc = sub_parser.add_argument_group("misc")
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


# ----- Motion subcommand ---------------------------------------------------------------


def _resolve_motion_output_dir(cli_output: Optional[str]) -> Path:
    """Falls back to ``$WOD_MOTION_DATA_ROOT`` then ``./data/wod_motion``."""
    if cli_output:
        resolved = Path(cli_output)
    elif os.environ.get("WOD_MOTION_DATA_ROOT"):
        resolved = Path(os.environ["WOD_MOTION_DATA_ROOT"])
    else:
        resolved = Path.cwd() / "data" / "wod_motion"
    return resolved.expanduser().resolve()


def _resolve_motion_sections(section: str) -> List[str]:
    if section == "both":
        return ["scenario", "lidar"]
    return [section]


def _motion_split_allowed_for_section(section: str, split: str) -> bool:
    if section == "scenario":
        return split in MOTION_SCENARIO_SPLITS
    if section == "lidar":
        return split in MOTION_LIDAR_SPLITS
    return False


def _add_motion_subparser(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "motion",
        help="Download Waymo Open Motion Dataset (WOMD) shards.",
        description="Download the Waymo Open Motion Dataset (or a subset of shards) from Google Cloud Storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory. Falls back to $WOD_MOTION_DATA_ROOT, then ./data/wod_motion.",
    )
    sp.add_argument(
        "--version",
        type=str,
        default=WOMD_DEFAULT_VERSION,
        help=f"WOMD version string, mapped to bucket {WOMD_BUCKET_PREFIX}<version>.",
    )
    sp.add_argument(
        "--section",
        choices=MOTION_SECTION_CHOICES,
        default="scenario",
        help=(
            "Which part of WOMD to download. 'scenario' = motion tfrecords (the usual dataset). "
            "'lidar' = the separate first-1s lidar archive. 'both' = pull both."
        ),
    )
    sp.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(set(MOTION_SCENARIO_SPLITS) | set(MOTION_LIDAR_SPLITS)),
        default=["training", "validation", "testing"],
        help="Which splits to download. Lidar section only has training/validation/testing.",
    )
    _add_common_selection_args(sp)
    sp.set_defaults(run=_run_motion)


def _plan_motion_downloads(
    client: "storage.Client",
    sections: Sequence[str],
    splits: Sequence[str],
    version: str,
    shard_indices: Optional[Sequence[int]],
    num_shards: Optional[int],
    sample_random: bool,
    seed: int,
) -> List[str]:
    """Enumerate the motion blob names that would be downloaded given CLI selectors."""
    plan: List[str] = []
    for section in sections:
        for split in splits:
            if not _motion_split_allowed_for_section(section, split):
                logger.debug("Skip %s split %s — not available in this section.", section, split)
                continue
            all_shards = list_motion_split_shards(client, section=section, split=split, version=version)
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


def _run_motion(args: argparse.Namespace) -> int:
    output_dir = _resolve_motion_output_dir(args.output_dir)
    bucket = motion_bucket_name(args.version)
    sections = _resolve_motion_sections(args.section)

    credentials_file = Path(args.credentials_file) if args.credentials_file else None
    client = resolve_gcs_client(credentials_file)

    plan = _plan_motion_downloads(
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
        spec=motion_spec(args.version),
        client=client,
        blob_names=plan,
        output_dir=output_dir,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )
    logger.info("Done. Data written to %s", output_dir)
    return 0


# ----- Perception subcommand -----------------------------------------------------------


def _resolve_perception_output_dir(cli_output: Optional[str]) -> Path:
    """Falls back to ``$WOD_PERCEPTION_DATA_ROOT`` then ``./data/wod_perception``."""
    if cli_output:
        resolved = Path(cli_output)
    elif os.environ.get("WOD_PERCEPTION_DATA_ROOT"):
        resolved = Path(os.environ["WOD_PERCEPTION_DATA_ROOT"])
    else:
        resolved = Path.cwd() / "data" / "wod_perception"
    return resolved.expanduser().resolve()


def _add_perception_subparser(subparsers: argparse._SubParsersAction) -> None:
    sp = subparsers.add_parser(
        "perception",
        help="Download Waymo Open Dataset Perception segments.",
        description=(
            "Download the Waymo Open Perception Dataset (or a subset of segments) from "
            "Google Cloud Storage. Note: each perception segment is roughly 1 GB — small "
            "values of --num-shards already imply multiple GB of download traffic. "
            "Requires an authenticated GCS client (the bucket is not anonymously readable)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory. Falls back to $WOD_PERCEPTION_DATA_ROOT, then ./data/wod_perception.",
    )
    sp.add_argument(
        "--version",
        type=str,
        choices=list(WOD_PERCEPTION_SUPPORTED_VERSIONS),
        default=WOD_PERCEPTION_DEFAULT_VERSION,
        help=f"Perception version, mapped to bucket {WOD_PERCEPTION_BUCKET_PREFIX}<version>.",
    )
    sp.add_argument(
        "--splits",
        nargs="+",
        choices=list(PERCEPTION_SPLITS),
        default=list(PERCEPTION_SPLITS),
        help="Which splits to download.",
    )
    _add_common_selection_args(sp)
    sp.set_defaults(run=_run_perception)


def _plan_perception_downloads(
    client: "storage.Client",
    splits: Sequence[str],
    version: str,
    shard_indices: Optional[Sequence[int]],
    num_shards: Optional[int],
    sample_random: bool,
    seed: int,
) -> List[str]:
    """Enumerate the perception blob names that would be downloaded given CLI selectors."""
    plan: List[str] = []
    for split in splits:
        all_shards = list_perception_split_shards(client, split=split, version=version)
        selected = select_shards(
            all_shards,
            shard_indices=shard_indices,
            num_shards=num_shards,
            sample_random=sample_random,
            seed=seed,
        )
        logger.info("Selected %d / %d segments from %s", len(selected), len(all_shards), split)
        plan.extend(selected)
    return plan


def _run_perception(args: argparse.Namespace) -> int:
    output_dir = _resolve_perception_output_dir(args.output_dir)
    bucket = perception_bucket_name(args.version)

    credentials_file = Path(args.credentials_file) if args.credentials_file else None
    client = resolve_gcs_client(credentials_file)

    plan = _plan_perception_downloads(
        client=client,
        splits=args.splits,
        version=args.version,
        shard_indices=args.shard_indices,
        num_shards=args.num_shards,
        sample_random=args.sample_random,
        seed=args.seed,
    )

    logger.info("Target directory: %s", output_dir)
    logger.info("Bucket:           gs://%s/", bucket)
    logger.info("Splits:           %s", ", ".join(args.splits))
    logger.info("Segments selected: %d", len(plan))
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
        logger.error("No segments selected — nothing to download.")
        return 1

    download_shards(
        spec=perception_spec(args.version),
        client=client,
        blob_names=plan,
        output_dir=output_dir,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )
    logger.info("Done. Data written to %s", output_dir)
    return 0


# ----- Top-level dispatcher ------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="py123d-wod-download",
        description="Download Waymo Open Dataset (motion or perception) shards from Google Cloud Storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True, metavar="{motion,perception}")
    _add_motion_subparser(subparsers)
    _add_perception_subparser(subparsers)

    args = parser.parse_args(argv)

    if args.shard_indices is not None and args.num_shards is not None:
        parser.error("--shard-indices and --num-shards are mutually exclusive")
    if args.num_shards is not None and args.num_shards <= 0:
        parser.error("--num-shards must be a positive integer")
    if args.sample_random and args.num_shards is None:
        parser.error("--random requires --num-shards")

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
