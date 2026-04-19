"""Unit tests for the WOMD GCS downloader.

All tests mock ``google.cloud.storage.Client`` — none of them touch the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from py123d.parser.wod import motion_download


class _FakeBlob:
    """Stand-in for ``google.cloud.storage.Blob`` used in listing/download mocks."""

    def __init__(self, name: str, payload: bytes = b"fake-tfrecord-data") -> None:
        self.name = name
        self._payload = payload

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_bytes(self._payload)


class _FakeBucket:
    def __init__(self, name: str, blobs: List[_FakeBlob]) -> None:
        self.name = name
        self._blobs = blobs

    def blob(self, blob_name: str) -> _FakeBlob:
        for b in self._blobs:
            if b.name == blob_name:
                return b
        raise KeyError(blob_name)


class _FakeClient:
    """Minimal ``storage.Client`` surface used by the downloader."""

    def __init__(self, blobs_by_bucket: dict) -> None:
        self._blobs_by_bucket = blobs_by_bucket

    def list_blobs(self, bucket_name: str, prefix: str = ""):
        return [b for b in self._blobs_by_bucket.get(bucket_name, []) if b.name.startswith(prefix)]

    def bucket(self, bucket_name: str) -> _FakeBucket:
        return _FakeBucket(bucket_name, self._blobs_by_bucket.get(bucket_name, []))


def _make_fake_client_with_three_splits() -> _FakeClient:
    bucket = f"{motion_download.WOMD_BUCKET_PREFIX}{motion_download.WOMD_DEFAULT_VERSION}"
    blobs: List[_FakeBlob] = []
    for split in ("training", "validation", "testing"):
        for idx in range(5):
            blobs.append(_FakeBlob(f"uncompressed/scenario/{split}/{split}.tfrecord-{idx:05d}-of-00005"))
    for split in ("training", "validation"):
        for idx in range(3):
            blobs.append(_FakeBlob(f"uncompressed/lidar/{split}/lidar.tfrecord-{idx:05d}-of-00003"))
    # A non-tfrecord file that list_split_shards must filter out.
    blobs.append(_FakeBlob("uncompressed/scenario/training/README.txt"))
    return _FakeClient({bucket: blobs})


# ----------------------------------------------------------------------------------------------------------------------
# Listing + selection
# ----------------------------------------------------------------------------------------------------------------------


def test_list_split_shards_filters_non_tfrecords() -> None:
    client = _make_fake_client_with_three_splits()
    shards = motion_download.list_split_shards(client, section="scenario", split="training")
    assert len(shards) == 5
    assert all(".tfrecord" in s for s in shards)
    assert all(s.startswith("uncompressed/scenario/training/") for s in shards)
    assert shards == sorted(shards), "Shards should be sorted."


def test_list_split_shards_lidar_uses_different_prefix() -> None:
    client = _make_fake_client_with_three_splits()
    shards = motion_download.list_split_shards(client, section="lidar", split="training")
    assert len(shards) == 3
    assert all(s.startswith("uncompressed/lidar/training/") for s in shards)


def test_select_shards_defaults_to_full_list() -> None:
    shards = [f"a/b/file-{i:03d}" for i in range(10)]
    assert motion_download.select_shards(shards) == shards


def test_select_shards_num_shards_takes_first_n() -> None:
    shards = [f"a/b/file-{i:03d}" for i in range(10)]
    assert motion_download.select_shards(shards, num_shards=3) == shards[:3]


def test_select_shards_random_sample_is_deterministic_with_seed() -> None:
    shards = [f"a/b/file-{i:03d}" for i in range(20)]
    out_a = motion_download.select_shards(shards, num_shards=5, sample_random=True, seed=42)
    out_b = motion_download.select_shards(shards, num_shards=5, sample_random=True, seed=42)
    out_c = motion_download.select_shards(shards, num_shards=5, sample_random=True, seed=123)
    assert out_a == out_b
    assert out_a != out_c
    assert len(out_a) == 5
    assert out_a == sorted(out_a)


def test_select_shards_by_indices() -> None:
    shards = [f"a/b/file-{i:03d}" for i in range(10)]
    assert motion_download.select_shards(shards, shard_indices=[0, 3, 9]) == [shards[0], shards[3], shards[9]]


def test_select_shards_indices_out_of_range_raises() -> None:
    shards = [f"a/b/file-{i:03d}" for i in range(3)]
    with pytest.raises(IndexError):
        motion_download.select_shards(shards, shard_indices=[5])


def test_select_shards_num_greater_than_total_returns_all() -> None:
    shards = [f"a/b/file-{i:03d}" for i in range(3)]
    assert motion_download.select_shards(shards, num_shards=10) == shards


# ----------------------------------------------------------------------------------------------------------------------
# Destination path derivation
# ----------------------------------------------------------------------------------------------------------------------


def test_destination_for_blob_scenario(tmp_path: Path) -> None:
    dest = motion_download._destination_for_blob(
        "uncompressed/scenario/training/training.tfrecord-00000-of-01000",
        tmp_path,
    )
    assert dest == tmp_path / "training" / "training.tfrecord-00000-of-01000"


def test_destination_for_blob_lidar(tmp_path: Path) -> None:
    dest = motion_download._destination_for_blob(
        "uncompressed/lidar/validation/validation.tfrecord-00003-of-00150",
        tmp_path,
    )
    assert dest == tmp_path / "lidar" / "validation" / "validation.tfrecord-00003-of-00150"


def test_destination_for_blob_rejects_unknown_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        motion_download._destination_for_blob("something/else/file.tfrecord", tmp_path)


# ----------------------------------------------------------------------------------------------------------------------
# download_shards end-to-end with a fake client
# ----------------------------------------------------------------------------------------------------------------------


def test_download_shards_writes_files_in_expected_layout(tmp_path: Path) -> None:
    client = _make_fake_client_with_three_splits()
    blob_names = [
        "uncompressed/scenario/validation/validation.tfrecord-00000-of-00005",
        "uncompressed/scenario/validation/validation.tfrecord-00001-of-00005",
        "uncompressed/lidar/training/lidar.tfrecord-00000-of-00003",
    ]
    paths = motion_download.download_shards(
        client=client,
        blob_names=blob_names,
        output_dir=tmp_path,
        max_workers=2,
    )
    assert len(paths) == 3
    assert (tmp_path / "validation" / "validation.tfrecord-00000-of-00005").exists()
    assert (tmp_path / "validation" / "validation.tfrecord-00001-of-00005").exists()
    assert (tmp_path / "lidar" / "training" / "lidar.tfrecord-00000-of-00003").exists()


def test_download_shards_skips_existing_files(tmp_path: Path) -> None:
    client = _make_fake_client_with_three_splits()
    blob_name = "uncompressed/scenario/validation/validation.tfrecord-00000-of-00005"

    existing = tmp_path / "validation" / "validation.tfrecord-00000-of-00005"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"preexisting-content")

    motion_download.download_shards(
        client=client,
        blob_names=[blob_name],
        output_dir=tmp_path,
        max_workers=1,
        overwrite=False,
    )
    assert existing.read_bytes() == b"preexisting-content"


def test_download_shards_overwrite_replaces_existing(tmp_path: Path) -> None:
    client = _make_fake_client_with_three_splits()
    blob_name = "uncompressed/scenario/validation/validation.tfrecord-00000-of-00005"

    existing = tmp_path / "validation" / "validation.tfrecord-00000-of-00005"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"preexisting-content")

    motion_download.download_shards(
        client=client,
        blob_names=[blob_name],
        output_dir=tmp_path,
        max_workers=1,
        overwrite=True,
    )
    assert existing.read_bytes() == b"fake-tfrecord-data"


def test_download_shards_empty_plan_is_noop(tmp_path: Path) -> None:
    client = _make_fake_client_with_three_splits()
    assert motion_download.download_shards(client=client, blob_names=[], output_dir=tmp_path) == []


def test_download_single_shard(tmp_path: Path) -> None:
    client = _make_fake_client_with_three_splits()
    dest = motion_download.download_single_shard(
        client=client,
        section="scenario",
        split="testing",
        shard_idx=2,
        output_dir=tmp_path,
    )
    assert dest == tmp_path / "testing" / "testing.tfrecord-00002-of-00005"
    assert dest.exists()


def test_download_single_shard_invalid_index(tmp_path: Path) -> None:
    client = _make_fake_client_with_three_splits()
    with pytest.raises(IndexError):
        motion_download.download_single_shard(
            client=client, section="scenario", split="testing", shard_idx=99, output_dir=tmp_path
        )


# ----------------------------------------------------------------------------------------------------------------------
# CLI layer: arg parsing + plan enumeration
# ----------------------------------------------------------------------------------------------------------------------


def test_cli_rejects_shard_indices_and_num_shards_together() -> None:
    with pytest.raises(SystemExit):
        motion_download._parse_args(["--num-shards", "3", "--shard-indices", "0", "1"])


def test_cli_rejects_random_without_num_shards() -> None:
    with pytest.raises(SystemExit):
        motion_download._parse_args(["--random"])


def test_cli_rejects_non_positive_num_shards() -> None:
    with pytest.raises(SystemExit):
        motion_download._parse_args(["--num-shards", "0"])


def test_plan_downloads_skips_lidar_for_unsupported_split() -> None:
    client = _make_fake_client_with_three_splits()
    # 'training_20s' is a scenario-only split; lidar must be skipped silently.
    plan = motion_download._plan_downloads(
        client=client,
        sections=["lidar"],
        splits=["training_20s"],
        version=motion_download.WOMD_DEFAULT_VERSION,
        shard_indices=None,
        num_shards=None,
        sample_random=False,
        seed=0,
    )
    assert plan == []


def test_plan_downloads_both_sections_mixes_scenario_and_lidar() -> None:
    client = _make_fake_client_with_three_splits()
    plan = motion_download._plan_downloads(
        client=client,
        sections=["scenario", "lidar"],
        splits=["validation"],
        version=motion_download.WOMD_DEFAULT_VERSION,
        shard_indices=None,
        num_shards=2,
        sample_random=False,
        seed=0,
    )
    # 2 scenario + 2 lidar = 4
    assert len(plan) == 4
    assert any("scenario/validation" in b for b in plan)
    assert any("lidar/validation" in b for b in plan)


def test_main_dry_run_does_not_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_client_with_three_splits()
    monkeypatch.setattr(motion_download, "resolve_gcs_client", lambda credentials_file=None: fake)

    rc = motion_download.main(
        [
            "--output-dir",
            str(tmp_path),
            "--splits",
            "validation",
            "--num-shards",
            "2",
            "--dry-run",
        ]
    )
    assert rc == 0
    # No files should have been written.
    assert list(tmp_path.rglob("*.tfrecord*")) == []


def test_main_downloads_expected_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_client_with_three_splits()
    monkeypatch.setattr(motion_download, "resolve_gcs_client", lambda credentials_file=None: fake)

    rc = motion_download.main(
        [
            "--output-dir",
            str(tmp_path),
            "--splits",
            "validation",
            "--num-shards",
            "2",
        ]
    )
    assert rc == 0
    out_files = sorted((tmp_path / "validation").glob("*.tfrecord*"))
    assert len(out_files) == 2


# ----------------------------------------------------------------------------------------------------------------------
# Auth resolution
# ----------------------------------------------------------------------------------------------------------------------


def test_resolve_gcs_client_uses_service_account_when_file_given(tmp_path: Path) -> None:
    key_file = tmp_path / "key.json"
    key_file.write_text("{}")

    mock_storage_module = MagicMock()
    mock_storage_module.Client.from_service_account_json.return_value = "SA_CLIENT"
    with patch.object(motion_download, "_require_gcs", return_value=mock_storage_module):
        client = motion_download.resolve_gcs_client(key_file)
    assert client == "SA_CLIENT"
    mock_storage_module.Client.from_service_account_json.assert_called_once_with(str(key_file))


def test_resolve_gcs_client_missing_service_account_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"
    mock_storage_module = MagicMock()
    with patch.object(motion_download, "_require_gcs", return_value=mock_storage_module):
        with pytest.raises(FileNotFoundError):
            motion_download.resolve_gcs_client(missing)


def test_resolve_gcs_client_falls_back_to_default() -> None:
    mock_storage_module = MagicMock()
    mock_storage_module.Client.return_value = MagicMock(project="my-project")
    with patch.object(motion_download, "_require_gcs", return_value=mock_storage_module):
        client = motion_download.resolve_gcs_client(None)
    assert client is mock_storage_module.Client.return_value


def test_resolve_gcs_client_falls_back_to_anonymous_on_default_failure() -> None:
    mock_storage_module = MagicMock()
    mock_storage_module.Client.side_effect = RuntimeError("no credentials")
    mock_storage_module.Client.create_anonymous_client.return_value = "ANON"
    with patch.object(motion_download, "_require_gcs", return_value=mock_storage_module):
        client = motion_download.resolve_gcs_client(None)
    assert client == "ANON"


def test_resolve_output_dir_env_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WOD_MOTION_DATA_ROOT", str(tmp_path))
    assert motion_download._resolve_output_dir(None) == tmp_path.resolve()


def test_resolve_output_dir_cli_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WOD_MOTION_DATA_ROOT", str(tmp_path / "env"))
    cli = tmp_path / "cli"
    cli.mkdir()
    assert motion_download._resolve_output_dir(str(cli)) == cli.resolve()
