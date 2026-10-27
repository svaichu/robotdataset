"""OXEDataset with a cached dataset must not need the bucket; a missing one must say why (offline, faked TFDS)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorflow", reason="oxe extras not installed")

import robotdataset.oxe_dataset as oxe  # noqa: E402
import test_oxe_dataset as oxe_t  # noqa: E402


def _cache_metadata(root: Path, name: str = "droid") -> None:
    d = root / "oxe" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "dataset_info.json").write_text("{}")


def _bucket_down(monkeypatch: pytest.MonkeyPatch, reason: str = "libcurl code 77") -> list:
    """Every bucket access fails: an empty listing (as TensorFlow gives it) and a failing listdir."""
    calls: list = []

    def listdir(path):
        calls.append(path)
        raise RuntimeError(reason)

    monkeypatch.setattr(oxe, "_DATASET_CACHE", {}, raising=False)
    monkeypatch.setattr(oxe.tf.io, "gfile", SimpleNamespace(listdir=listdir, isdir=lambda p: False), raising=False)
    return calls


def test_cached_dataset_loads_without_touching_the_bucket(monkeypatch, tmp_path):
    oxe_t._patch(monkeypatch)
    _cache_metadata(tmp_path)
    ds = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))
    assert ds.num_episodes == 2

    calls = _bucket_down(monkeypatch)  # a second process on a node that cannot reach gs://
    ds2 = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))
    assert ds2.num_episodes == 2 and ds2.sample().batch_size is not None
    assert calls == [], "a fully cached dataset must not list the bucket"


def test_dataset_path_is_looked_up_lazily(monkeypatch, tmp_path):
    oxe_t._patch(monkeypatch)
    _cache_metadata(tmp_path)
    first = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0], root=str(tmp_path))
    assert first.dataset_path == "gs://gresearch/robotics/droid/1.0.1"  # building episode 0 needed the bucket
    again = oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0], root=str(tmp_path))
    assert again._dataset_path is None  # everything cached: never resolved
    assert again.dataset_path == "gs://gresearch/robotics/droid/1.0.1" and again._dataset_path is not None


def test_missing_episodes_without_bucket_say_why(monkeypatch, tmp_path):
    oxe_t._patch(monkeypatch)
    _cache_metadata(tmp_path)
    oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0], root=str(tmp_path))
    _bucket_down(monkeypatch, reason="libcurl code 77 meaning 'Problem with the SSL CA cert'")
    with pytest.raises(RuntimeError, match=r"Cannot read the OXE bucket gs://gresearch/robotics .*libcurl code 77.*CURL_CA_BUNDLE"):
        oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0, 1], root=str(tmp_path))  # episode 1 is not cached


def test_the_reason_comes_from_exists_when_listdir_only_says_not_found(monkeypatch, tmp_path):
    """TensorFlow reports a CA-bundle failure on exists() but a plain 'not found' on listdir()."""
    oxe_t._patch(monkeypatch)
    _bucket_down(monkeypatch)

    def exists(path):
        raise RuntimeError("libcurl code 77 meaning 'Problem with the SSL CA cert'")

    def listdir(path):
        raise FileNotFoundError(f"Could not find directory {path}")

    monkeypatch.setattr(oxe.tf.io, "gfile", SimpleNamespace(exists=exists, listdir=listdir, isdir=lambda p: False), raising=False)
    with pytest.raises(RuntimeError, match=r"libcurl code 77.*CURL_CA_BUNDLE"):
        oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0], root=str(tmp_path / "fresh"))


def test_first_use_without_bucket_is_not_reported_as_unknown_dataset(monkeypatch, tmp_path):
    oxe_t._patch(monkeypatch)
    _bucket_down(monkeypatch)
    with pytest.raises(RuntimeError, match="Cannot read the OXE bucket"):
        oxe.OXEDataset(dataset_name="droid", split="train", episodes=[0], root=str(tmp_path / "fresh"))


def test_unknown_dataset_still_lists_the_available_ones(monkeypatch, tmp_path):
    oxe_t._patch(monkeypatch)
    with pytest.raises(ValueError, match=r"Unknown dataset 'nope'.*Available datasets: droid"):
        oxe.OXEDataset(dataset_name="nope", split="train", episodes=[0], root=str(tmp_path / "fresh"))


def _clean_ca_env(monkeypatch) -> None:
    """Leave both variables unset, and make sure monkeypatch removes what the code under test sets."""
    for name in ("CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)


def test_ca_bundle_fallback(monkeypatch):
    _clean_ca_env(monkeypatch)
    present = {"/etc/pki/tls/certs/ca-bundle.crt"}
    monkeypatch.setattr(oxe.os.path, "exists", lambda p: p in present)
    oxe._ensure_ca_bundle()
    assert oxe.os.environ["CURL_CA_BUNDLE"] == "/etc/pki/tls/certs/ca-bundle.crt"


def test_ca_bundle_user_choice_stands(monkeypatch):
    _clean_ca_env(monkeypatch)
    monkeypatch.setenv("CURL_CA_BUNDLE", "/user/choice.pem")
    monkeypatch.setattr(oxe.os.path, "exists", lambda p: True)
    oxe._ensure_ca_bundle()
    assert oxe.os.environ["CURL_CA_BUNDLE"] == "/user/choice.pem"


def test_ca_bundle_from_ssl_cert_file(monkeypatch):
    """The plugin ignores SSL_CERT_FILE, so an existing one is handed over as CURL_CA_BUNDLE."""
    _clean_ca_env(monkeypatch)
    monkeypatch.setenv("SSL_CERT_FILE", "/somewhere/else.pem")
    monkeypatch.setattr(oxe.os.path, "exists", lambda p: p == "/somewhere/else.pem")
    oxe._ensure_ca_bundle()
    assert oxe.os.environ["CURL_CA_BUNDLE"] == "/somewhere/else.pem"


def test_ca_bundle_untouched_when_the_default_exists(monkeypatch):
    _clean_ca_env(monkeypatch)
    monkeypatch.setattr(oxe.os.path, "exists", lambda p: p == "/etc/ssl/certs/ca-certificates.crt")
    oxe._ensure_ca_bundle()
    assert "CURL_CA_BUNDLE" not in oxe.os.environ
