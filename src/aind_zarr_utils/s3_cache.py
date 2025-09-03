from __future__ import annotations

import hashlib
import os
import pathlib
import re
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class CacheManager(AbstractContextManager):
    """
    Manage a cache directory for S3 downloads.

    Modes:
      1) cache_dir provided  -> use it (persistent; caller owns lifetime)
      2) persistent=True     -> use a conventional path (persistent)
      3) persistent=False    -> create a TemporaryDirectory (ephemeral;
         auto-clean)

    Usage:
      with CacheManager(persistent=False) as cm:
          # pass to get_local_path_for_resource(..., cache_dir=cm.dir)
          path = cm.dir
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        *,
        persistent: bool = True,
    ):
        self._temp: Optional[tempfile.TemporaryDirectory] = None
        if cache_dir is not None:
            self.dir = Path(cache_dir).expanduser().resolve()
            self.dir.mkdir(parents=True, exist_ok=True)
            self._owns = False
        elif persistent:
            # Conventional persistent location
            self.dir = (Path.home() / ".cache" / "s3-ants").resolve()
            self.dir.mkdir(parents=True, exist_ok=True)
            self._owns = False
        else:
            # Ephemeral cache: cleaned up on exit
            self._temp = tempfile.TemporaryDirectory()
            self.dir = Path(self._temp.name).resolve()
            self._owns = True

    def __enter__(self) -> "CacheManager":
        return self

    def close(self) -> None:
        # If we created a TemporaryDirectory, clean it up
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass(frozen=True)
class LocalizedResource:
    """Result of resolving a local or S3 resource to a local filesystem
    path."""

    path: pathlib.Path  # local path to use (e.g., for ants.read_transform)
    from_cache: bool  # True if we used (or created) an S3-backed cache
    source: str  # "local" or "s3"


def _default_cache_dir() -> pathlib.Path:
    d = pathlib.Path(tempfile.gettempdir()) / "s3-ants-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_s3_uri(s: Union[str, os.PathLike]) -> bool:
    try:
        p = urlparse(str(s))
        return bool(p.scheme.lower() == "s3" and p.netloc and p.path)
    except Exception:
        return False


def _safe_slug(s: str, *, maxlen: int = 120) -> str:
    """
    Make a filesystem-safe slug for cache prefixes:
      - keep letters, digits, ., _, -
      - convert everything else to _
      - collapse repeats; trim to maxlen
    """
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if len(s) > maxlen:
        s = s[:maxlen]
    return s or "root"


def _cache_name(bucket: str, key: str, etag: str) -> str:
    """
    Build a cache filename that *preserves the original basename*
    (e.g., foo.nii.gz) and prepends a unique prefix derived from bucket, parent
    path, and ETag.
    Example:
      my-bucket__path_to__a1b2c3d4e5f6__foo.nii.gz
    """
    base = pathlib.Path(key).name  # keeps .nii.gz intact
    parent = pathlib.Path(key).parent.as_posix()
    short = hashlib.sha256(
        f"{bucket}/{key}:{etag}".encode("utf-8")
    ).hexdigest()[:12]

    prefix_bits = [_safe_slug(bucket)]
    if parent not in ("", "."):
        prefix_bits.append(_safe_slug(parent.replace("/", "_")))
    prefix_bits.append(short)

    prefix = "__".join(prefix_bits)
    return f"{prefix}__{base}"


def _safe_atomic_replace(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def _ensure_client(
    anonymous: bool, max_concurrency: int, s3_client: Optional[S3Client] = None
):
    if s3_client is not None:
        return s3_client
    cfg = Config(
        retries={"max_attempts": 10, "mode": "standard"},
        max_pool_connections=max_concurrency,
        signature_version=UNSIGNED if anonymous else None,
    )
    return boto3.client("s3", config=cfg)


def _head_or_peek_byte(
    s3_client: S3Client, bucket: str, key: str
) -> tuple[str, Optional[int]]:
    """
    Return (etag, size) using HEAD when allowed; otherwise fall back to a
    1-byte GET.

    Some public buckets allow GetObject but block HeadObject; in that case we:
      - do GetObject with Range='bytes=0-0'
      - parse 'Content-Range: bytes 0-0/TOTAL'
    """
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        etag = head.get("ETag", "").strip('"')
        size = head.get("ContentLength")
        return etag, size
    except ClientError as e:
        code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        # Fall back if 403/400/etc. (blocked HEAD), but still verify object
        # exists via GET
        if code in (400, 401, 403, 404):
            resp = s3_client.get_object(
                Bucket=bucket, Key=key, Range="bytes=0-0"
            )
            # ETag may be in top-level or headers depending on signer/proxy
            etag = (
                resp.get("ETag")
                or resp["ResponseMetadata"]["HTTPHeaders"].get("etag", "")
            ).strip('"')
            cr = resp["ResponseMetadata"]["HTTPHeaders"].get("content-range")
            size = None
            if cr:
                # e.g., "bytes 0-0/12345" -> 12345
                m = re.search(r"/(\d+)$", cr)
                if m:
                    size = int(m.group(1))
            return etag, size
        raise


def get_local_path_for_resource(
    uri_or_path: Union[str, os.PathLike],
    *,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    s3_client: Optional[S3Client] = None,
    anonymous: bool = False,
    max_concurrency: int = 16,
    multipart_threshold_bytes: int = 8 * 1024 * 1024,  # 8 MB
) -> LocalizedResource:
    """
    Resolve an S3 object or local path to a local, readable file path.

    - Local path: returned as-is.
    - s3://bucket/key: downloads into a cache keyed by ETag and returns that
    path.
    - Supports anonymous (unsigned) access for public buckets with
    `anonymous=True`.
    - Atomic on writes; safe for concurrent processes.
    """
    p = pathlib.Path(str(uri_or_path))
    if not _is_s3_uri(uri_or_path):
        if not p.exists():
            raise FileNotFoundError(f"Local path not found: {p}")
        return LocalizedResource(
            path=p.resolve(), from_cache=False, source="local"
        )

    parsed = urlparse(str(uri_or_path))
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    s3 = _ensure_client(
        anonymous=anonymous,
        max_concurrency=max_concurrency,
        s3_client=s3_client,
    )

    # Discover identity/size even when HEAD is blocked
    try:
        etag, size = _head_or_peek_byte(s3, bucket, key)
    except ClientError as e:
        raise FileNotFoundError(
            f"S3 object not found or inaccessible: s3://{bucket}/{key}"
        ) from e

    cache_root = pathlib.Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_name = _cache_name(bucket, key, etag)
    cache_path = cache_root / cache_name

    if cache_path.exists():
        try:
            if size is None or cache_path.stat().st_size == size:
                return LocalizedResource(
                    path=cache_path, from_cache=True, source="s3"
                )
        except OSError:
            pass  # re-download

    part_path = cache_path.with_suffix(cache_path.suffix + ".part")
    try:
        if part_path.exists():
            part_path.unlink()
    except OSError:
        pass

    from boto3.s3.transfer import TransferConfig

    t_config = TransferConfig(
        multipart_threshold=multipart_threshold_bytes,
        max_concurrency=max_concurrency,
        multipart_chunksize=8 * 1024 * 1024,
        use_threads=True,
    )

    s3.download_file(
        Bucket=bucket, Key=key, Filename=str(part_path), Config=t_config
    )
    _safe_atomic_replace(part_path, cache_path)

    return LocalizedResource(path=cache_path, from_cache=True, source="s3")
