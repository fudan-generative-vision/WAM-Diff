# LINT_ME
"""Filesystem operations supporting local and remote (obs://, s3://, gs://) paths."""
from __future__ import annotations

import io
import json
import posixpath
from pathlib import Path
from typing import IO, List, Union
from urllib.parse import urlsplit, urlunsplit

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

try:
    import moxing as mox
    _HAS_MOX = True
except ImportError:
    mox = None  # type: ignore
    _HAS_MOX = False

PathLike = Union[str, Path]

# ---------- helpers ----------


def _is_remote(p: PathLike) -> bool:
    """
    Check if path is a remote storage URL (obs://, s3://, gs://).
    """
    s = str(p)
    if s.startswith(("obs://", "s3://", "gs://")):
        return True
    return False


def join(base: PathLike, *parts: PathLike) -> str:
    """
    Join path components for local or remote paths.
    """
    b = str(base)
    if _is_remote(b):
        scheme, netloc, path, query, frag = urlsplit(b)
        segs = [path] + [str(p) for p in parts]
        new_path = posixpath.join(*[s.lstrip("/") for s in segs if s])
        if not new_path.startswith("/"):
            new_path = "/" + new_path
        return urlunsplit((scheme, netloc, new_path, query, frag))
    return str(Path(b).joinpath(*map(str, parts)))

def exists(p: PathLike) -> bool:
    """
    Check if path exists (local or remote).
    """
    if _is_remote(p):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        return bool(mox.file.exists(str(p)))
    return Path(p).exists()


def isdir(p: PathLike) -> bool:
    """
    Check if path is a directory (local or remote).
    """
    if _is_remote(p):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        return bool(mox.file.is_directory(str(p)))
    return Path(p).is_dir()


def listdir(p: PathLike) -> List[str]:
    """
    List directory contents for local or obs:// paths.
    Remote returns full paths, local returns names (like os.listdir).
    """
    if _is_remote(p):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        return mox.file.list_directory(str(p))
    return [x.name for x in Path(p).iterdir()]


def basename(p: PathLike) -> str:
    """
    Return the final component of a path, works for local and obs:// URLs.
    """
    s = str(p)
    if _is_remote(s):
        # Strip trailing slash if present, then take the last segment
        return s.rstrip("/").rsplit("/",maxsplit=1)[-1]
    return Path(s).name


def split(p: PathLike) -> tuple[str, str]:
    """
    Split path into (head, tail).
    For remote (obs://...), head keeps scheme+netloc+parent, tail is the last component.
    For local, same as os.path.split.
    """
    s = str(p)
    if _is_remote(s):
        scheme, netloc, path, query, frag = urlsplit(s)
        parts = path.rstrip("/").rsplit("/",maxsplit=1)
        tail = parts[-1] if parts else ""
        head_path = "/".join(parts[:-1])
        if head_path and not head_path.startswith("/"):
            head_path = "/" + head_path
        head = urlunsplit((scheme, netloc, head_path, query, frag))
        return head, tail
    return str(Path(s).parent), Path(s).name


def splitext(p: PathLike) -> tuple[str, str]:
    """
    Split path into (root, ext).
    Works like os.path.splitext, for both local and remote.
    """
    s = str(p)
    if _is_remote(s):
        base = basename(s)  # last part only
        root, ext = posixpath.splitext(base)
        # Reconstruct full root path (without extension)
        head, _ = split(s)
        return join(head, root), ext
    root, ext = Path(s).with_suffix("").as_posix(), Path(s).suffix
    return root, ext


def open_file(p: PathLike, mode: str = "r", **kwargs) -> IO:
    """
    Open a file for local or remote (obs://, s3://, gs://) paths.
    """
    if _is_remote(p):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        return mox.file.File(str(p), mode, **kwargs)  # type: ignore
    return open(p, mode, **kwargs)  # pylint: disable=W1514


def json_load(p: PathLike, **json_kwargs):
    """
    Load JSON or JSONL files automatically.
    Returns:
        - list[dict]: for JSONL or list-based JSON
        - dict: for JSON object
    """
    encoding = json_kwargs.pop("encoding", "utf-8")

    with open_file(p, "r", encoding=encoding) as f:
        f.seek(0)

        # JSONL: detect line-delimited JSON
        if p.endswith(".jsonl"):
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]
        f.seek(0)
        return json.load(f, **json_kwargs)


def image_open(p: PathLike):
    """
    Open an image from local or remote path.
    """

    if str(p).endswith(".parquet"):
        with open_file(p, "rb") as fp:
            file_data = fp.read()
        table = pq.read_table(pa.py_buffer(file_data)).to_pandas()
        images = {}
        for cam_id in table.columns:
            data = table[cam_id].iloc[0]  # 取第一行的 bytes 数据
            try:
                img = Image.open(io.BytesIO(data))
                img.load()  # 强制加载，释放文件句柄
                images[cam_id] = img
            except Exception as e:
                raise RuntimeError(
                    f"Failed to decode image from camera '{cam_id}' in {p}: {e}"
                ) from e
        return images["cam_1"]

    f = open_file(p, "rb")
    try:
        img = Image.open(f)
        # Force load into memory so we can close the file immediately
        img.load()
        f.close()
        return img
    except Exception:
        f.close()
        raise


def makedirs(p: PathLike, exist_ok: bool = True) -> None:
    """
    Create directories (local or remote).
    - Local: same as Path(...).mkdir(parents=True, exist_ok=...)
    - Remote: uses mox.file.make_dirs
    """
    s = str(p)
    if _is_remote(s):
        if not _HAS_MOX:
            raise RuntimeError("moxing not installed.")
        # moxing's make_dirs creates all necessary parent dirs automatically
        if exist_ok:
            # No harm if it already exists
            if not mox.file.exists(s):
                mox.file.make_dirs(s)
        else:
            if mox.file.exists(s):
                raise FileExistsError(f"Remote path already exists: {s}")
            mox.file.make_dirs(s)
    else:
        Path(s).mkdir(parents=True, exist_ok=exist_ok)
