"""Local-only data releases and durable single-writer pipeline state.

Cloud readers do not opt into data_scope and retain their packaged data paths.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .paths import DATA_DIR, OBJECT_TYPES, normalize_translation_provider


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Invalid metadata file: {path.name}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows readers/virus scanners can briefly deny replacement.
                # Keep the previous complete JSON visible throughout the retry.
                if attempt == 7:
                    raise
                time.sleep(min(0.02 * (2 ** attempt), 0.2))
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class DataRelease:
    root: Path
    release_id: str
    providers: dict[str, str]


_scope: ContextVar[DataRelease | None] = ContextVar("local_data_release", default=None)


def current_release() -> DataRelease | None:
    return _scope.get()


@contextmanager
def data_scope(release: DataRelease | None):
    token = _scope.set(release)
    try:
        yield
    finally:
        _scope.reset(token)


class PipelineBusy(RuntimeError):
    pass


class PipelineLock:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.handle.seek(0, 2) == 0:
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise PipelineBusy("Another data update or index build is running") from exc

    def close(self):
        if not self.handle.closed:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def copy_catalog(source: Path, destination: Path, *, include_cache: bool = True) -> None:
    """Copy catalog files only, never indexes, conversation data or job state."""
    for object_type in OBJECT_TYPES:
        for pattern in ("*/raw.jsonl", "*/translated/*.jsonl"):
            for path in (source / object_type).glob(pattern):
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    support_files = ["series_manifest.json"]
    if include_cache:
        support_files.append(".translation_cache.json")
    for name in support_files:
        path = source / name
        if path.exists():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / name)


def copy_translation_cache(source: Path, destination: Path) -> None:
    path = source / ".translation_cache.json"
    if path.exists():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination / path.name)


def _replace_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        for attempt in range(8):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.02 * (2 ** attempt), 0.2))
    finally:
        temporary.unlink(missing_ok=True)


def sync_catalog(source: Path, destination: Path) -> None:
    """Mirror a validated release to the canonical data paths, file-atomically."""
    for object_type in OBJECT_TYPES:
        for pattern in ("*/raw.jsonl", "*/translated/*.jsonl"):
            for path in (source / object_type).glob(pattern):
                _replace_file(path, destination / path.relative_to(source))
    for name in ("series_manifest.json", ".translation_cache.json"):
        path = source / name
        if path.exists():
            _replace_file(path, destination / name)


class ArtifactLease:
    """Cross-process shared/exclusive byte lock for a release or index."""
    def __init__(self, path: Path, *, shared: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        self.shared = shared
        try:
            if os.name == "nt":
                import ctypes
                import msvcrt
                from ctypes import wintypes

                class Overlapped(ctypes.Structure):
                    _fields_ = [
                        ("Internal", ctypes.c_void_p),
                        ("InternalHigh", ctypes.c_void_p),
                        ("Offset", wintypes.DWORD),
                        ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", wintypes.HANDLE),
                    ]

                if self.handle.seek(0, 2) == 0:
                    self.handle.write(b"0")
                    self.handle.flush()
                self._overlapped = Overlapped()
                self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                flags = 0x00000001 | (0 if shared else 0x00000002)
                locked = self._kernel32.LockFileEx(
                    msvcrt.get_osfhandle(self.handle.fileno()), flags, 0, 1, 0,
                    ctypes.byref(self._overlapped),
                )
                if not locked:
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                import fcntl
                mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(self.handle, mode | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise PipelineBusy("Artifact is still in use") from exc

    def close(self):
        if self.handle.closed:
            return
        if os.name == "nt":
            import ctypes
            import msvcrt
            unlocked = self._kernel32.UnlockFileEx(
                msvcrt.get_osfhandle(self.handle.fileno()), 0, 1, 0,
                ctypes.byref(self._overlapped),
            )
            if not unlocked:
                error = ctypes.WinError(ctypes.get_last_error())
                self.handle.close()
                raise error
        else:
            import fcntl
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LocalDataStore:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir).resolve()
        self.root = self.data_dir / ".pipeline"

    def lock(self) -> PipelineLock:
        return PipelineLock(self.root / "writer.lock")

    def index_lock(self) -> PipelineLock:
        return PipelineLock(self.root / "index.lock")

    def artifact_lease(self, kind: str, artifact_id: str, *, shared: bool = True) -> ArtifactLease:
        if kind not in {"release", "index"}:
            raise ValueError("Unknown artifact kind")
        if len(artifact_id) != 32 or any(c not in "0123456789abcdef" for c in artifact_id):
            raise ValueError("Invalid artifact ID")
        return ArtifactLease(self.root / "leases" / f"{kind}-{artifact_id}.lock", shared=shared)

    @contextmanager
    def wiki_snapshot(self):
        """Lease the selected wiki release so pruning cannot remove it mid-request."""
        for _ in range(3):
            release = self.wiki_release()
            if release is None:
                yield None
                return
            try:
                lease = self.artifact_lease("release", release.release_id)
            except PipelineBusy:
                continue
            with lease:
                if (release.root / "release.json").exists():
                    yield release
                    return
        raise PipelineBusy("Wiki release changed while opening the request snapshot")

    def pointers(self) -> dict:
        return read_json(self.root / "active.json")

    def release(self, release_id: str | None) -> DataRelease | None:
        if not release_id:
            return None
        if len(release_id) != 32 or any(c not in "0123456789abcdef" for c in release_id):
            raise ValueError("Invalid local release ID")
        root = self.root / "releases" / release_id
        metadata = read_json(root / "release.json")
        if metadata.get("release_id") != release_id:
            raise ValueError("Local data release is missing or incomplete")
        return DataRelease(root, release_id, metadata["providers"])

    def wiki_release(self) -> DataRelease | None:
        return self.release(self.pointers().get("wiki"))

    def ensure_wiki_release(self) -> DataRelease:
        """Bootstrap immutable data for an explicit index build; caller holds lock."""
        existing = self.wiki_release()
        if existing:
            return existing
        run_id = uuid4().hex
        stage = self.root / "jobs" / "bootstrap" / run_id
        copy_catalog(self.data_dir, stage)
        try:
            return self.publish(stage, {kind: normalize_translation_provider() for kind in OBJECT_TYPES}, run_id)
        finally:
            target = stage.resolve()
            if target.parent == (self.root / "jobs" / "bootstrap").resolve() and target.exists():
                shutil.rmtree(target)

    def chat_release(self, manifest: dict | None = None) -> DataRelease | None:
        if manifest is None and (self.root / "chat.json").exists():
            return self.chat_target()[0]
        if manifest is None:
            manifest = read_json(self.data_dir / "rag_index" / "manifest.json")
        return self.release(manifest.get("local_release_id") or self.pointers().get("baseline"))

    def chat_target(self) -> tuple[DataRelease | None, Path]:
        active = read_json(self.root / "chat.json")
        if active:
            index_id = str(active.get("index_id", ""))
            if len(index_id) != 32 or any(c not in "0123456789abcdef" for c in index_id):
                raise ValueError("Invalid local index ID")
            return self.release(active["release_id"]), self.root / "indexes" / index_id
        index = self.data_dir / "rag_index"
        manifest = read_json(index / "manifest.json")
        return self.release(manifest.get("local_release_id") or self.pointers().get("baseline")), index

    def activate_chat(self, release_id: str, index_id: str) -> None:
        pointer = self.pointers()
        if pointer.pop("baseline", None) is not None:
            write_json(self.root / "active.json", pointer)
        write_json(self.root / "chat.json", {"release_id": release_id, "index_id": index_id})

    def status(self) -> dict:
        value = read_json(self.root / "status.json") or {
            "state": "idle", "message": "Ready", "crawl_progress": {},
        }
        pointer = self.pointers()
        value["data_revision"] = pointer.get("wiki", "legacy")
        chat_release, _ = self.chat_target()
        value["pending_index"] = bool(pointer.get("wiki") and
                                      pointer["wiki"] != (chat_release.release_id if chat_release else None))
        value["chat_data_revision"] = chat_release.release_id if chat_release else "legacy"
        if value.get("state") in {"starting", "updating", "translating", "validating", "publishing", "indexing"}:
            try:
                with self.lock():
                    value.update(state="interrupted", message="The update was interrupted. Retry to resume.")
            except PipelineBusy:
                pass
        return value

    def set_status(self, **values) -> dict:
        status = read_json(self.root / "status.json")
        status.update(values)
        status["updated_at"] = time.time()
        write_json(self.root / "status.json", status)
        return status

    def publish(self, staging: Path, providers: dict[str, str], run_id: str) -> DataRelease:
        pointer = self.pointers()
        if not pointer.get("baseline") and not (self.root / "chat.json").exists():
            baseline_id = uuid4().hex
            baseline = self.root / "releases" / baseline_id
            copy_catalog(self.data_dir, baseline, include_cache=False)
            write_json(baseline / "release.json", {
                "release_id": baseline_id,
                "providers": {kind: normalize_translation_provider() for kind in OBJECT_TYPES},
                "created_at": time.time(),
            })
            pointer["baseline"] = baseline_id
        release_id = run_id
        target = self.root / "releases" / release_id
        # Keep the resumable stage until the pointer is durable. Reusing run_id
        # makes a retry after a failed pointer write idempotent.
        copy_catalog(staging, target, include_cache=False)
        write_json(target / "release.json", {
            "release_id": release_id, "providers": providers,
            "created_at": time.time(), "run_id": run_id,
        })
        # Canonical paths stay current for direct scripts and cloud packaging.
        # The serving pointer changes only after this mirror is complete.
        sync_catalog(staging, self.data_dir)
        pointer.update(wiki=release_id, run_id=run_id)
        write_json(self.root / "active.json", pointer)
        return self.release(release_id)

    def prune_artifacts(self) -> None:
        """Remove obsolete pairs unless another process still leases them."""
        pointer = self.pointers()
        chat = read_json(self.root / "chat.json")
        protected_releases = {value for value in (pointer.get("wiki"), chat.get("release_id")) if value}
        protected_index = chat.get("index_id")

        indexes = self.root / "indexes"
        for path in list(indexes.iterdir()) if indexes.exists() else []:
            if not path.is_dir() or path.name == protected_index:
                continue
            manifest = read_json(path / "manifest.json")
            release_id = manifest.get("local_release_id")
            leases = []
            try:
                leases.append(self.artifact_lease("index", path.name, shared=False))
                if release_id:
                    leases.append(self.artifact_lease("release", release_id, shared=False))
                shutil.rmtree(path)
            except (PipelineBusy, PermissionError, OSError):
                pass
            finally:
                for lease in reversed(leases):
                    lease.close()

        referenced = set(protected_releases)
        if indexes.exists():
            for path in indexes.iterdir():
                if path.is_dir():
                    release_id = read_json(path / "manifest.json").get("local_release_id")
                    if release_id:
                        referenced.add(release_id)
        releases = self.root / "releases"
        for path in list(releases.iterdir()) if releases.exists() else []:
            if not path.is_dir() or path.name in referenced:
                continue
            try:
                with self.artifact_lease("release", path.name, shared=False):
                    shutil.rmtree(path)
            except (PipelineBusy, PermissionError, OSError):
                pass

        lease_dir = self.root / "leases"
        if lease_dir.exists():
            roots = {"release": self.root / "releases", "index": self.root / "indexes"}
            for path in lease_dir.glob("*.lock"):
                kind, separator, artifact_id = path.stem.partition("-")
                artifact_root = roots.get(kind)
                if separator and artifact_root and not (artifact_root / artifact_id).exists():
                    try:
                        path.unlink()
                    except (PermissionError, OSError):
                        pass

    def prune_history(self, keep: int | None = None) -> None:
        """Bound durable run summaries; checkpoints and the latest status are separate."""
        limit = max(1, keep if keep is not None else int(os.getenv("KAMI_LOCAL_RUN_HISTORY", "100")))
        runs = self.root / "runs"
        if not runs.exists():
            return
        history = sorted(
            runs.glob("*.json"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        for path in history[limit:]:
            path.unlink(missing_ok=True)

    def compact_storage(self) -> None:
        """Migrate the active snapshot to canonical paths and prune old artifacts."""
        release = self.wiki_release()
        if release:
            sync_catalog(release.root, self.data_dir)
            (release.root / ".translation_cache.json").unlink(missing_ok=True)
        if read_json(self.root / "chat.json"):
            pointer = self.pointers()
            if pointer.pop("baseline", None) is not None:
                write_json(self.root / "active.json", pointer)
        self.prune_artifacts()
