"""Download the INCLUDE videos needed by a config, plus the official splits.

INCLUDE (Sridhar et al., 2020; CC BY 4.0) is ~57 GB of category zips on
Zenodo. Instead of downloading whole zips, this reads each zip's central
directory with HTTP range requests, then fetches only the videos of the signs
in the config: one request per video, several in parallel.

    python scripts/download_include.py --config configs/isl24.yaml
    python scripts/download_include.py --config configs/isl24.yaml --list   # dry run
"""

import argparse
import io
import shutil
import struct
import threading
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import load_config  # noqa: E402

ZENODO_RECORD = "https://zenodo.org/api/records/4010759"
SPLITS_URL = "https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/{name}.txt"
SPLIT_FILES = [f"{name}_{split}" for name in ("include", "include50") for split in ("train", "val", "test")]


_pause_until = 0.0  # shared by all download threads: when Zenodo rate-limits one, all back off
_pause_lock = threading.Lock()


def get(client: httpx.Client, url: str, retries: int = 8, **kwargs) -> httpx.Response:
    """GET with backoff on network errors, 5xx and 429 rate limits (Zenodo is flaky and rate-limited)."""
    global _pause_until
    for attempt in range(retries):
        time.sleep(max(0.0, _pause_until - time.time()))
        try:
            r = client.get(url, **kwargs)
            if r.status_code < 500 and r.status_code != 429:
                return r.raise_for_status()
            error = httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
        except httpx.TransportError as e:
            r, error = None, e
        if attempt == retries - 1:
            break
        if r is not None and r.status_code == 429:
            try:
                wait = float(r.headers.get("Retry-After", 60))
            except ValueError:
                wait = 60.0
            with _pause_lock:
                _pause_until = max(_pause_until, time.time() + wait)
            print(f"  rate-limited by Zenodo; all downloads pausing {wait:.0f}s")
        else:
            wait = min(2 ** attempt, 60)
            print(f"  {url}: {error}; retrying in {wait}s")
            time.sleep(wait)
    raise error


class HttpRangeFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP Range requests."""

    def __init__(self, client: httpx.Client, url: str, size: int, block: int = 1 << 20):
        self.client, self.url, self.size, self.block = client, url, size, block
        self.pos = 0
        self._cache_start, self._cache = -1, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _fetch(self, start, end):
        r = get(self.client, self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        if r.status_code != 206:
            raise IOError(f"server ignored range request (HTTP {r.status_code}); use --full")
        return r.content

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        cache_end = self._cache_start + len(self._cache)
        if self._cache_start <= self.pos and self.pos + n <= cache_end:
            out = self._cache[self.pos - self._cache_start:self.pos - self._cache_start + n]
        elif n >= self.block:
            out = self._fetch(self.pos, self.pos + n)
        else:
            end = min(self.size, self.pos + max(n, self.block))
            self._cache_start, self._cache = self.pos, self._fetch(self.pos, end)
            out = self._cache[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def normalize(path: str) -> str:
    """Case-, slash- and underscore-insensitive path key ("Days_and_Time" == "Days and Time")."""
    parts = path.replace("\\", "/").replace("_", " ").split("/")
    return "/".join(" ".join(p.lower().split()) for p in parts if p.strip())


def zenodo_files(client) -> dict:
    record = get(client, ZENODO_RECORD).json()
    return {f["key"]: (f["links"]["self"], f["size"]) for f in record["files"]}


def wanted_member(name: str, folders: dict):
    """Return (config folder key, path inside the sign folder) for a zip member, or None.

    The inner path keeps sub-folders such as "Extra/", which the split lists reference.
    """
    if name.endswith("/"):
        return None
    norm = normalize(name)
    for folder_norm in folders:
        marker = f"/{folder_norm}/"
        if marker in f"/{norm}":
            depth = folder_norm.count("/") + 1
            raw = [p for p in name.replace("\\", "/").split("/") if p.strip()]
            start = next(i for i in range(len(raw)) if normalize("/".join(raw[i:i + depth])) == folder_norm)
            return folder_norm, "/".join(raw[start + depth:])
    return None


def fetch_member(client, url: str, info: zipfile.ZipInfo) -> bytes:
    """Download one zip member with a single range request and return its contents."""
    start = info.header_offset
    end = start + 30 + len(info.filename.encode()) + 1024 + info.compress_size  # 1 KB slack for extra fields
    data = get(client, url, headers={"Range": f"bytes={start}-{end - 1}"}).content
    if data[:4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", data[26:30])
    offset = 30 + name_len + extra_len
    if len(data) < offset + info.compress_size:  # unusually large extra field
        data += get(client, url, headers={"Range": f"bytes={start + len(data)}-{start + offset + info.compress_size - 1}"}).content
    payload = data[offset:offset + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        content = payload
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(payload, -15)
    else:
        raise IOError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(content) != info.CRC:
        raise IOError(f"checksum mismatch for {info.filename}")
    return content


def save_member(client, url, info, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(fetch_member(client, url, info))
    tmp.replace(dest)  # only complete, checksum-verified files get the final name


def download_splits(client, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_FILES:
        path = out_dir / f"{name}.txt"
        if not path.exists():
            path.write_text(get(client, SPLITS_URL.format(name=name)).text, encoding="utf-8")
    print(f"Official splits -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="override data.include_root")
    ap.add_argument("--list", action="store_true", help="only list matching videos")
    ap.add_argument("--full", action="store_true", help="download whole zips (if range requests fail)")
    ap.add_argument("--workers", type=int, default=4, help="parallel video downloads (Zenodo rate-limits more)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.out) if args.out else cfg.include_root
    folders = {normalize(s.include): s.include for s in cfg.signs}
    categories = sorted({s.include.split("/")[0] for s in cfg.signs})

    with httpx.Client(follow_redirects=True, timeout=120) as client:
        download_splits(client, root / "splits")
        files = zenodo_files(client)
        wanted = {normalize(c) for c in categories}
        zips = [k for k in files if k.endswith(".zip") and normalize(k.rsplit("_", 1)[0]) in wanted]
        print(f"{len(cfg.signs)} signs in {len(categories)} categories -> {len(zips)} zip(s) to scan")

        found = {f: 0 for f in folders}
        tasks = []  # (url, ZipInfo, dest) for videos not yet on disk
        for key in sorted(zips):
            url, size = files[key]
            if args.full:
                local = root / "_zips" / key
                local.parent.mkdir(parents=True, exist_ok=True)
                if not local.exists():
                    print(f"Downloading {key} ({size / 1e9:.2f} GB) ...")
                    with client.stream("GET", url) as r, open(local.with_suffix(".part"), "wb") as f:
                        for chunk in r.iter_bytes(1 << 20):
                            f.write(chunk)
                    local.with_suffix(".part").replace(local)
                source = open(local, "rb")
            else:
                source = io.BufferedReader(HttpRangeFile(client, url, size), buffer_size=1 << 16)

            with source, zipfile.ZipFile(source) as zf:
                members = [(m, wanted_member(m.filename, folders)) for m in zf.infolist()]
                members = [(m, w) for m, w in members if w]
                todo = 0
                for member, (folder, inner) in members:
                    found[folder] += 1
                    dest = root / folders[folder] / inner
                    if args.list or (dest.exists() and dest.stat().st_size == member.file_size):
                        continue
                    todo += 1
                    if args.full:  # zip is local: extract directly
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    else:
                        tasks.append((url, member, dest))
                print(f"{key}: {len(members)} matching file(s), {todo} to download")

        if tasks:
            total = sum(m.compress_size for _, m, _ in tasks)
            print(f"\nDownloading {len(tasks)} videos ({total / 1e9:.1f} GB) with {args.workers} parallel requests ...")
            done, done_bytes, failed, start = 0, 0, [], time.time()
            with ThreadPoolExecutor(args.workers) as pool:
                futures = {pool.submit(save_member, client, url, m, dest): (m, dest) for url, m, dest in tasks}
                for fut in as_completed(futures):
                    member, dest = futures[fut]
                    try:
                        fut.result()
                        done += 1
                        done_bytes += member.compress_size
                    except Exception as e:
                        failed.append(dest)
                        print(f"  failed: {dest.relative_to(root)}: {e}")
                    rate = done_bytes / max(time.time() - start, 1e-6)
                    eta = (total - done_bytes) / rate / 60 if rate else 0
                    print(f"  [{done}/{len(tasks)}] {dest.relative_to(root).as_posix()}  "
                          f"{rate / 1e6:.1f} MB/s, ~{eta:.0f} min left", flush=True)
            if failed:
                print(f"\n{len(failed)} video(s) failed; run the same command again to retry them.")
                sys.exit(1)

    missing = [folders[f] for f, n in found.items() if n == 0]
    print(f"\n{sum(found.values())} videos across {len(found) - len(missing)} signs -> {root}")
    if missing:
        print("WARNING: no videos found for: " + ", ".join(missing))
        sys.exit(1)


if __name__ == "__main__":
    main()
