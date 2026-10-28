"""Local read-only media server with HTTP range support, for the Streamlit apps.

Streamlit cannot serve long recordings: its static files are limited to 200 MB and its
component files are read into memory without range support. This server streams video files
from a few allowed folders straight from disk, so browsers can seek in a 90-minute recording.
It binds to 127.0.0.1 only, answers GET and HEAD, and never lists directories.
"""

import mimetypes
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

HOST = "127.0.0.1"
CHUNK = 1 << 20
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse a single ``bytes=a-b`` / ``a-`` / ``-n`` range into inclusive (start, end).

    Returns None for an unsatisfiable or malformed range.
    """
    m = _RANGE.match(header.strip())
    if not m or (not m.group(1) and not m.group(2)) or size == 0:
        return None
    if not m.group(1):  # suffix: the last n bytes
        n = int(m.group(2))
        return (max(size - n, 0), size - 1) if n > 0 else None
    start = int(m.group(1))
    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    return (start, end) if start <= end and start < size else None


def resolve(roots: dict[str, Path], url_path: str) -> Path | None:
    """Map ``/<root>/<relative path>`` to a file inside that root; None if not allowed.

    Raises ValueError for paths containing ``..`` segments.
    """
    parts = unquote(urlsplit(url_path).path).lstrip("/").split("/")
    if ".." in parts:
        raise ValueError("parent segments are not allowed")
    if len(parts) < 2 or parts[0] not in roots:
        return None
    root = roots[parts[0]].resolve()
    target = root.joinpath(*parts[1:]).resolve()
    return target if target.is_relative_to(root) and target.is_file() else None


class _MediaHandler(BaseHTTPRequestHandler):
    """GET/HEAD with single-range support for files below ``roots``."""

    roots: dict[str, Path] = {}

    def do_HEAD(self) -> None:  # noqa: N802 (http.server API)
        self._serve(body=False)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        self._serve(body=True)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence per-request logging."""

    def _serve(self, body: bool) -> None:
        try:
            path = resolve(self.roots, self.path)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST)
            return
        if path is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        header = self.headers.get("Range")
        span = (0, size - 1) if header is None else parse_range(header, size)
        if span is None:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self._send(path, size, span, partial=header is not None, body=body)

    def _send(
        self, path: Path, size: int, span: tuple[int, int], partial: bool, body: bool
    ) -> None:
        start, end = span
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_header("Content-Type", mime)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1 if size else 0))
        self.send_header("Access-Control-Allow-Origin", "*")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if body and size:
            with path.open("rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)


def _handler(roots: dict[str, Path]) -> type[BaseHTTPRequestHandler]:
    """A handler class bound to the given roots."""
    return type("Handler", (_MediaHandler,), {"roots": roots})


class MediaServer:
    """Serve files below named roots at ``http://127.0.0.1:<port>/<root>/<path>``."""

    def __init__(self, roots: dict[str, Path], port: int = 0) -> None:
        """Create the server (port 0 picks a free port); call ``start`` to serve."""
        self.roots = {name: Path(p) for name, p in roots.items()}
        self.httpd = ThreadingHTTPServer((HOST, port), _handler(self.roots))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """``http://127.0.0.1:<port>``."""
        return f"http://{HOST}:{self.httpd.server_address[1]}"

    def start(self) -> "MediaServer":
        """Start serving in a background thread."""
        self.thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the port."""
        self.httpd.shutdown()
        self.httpd.server_close()

    def url_for(self, path: Path) -> str:
        """URL of a file below one of the roots; raises ValueError for other files."""
        target = Path(path).resolve()
        for name, root in self.roots.items():
            if target.is_relative_to(root.resolve()):
                rel = target.relative_to(root.resolve()).as_posix()
                return f"{self.base_url}/{name}/{quote(rel)}"
        raise ValueError(f"{path} is not below a served root")
