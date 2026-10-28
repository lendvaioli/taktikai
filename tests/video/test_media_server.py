import http.client

import pytest

from match_tagger.video.media_server import MediaServer, parse_range

DATA = bytes(range(10))  # 10 bytes: 0..9


@pytest.fixture()
def server(tmp_path):
    root = tmp_path / "raw"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "clip.mp4").write_bytes(DATA)
    (tmp_path / "secret.txt").write_text("nope")
    srv = MediaServer({"raw": root}).start()
    yield srv, root, tmp_path
    srv.stop()


def _request(srv, method, path, headers=None):
    host, port = srv.httpd.server_address
    conn = http.client.HTTPConnection(host, port, timeout=5)
    conn.request(method, path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        ("bytes=2-5", 10, (2, 5)),
        ("bytes=7-", 10, (7, 9)),
        ("bytes=-3", 10, (7, 9)),
        ("bytes=8-100", 10, (8, 9)),
        ("bytes=10-12", 10, None),
        ("bytes=5-2", 10, None),
        ("bytes=-", 10, None),
        ("items=0-1", 10, None),
    ],
)
def test_parse_range(header, size, expected):
    assert parse_range(header, size) == expected


def test_binds_to_localhost_only(server):
    srv, _, _ = server
    assert srv.httpd.server_address[0] == "127.0.0.1"
    assert srv.base_url.startswith("http://127.0.0.1:")


def test_full_get(server):
    srv, _, _ = server
    resp, body = _request(srv, "GET", "/raw/sub/clip.mp4")
    assert (resp.status, body) == (200, DATA)
    assert resp.getheader("Content-Type") == "video/mp4"
    assert resp.getheader("Accept-Ranges") == "bytes"


def test_range_get_returns_206_and_the_exact_bytes(server):
    srv, _, _ = server
    resp, body = _request(srv, "GET", "/raw/sub/clip.mp4", {"Range": "bytes=2-5"})
    assert (resp.status, body) == (206, DATA[2:6])
    assert resp.getheader("Content-Range") == "bytes 2-5/10"
    assert resp.getheader("Content-Length") == "4"


def test_head_has_headers_but_no_body(server):
    srv, _, _ = server
    resp, body = _request(srv, "HEAD", "/raw/sub/clip.mp4")
    assert (resp.status, body, resp.getheader("Content-Length")) == (200, b"", "10")


def test_unsatisfiable_range(server):
    srv, _, _ = server
    resp, _ = _request(srv, "GET", "/raw/sub/clip.mp4", {"Range": "bytes=20-30"})
    assert resp.status == 416
    assert resp.getheader("Content-Range") == "bytes */10"


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("/raw/../secret.txt", 400),
        ("/raw/sub/%2e%2e/%2e%2e/secret.txt", 400),
        ("/other/secret.txt", 404),
        ("/raw/sub", 404),  # directories are never listed
        ("/raw/missing.mp4", 404),
    ],
)
def test_paths_outside_the_roots_are_refused(server, path, status):
    srv, _, _ = server
    resp, _ = _request(srv, "GET", path)
    assert resp.status == status


def test_url_for(server):
    srv, root, tmp_path = server
    assert srv.url_for(root / "sub" / "clip.mp4") == f"{srv.base_url}/raw/sub/clip.mp4"
    with pytest.raises(ValueError):
        srv.url_for(tmp_path / "secret.txt")
