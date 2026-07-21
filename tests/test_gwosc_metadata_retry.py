from __future__ import annotations

import urllib.request
from email.message import Message
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import pytest

from gwyolo.gwosc import _remote_size, _urlopen_metadata


class _MetadataResponse:
    def __init__(self, content_length: str = "123") -> None:
        self.headers = Message()
        self.headers["Content-Length"] = content_length

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


def test_public_metadata_retry_is_bounded() -> None:
    request = urllib.request.Request("https://example.test/metadata")
    response = _MetadataResponse()
    with (
        patch(
            "gwyolo.gwosc.urllib.request.urlopen",
            side_effect=[URLError("TLS timeout"), response],
        ) as opener,
        patch("gwyolo.gwosc.time.sleep") as sleep,
    ):
        assert _urlopen_metadata(request, timeout=1, max_attempts=2) is response
    assert opener.call_count == 2
    sleep.assert_called_once_with(0.5)


def test_public_metadata_retry_rejects_permanent_http_error() -> None:
    request = urllib.request.Request("https://example.test/metadata")
    permanent = HTTPError(request.full_url, 404, "not found", {}, None)
    with (
        patch("gwyolo.gwosc.urllib.request.urlopen", side_effect=permanent) as opener,
        pytest.raises(HTTPError),
    ):
        _urlopen_metadata(request, timeout=1, max_attempts=5)
    assert opener.call_count == 1


def test_remote_size_retries_transient_head_failure() -> None:
    response = _MetadataResponse(content_length="456")
    with (
        patch(
            "gwyolo.gwosc.urllib.request.urlopen",
            side_effect=[TimeoutError("handshake"), response],
        ),
        patch("gwyolo.gwosc.time.sleep"),
    ):
        assert _remote_size("https://example.test/file.hdf5") == 456
