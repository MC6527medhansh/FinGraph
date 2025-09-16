"""Minimal httpx stub required for FastAPI TestClient in offline tests."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Tuple, Union
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

__all__ = [
    "Client",
    "Request",
    "Response",
    "BaseTransport",
    "ByteStream",
    "URL",
    "USE_CLIENT_DEFAULT",
    "_types",
    "_client",
]


class Headers:
    """Simplified case-insensitive headers container."""

    def __init__(self, initial: Optional[Union[Mapping[str, str], Iterable[Tuple[str, str]]]] = None) -> None:
        self._items: list[Tuple[str, str]] = []
        if initial:
            self.update(initial)

    def add(self, key: str, value: Union[str, int]) -> None:
        self._items.append((str(key), str(value)))

    def update(self, values: Union["Headers", Mapping[str, str], Iterable[Tuple[str, str]]]) -> None:
        if isinstance(values, Headers):
            self._items.extend(values.multi_items())
            return
        if isinstance(values, Mapping):
            for key, value in values.items():
                self.add(key, value)
        else:
            for key, value in values:
                self.add(key, value)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        lowered = key.lower()
        for existing_key, value in reversed(self._items):
            if existing_key.lower() == lowered:
                return value
        return default

    def multi_items(self) -> list[Tuple[str, str]]:
        return list(self._items)

    def copy(self) -> "Headers":
        clone = Headers()
        clone._items = self._items.copy()
        return clone

    def __contains__(self, key: str) -> bool:
        lowered = key.lower()
        return any(existing_key.lower() == lowered for existing_key, _ in self._items)


class URL:
    """Minimal URL representation supporting attributes used by Starlette."""

    def __init__(self, value: Union[str, "URL"]) -> None:
        if isinstance(value, URL):
            self._parsed = value._parsed
        else:
            if not value:
                value = ""
            self._parsed = urlparse(value)

    def __str__(self) -> str:
        return urlunparse(self._parsed)

    @property
    def scheme(self) -> str:
        return self._parsed.scheme or "http"

    @property
    def netloc(self) -> bytes:
        return (self._parsed.netloc or "").encode("ascii", "ignore")

    @property
    def path(self) -> str:
        return self._parsed.path or "/"

    @property
    def raw_path(self) -> bytes:
        path = self._parsed.path or "/"
        if self._parsed.query:
            return f"{path}?{self._parsed.query}".encode("ascii", "ignore")
        return path.encode("ascii", "ignore")

    @property
    def query(self) -> bytes:
        return (self._parsed.query or "").encode("ascii", "ignore")


class Request:
    """Simplified request object passed to the transport."""

    def __init__(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Headers] = None,
        content: Optional[Union[str, bytes]] = None,
        data: Optional[Union[str, bytes, Mapping[str, Any], Iterable[Tuple[str, Any]]]] = None,
        json_data: Any = None,
    ) -> None:
        self.method = method.upper()
        self.url = URL(url)
        self.headers = headers.copy() if headers else Headers()
        self._body = self._encode_body(content=content, data=data, json_data=json_data)

    @staticmethod
    def _encode_body(
        *,
        content: Optional[Union[str, bytes]] = None,
        data: Optional[Union[str, bytes, Mapping[str, Any], Iterable[Tuple[str, Any]]]] = None,
        json_data: Any = None,
    ) -> bytes:
        if json_data is not None:
            return json.dumps(json_data).encode("utf-8")
        if content is not None:
            return content.encode("utf-8") if isinstance(content, str) else content
        if data is None:
            return b""
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode("utf-8")
        if isinstance(data, Mapping):
            return urlencode(list(data.items())).encode("utf-8")
        if isinstance(data, Iterable):
            return urlencode(list(data)).encode("utf-8")
        return b""

    def read(self) -> bytes:
        return self._body


class ByteStream:
    """Wrapper around bytes used by the response."""

    def __init__(self, data: Union[bytes, bytearray]) -> None:
        self._buffer = io.BytesIO(bytes(data))

    def read(self) -> bytes:
        return self._buffer.read()

    def __iter__(self) -> Iterator[bytes]:
        chunk = self._buffer.read()
        if chunk:
            yield chunk


class Response:
    """Minimal response implementation supporting JSON access."""

    def __init__(
        self,
        status_code: int = 200,
        headers: Optional[Iterable[Tuple[str, str]]] = None,
        stream: Optional[ByteStream] = None,
        request: Optional[Request] = None,
    ) -> None:
        self.status_code = status_code
        self.headers = Headers(headers or [])
        self._stream = stream or ByteStream(b"")
        self.request = request
        self._content: Optional[bytes] = None
        self.encoding = "utf-8"

    def read(self) -> bytes:
        if self._content is None:
            self._content = self._stream.read()
        return self._content

    @property
    def content(self) -> bytes:
        return self.read()

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding, "replace")

    def json(self) -> Any:
        text = self.text
        if not text:
            return None
        return json.loads(text)

    def close(self) -> None:
        self._stream = ByteStream(b"")


class BaseTransport:
    """Base class for transports used by the TestClient."""

    def handle_request(self, request: Request) -> Response:  # pragma: no cover - interface definition
        raise NotImplementedError


def _encode_params(params: Any) -> str:
    if params is None:
        return ""
    if isinstance(params, str):
        return params
    if isinstance(params, Mapping):
        return urlencode(list(params.items()), doseq=True)
    if isinstance(params, Iterable):
        return urlencode(list(params), doseq=True)
    return str(params)


class Client:
    """Very small subset of httpx.Client sufficient for Starlette's TestClient."""

    def __init__(
        self,
        *,
        base_url: Union[str, URL] = "http://testserver",
        headers: Optional[Union[Mapping[str, str], Iterable[Tuple[str, str]]]] = None,
        transport: Optional[BaseTransport] = None,
        follow_redirects: bool = True,
        cookies: Optional[MutableMapping[str, str]] = None,
    ) -> None:
        self.base_url = URL(base_url)
        self.transport = transport
        self.follow_redirects = follow_redirects
        self.cookies = cookies or {}
        self._headers = Headers(headers or {})
        self._is_closed = False

    def _merge_url(self, url: Union[str, URL]) -> str:
        if isinstance(url, URL):
            url_str = str(url)
        else:
            url_str = str(url)
        if url_str.startswith("http://") or url_str.startswith("https://"):
            return url_str
        base = str(self.base_url)
        if not base.endswith("/"):
            base = base + "/"
        return urljoin(base, url_str)

    def request(
        self,
        method: str,
        url: Union[str, URL],
        *,
        content: Optional[Union[str, bytes]] = None,
        data: Optional[Union[str, bytes, Mapping[str, Any], Iterable[Tuple[str, Any]]]] = None,
        files: Any = None,
        json: Any = None,
        params: Optional[Union[str, Mapping[str, Any], Iterable[Tuple[str, Any]]]] = None,
        headers: Optional[Union[Mapping[str, str], Iterable[Tuple[str, str]]]] = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: Any = None,
        timeout: Any = None,
        extensions: Optional[Dict[str, Any]] = None,
    ) -> Response:
        if self.transport is None:
            raise RuntimeError("A transport must be supplied to the client.")
        target = self._merge_url(url)
        if params:
            query_string = _encode_params(params)
            if query_string:
                separator = "&" if "?" in target else "?"
                target = f"{target}{separator}{query_string}"
        combined_headers = self._headers.copy()
        if headers:
            combined_headers.update(headers)
        request = Request(
            method,
            target,
            headers=combined_headers,
            content=content,
            data=data,
            json_data=json,
        )
        response = self.transport.handle_request(request)
        return response

    def get(self, url: Union[str, URL], **kwargs: Any) -> Response:
        return self.request("GET", url, **kwargs)

    def options(self, url: Union[str, URL], **kwargs: Any) -> Response:
        return self.request("OPTIONS", url, **kwargs)

    def head(self, url: Union[str, URL], **kwargs: Any) -> Response:
        return self.request("HEAD", url, **kwargs)

    def post(self, url: Union[str, URL], **kwargs: Any) -> Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: Union[str, URL], **kwargs: Any) -> Response:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: Union[str, URL], **kwargs: Any) -> Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: Union[str, URL], **kwargs: Any) -> Response:
        return self.request("DELETE", url, **kwargs)

    def close(self) -> None:
        self._is_closed = True

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class _UseClientDefault:
    def __repr__(self) -> str:
        return "USE_CLIENT_DEFAULT"


USE_CLIENT_DEFAULT = _UseClientDefault()
_client = SimpleNamespace(UseClientDefault=_UseClientDefault, USE_CLIENT_DEFAULT=USE_CLIENT_DEFAULT)
_types = SimpleNamespace(
    URLTypes=Union[str, URL],
    RequestContent=Any,
    RequestFiles=Any,
    QueryParamTypes=Union[str, Mapping[str, Any], Iterable[Tuple[str, Any]]],
    HeaderTypes=Union[Mapping[str, str], Iterable[Tuple[str, str]]],
    CookieTypes=Mapping[str, str],
    AuthTypes=Any,
    TimeoutTypes=Union[float, Tuple[float, float, float, float], None],
)
