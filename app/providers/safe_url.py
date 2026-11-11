"""Checks that an address typed in the admin panel points at a public server.

Base addresses for OpenAI-compatible providers and Hugging Face endpoints come from the admin panel. Without this check
they could be used to make the service call internal systems (server-side request forgery).
"""

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "metadata", "instance-data"}


class UnsafeUrlError(ValueError):
    pass


def _public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


def check_shape(url: str) -> tuple[str, int]:
    """Checks what can be checked without the network. Returns the host and port."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port or 443
    except ValueError as error:
        raise UnsafeUrlError("The address is not valid") from error
    if parts.scheme != "https":
        raise UnsafeUrlError("The address must start with https://")
    if parts.username or parts.password:
        raise UnsafeUrlError("The address must not contain a user name or password")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host or host in _BLOCKED_HOSTS or host.endswith((".localhost", ".local", ".internal")):
        raise UnsafeUrlError("The address must be a public server")
    if parts.query or parts.fragment:
        raise UnsafeUrlError("The address must not contain ? or #")
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None and not _public(str(literal)):
        raise UnsafeUrlError("The address must be a public server")
    return host, port


async def ensure_public_url(url: str) -> str:
    """Resolves the host and refuses the address if any of its IP addresses is not public."""
    host, port = check_shape(url)
    try:
        infos = await asyncio.wait_for(asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM), timeout=5)
    except (OSError, TimeoutError) as error:
        raise UnsafeUrlError("The address could not be found") from error
    addresses = {info[4][0] for info in infos}
    if not addresses or not all(_public(address) for address in addresses):
        raise UnsafeUrlError("The address must be a public server")
    return url.strip().rstrip("/")
