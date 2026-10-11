#!/usr/bin/env python3
"""
Minimal-overhead client for oet_server.

Functionally equivalent to oet.server_client.client, but speaks HTTP/1.1 over a
raw socket from the standard library instead of going through `requests`.
That matters because ORCA starts the client once per gradient: importing
`requests` pulls in urllib3, ssl, http.client and the email/charset machinery,
which costs ~50-90 ms per call -- more than the calculation it is asking for.
Only `socket` and `json` are imported here, which is ~3 ms.

The server is unchanged: this talks to the same Flask/waitress /calculate
endpoint. For the lowest possible per-call cost use the compiled client
(oet_client.c) instead, which avoids the Python interpreter start as well.

Provides
--------
function: send_to_server
    Post one calculation to the server and relay its result
function: main
    Entry point for CLI execution
"""

from __future__ import annotations

import json
import os
import socket
import sys

# Matches oet.server_client.client
DEFAULT_HOST_PORT = "127.0.0.1:8888"

# Read size per recv call
CHUNK = 65536


def _read_all(sock: socket.socket) -> bytes:
    """
    Read from the socket until the peer closes it.

    Parameters
    ----------
    sock: socket.socket
        Connected socket

    Returns
    -------
    bytes: Everything the server sent
    """
    chunks: list[bytes] = []
    while True:
        data = sock.recv(CHUNK)
        if not data:
            return b"".join(chunks)
        chunks.append(data)


def _dechunk(body: bytes) -> bytes:
    """
    Decode a chunked transfer-encoded body.

    Parameters
    ----------
    body: bytes
        Body as received

    Returns
    -------
    bytes: The decoded body
    """
    out: list[bytes] = []
    while True:
        line, _, rest = body.partition(b"\r\n")
        if not line:
            break
        size = int(line.split(b";")[0], 16)
        if size == 0:
            break
        out.append(rest[:size])
        body = rest[size + 2 :]
    return b"".join(out)


def send_to_server(host_port: str, arguments: list[str]) -> None:
    """
    Send a calculation to the server. The server writes the result files; this
    only relays the server's STDOUT and its exit status.

    Parameters
    ----------
    host_port: str
        Host:Port of the server
    arguments: list[str]
        Arguments to forward to the server's calculator

    Raises
    ------
    SystemExit: On any connection or server-side error
    """
    host, _, port = host_port.rpartition(":")
    payload = json.dumps({"arguments": arguments, "directory": os.getcwd()}).encode()
    request = (
        b"POST /calculate HTTP/1.1\r\n"
        b"Host: " + host_port.encode() + b"\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
        b"Connection: close\r\n\r\n" + payload
    )

    try:
        with socket.create_connection((host, int(port))) as sock:
            sock.sendall(request)
            raw = _read_all(sock)
    except OSError as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    head, sep, body = raw.partition(b"\r\n\r\n")
    if not sep:
        print(f"Malformed response from server: {raw!r}")
        sys.exit(1)
    if b"transfer-encoding: chunked" in head.lower():
        body = _dechunk(body)

    try:
        data = json.loads(body)
    except ValueError:
        print(f"Could not parse server response: {body!r}")
        sys.exit(1)

    # Print the stdout of the calculation, if any
    print(data.get("stdout", ""), end="")
    if data.get("status") == "Error":
        print(f"Server error {data.get('error_type')}: {data.get('error_message')}.")
        print(f"Exact traceback: {data.get('traceback')}")
        sys.exit(1)


def main() -> None:
    """
    Entry point for CLI execution.

    Arguments are not parsed with argparse: building a parser costs more than the
    whole request. Only -b/--bind is recognised, everything else is forwarded to
    the server verbatim.
    """
    argv = sys.argv[1:]
    host_port = DEFAULT_HOST_PORT
    forwarded: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-b", "--bind") and i + 1 < len(argv):
            host_port = argv[i + 1]
            i += 2
        else:
            forwarded.append(argv[i])
            i += 1

    send_to_server(host_port=host_port, arguments=forwarded)


if __name__ == "__main__":
    main()
