#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that a static build deployed to GitHub Pages loads its assets.

Waits until the page at URL serves the same index.html as the local build,
then requests every script and stylesheet the page references and fails if
any of them does not return 200. This catches a wrong build base path, which
otherwise leaves a blank page with no other error.

Usage: check_pages_deploy.py URL LOCAL_INDEX_HTML [TIMEOUT_SECONDS]
"""

from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ASSET_PATTERN = re.compile(r'<(?:script|link)\b[^>]*\b(?:src|href)="([^"]+)"')
POLL_INTERVAL_SECONDS = 15


def fetch(url: str) -> tuple[int | None, str]:
    """Return the status and body for URL, or no status if it is unreachable."""
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, ""
    # Pages can be briefly unreachable during a deployment; the caller retries.
    except urllib.error.URLError:
        return None, ""


def wait_for_deployment(url: str, expected: str, timeout: int) -> None:
    """Poll URL until it serves EXPECTED, or exit after TIMEOUT seconds."""
    deadline = time.monotonic() + timeout
    while True:
        status, body = fetch(url)
        if status == 200 and body.strip() == expected.strip():
            return
        if time.monotonic() >= deadline:
            sys.exit(
                f"{url} did not serve the new build within {timeout}s "
                f"(last status: {status})"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def main() -> None:
    """Check the deployment named on the command line."""
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    url = sys.argv[1]
    with open(sys.argv[2], encoding="utf-8") as index_file:
        expected = index_file.read()
    timeout = int(sys.argv[3]) if len(sys.argv) == 4 else 600

    wait_for_deployment(url, expected, timeout)

    # Third-party assets such as fonts are outside this deployment.
    host = urllib.parse.urlsplit(url).netloc
    assets = [
        urllib.parse.urljoin(url, reference)
        for reference in ASSET_PATTERN.findall(expected)
    ]
    assets = [asset for asset in assets if urllib.parse.urlsplit(asset).netloc == host]
    if not assets:
        sys.exit(f"{sys.argv[2]} references no same-site assets to check")

    failures = []
    for asset in assets:
        status, _ = fetch(asset)
        print(f"{status} {asset}")
        if status != 200:
            failures.append(asset)
    if failures:
        sys.exit(f"{len(failures)} asset(s) failed to load from {url}")


if __name__ == "__main__":
    main()
