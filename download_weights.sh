#!/usr/bin/env bash
# Fetch and verify the EdgeFace XS release used by the native API.
set -eu
destination="data/edgeface_xs_fp32.bin"
source_url="https://github.com/facex-engine/facex/releases/download/v1.0.0/edgeface_xs_fp32.bin"
expected="d23f7b3cf8bcda5e451b48cf012e7dc6c4963d5703fafb8893f956d0203691d8"

verify() {
    if command -v sha256sum >/dev/null; then
        actual=$(sha256sum < "$1") || return 1
    else
        actual=$(shasum -a 256 < "$1") || return 1
    fi
    [ "${actual%% *}" = "$expected" ]
}

if [ -f "$destination" ] && verify "$destination"; then
    echo "Verified $destination"
    exit 0
fi
mkdir -p data
temporary=$(mktemp "$destination.XXXXXX")
trap 'rm -f "$temporary"' EXIT
trap 'exit 1' HUP INT TERM
if command -v curl >/dev/null; then
    curl --fail --location --retry 2 --output "$temporary" "$source_url"
else
    wget -O "$temporary" "$source_url"
fi
if ! verify "$temporary"; then
    echo "EdgeFace XS checksum verification failed" >&2
    exit 1
fi
mv -f "$temporary" "$destination"
echo "Verified $destination"
