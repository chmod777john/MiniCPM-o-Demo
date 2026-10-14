#!/bin/sh
# Original .deb inputs selected from the fixed Python base and signed Debian indexes.
set -eu
input=$1
recipe=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cmp "$recipe/debian-package-inputs.json" "$input/debian-package-inputs.json"
(cd "$input" && sha256sum -c SHA256SUMS)
for release in "$input"/lists/*InRelease; do
    gpgv --keyring /usr/share/keyrings/debian-archive-keyring.gpg "$release"
done
python - "$input" <<'PYCHECK'
from pathlib import Path
import hashlib, json, subprocess, sys
root = Path(sys.argv[1])
manifest = json.loads((root / 'debian-package-inputs.json').read_text())
records = manifest['packages']
for index in manifest['indexes']:
    path = root / 'lists' / index['filename']
    raw = subprocess.check_output(['/usr/lib/apt/apt-helper', 'cat-file', str(path)])
    assert len(raw) == index['uncompressed_bytes'] and hashlib.sha256(raw).hexdigest() == index['uncompressed_sha256']
    release = path.with_name(path.name.replace('_main_binary-amd64_Packages.lz4', '_InRelease'))
    section = release.read_text().split('SHA256:\n', 1)[1]
    row = next(line.split() for line in section.splitlines() if line.endswith(' main/binary-amd64/Packages'))
    assert row[0] == index['uncompressed_sha256'] and int(row[1]) == len(raw)
expected = ''.join(p['name']+'='+p['version']+'\n' for p in records)
assert (root / 'packages.lock').read_text() == expected
for p in records:
    data = (root / 'archives' / p['filename']).read_bytes()
    assert len(data) == p['bytes'] and hashlib.sha256(data).hexdigest() == p['sha256'], p['filename']
PYCHECK
mkdir -p /var/lib/apt/lists /var/cache/apt/archives /opt/o5
cp "$input"/lists/* /var/lib/apt/lists/
cp "$input"/archives/*.deb /var/cache/apt/archives/
export DEBIAN_FRONTEND=noninteractive
xargs -r -a "$input/packages.lock" apt-get install --no-download --yes --no-install-recommends
dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' > /opt/o5/debian-installed.tsv
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb
