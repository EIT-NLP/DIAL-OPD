"""Install the exact validated TRL runtime without vendoring unrelated trainers.

The immutable first-release snapshot includes the PowerOPD fork plus its
restored TRL model helpers. Installing stock TRL would change this runtime.
"""
import argparse
import hashlib
import io
from pathlib import Path
import sys
import sysconfig
import tarfile
import urllib.request

REVISION = '881d86f06ca908bf9deedfecbd29bea8c4fe650c'
URL = f'https://codeload.github.com/EIT-NLP/DIAL-OPD/tar.gz/{REVISION}'
SHA256 = '960c2d71d658bf58a8901b38a6d6f82b953eb50abac88bf775c9df8353710095'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Optional offline copy of the pinned archive')
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise SystemExit('Activate a dedicated virtual environment first')
    target = Path(sys.prefix) / 'dial-runtime' / 'trl'
    data = args.archive.read_bytes() if args.archive else urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit('TRL source archive checksum mismatch')
    prefix = f'DIAL-OPD-{REVISION}/trl/'
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        files = {}
        for member in archive.getmembers():
            if not member.name.startswith(prefix) or not member.isfile():
                continue
            relative = Path(member.name[len(prefix):])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive path')
            files[relative] = archive.extractfile(member).read()
        if Path('pyproject.toml') not in files:
            raise ValueError('Pinned archive is missing the TRL package')
        # An existing dependency must be the same snapshot, never overwritten.
        if target.exists():
            actual = {p.relative_to(target) for p in target.rglob('*') if p.is_file()
                      and '__pycache__' not in p.parts and not any(x.endswith('.egg-info') for x in p.parts)}
            if actual != set(files) or any((target/p).read_bytes() != data for p, data in files.items()):
                raise SystemExit(f'Existing runtime differs from the pinned snapshot: {target}')
        else:
            for relative, content in files.items():
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    # This pure-Python runtime is imported directly, as in the validated release.
    # A venv-local .pth avoids building wheels (and cross-directory renames on
    # confined hosts). It neither changes dependencies nor modifies base Python.
    registration = Path(sysconfig.get_path('purelib')) / 'dial_trl_runtime.pth'
    content = str(target) + '\n'
    if registration.exists() and registration.read_text() != content:
        raise SystemExit(f'Refusing to overwrite a different runtime registration: {registration}')
    registration.write_text(content)
    print(f'Installed pinned TRL from {REVISION} into {sys.prefix}')


if __name__ == '__main__':
    main()
