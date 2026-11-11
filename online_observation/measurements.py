"""Read measurement payloads for online replay without opening label or reference data."""
from __future__ import annotations

from bisect import bisect_right
from contextlib import ExitStack
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile


_LEVEL1_MANIFEST = Path(__file__).resolve().parents[1] / 'configs/data/mtad-level1-scene3000.json'
_DEVELOPMENT_SPLITS = ('T_core', 'T_head', 'V_select')
_MEASUREMENT_FIELDS = ['batch', 'source', 'time', 'lat', 'lon', 'vel', 'cou']
_MEASUREMENT_PATH = re.compile(r'(?:^|/)量测场景/场景-([0-9]+)\.csv$')
_LEVEL = re.compile(r'(?<![a-z0-9])level([12])(?![a-z0-9])', re.IGNORECASE)


def csv_rows(payload):
    """Decode raw measurement rows, leaving time and state checks to the replay."""
    return _csv_rows(payload, _MEASUREMENT_FIELDS, 'measurement')


def _csv_rows(payload, fields, kind):
    try:
        content = payload.decode('utf-8-sig')
    except UnicodeDecodeError:
        content = payload.decode('gb18030')
    reader = csv.DictReader(io.StringIO(content, newline=''), strict=True)
    if reader.fieldnames != fields:
        raise ValueError(f'{kind} CSV has an unexpected header')
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'{kind} CSV line {reader.line_num} has the wrong column count')
        rows.append({field: value.strip() for field, value in row.items()})
    return rows


class _SplitZipView:
    """A seekable view over ordered byte segments of a single ZIP."""

    def __init__(self, parts):
        self._parts = parts
        self._starts = [0]
        for part in parts:
            size = os.fstat(part.fileno()).st_size
            if not size:
                raise ValueError('archive parts must be nonempty')
            self._starts.append(self._starts[-1] + size)
        self._position = 0

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_CUR:
            offset += self._position
        elif whence == os.SEEK_END:
            offset += self._starts[-1]
        elif whence != os.SEEK_SET:
            raise ValueError('invalid archive seek origin')
        if offset < 0:
            raise OSError('archive seek before the first byte')
        self._position = offset
        return offset

    def read(self, size=-1):
        end = self._starts[-1]
        if size >= 0:
            end = min(end, self._position + size)
        chunks = []
        while self._position < end:
            index = bisect_right(self._starts, self._position) - 1
            part = self._parts[index]
            part.seek(self._position - self._starts[index])
            length = min(end, self._starts[index + 1]) - self._position
            chunk = part.read(length)
            if len(chunk) != length:
                raise OSError('archive part was truncated while reading')
            self._position += length
            chunks.append(chunk)
        return b''.join(chunks)


def _development_inventory(path, scene, required_split):
    roster = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(roster, dict) or roster.get('dataset_level') != 'Level1':
        raise ValueError('a frozen Level1 roster is required')
    splits = roster.get('splits')
    if not isinstance(splits, dict) or any(
            not isinstance(splits.get(name), list) for name in (*_DEVELOPMENT_SPLITS, 'T_test')):
        raise ValueError('Level1 roster must declare development and sealed test splits')
    if scene in splits['T_test'] or sum(scene in splits[name] for name in _DEVELOPMENT_SPLITS) != 1:
        raise ValueError('scene must be selected for development in the frozen Level1 roster')
    if required_split is not None and scene not in splits[required_split]:
        raise ValueError(f'scene must belong to {required_split} in the frozen Level1 roster')
    scenes = roster.get('all_scenes')
    if not isinstance(scenes, list) or any(not isinstance(sid, str) for sid in scenes):
        raise ValueError('Level1 roster must list the scene inventory')
    inventory = set(scenes)
    if len(inventory) != len(scenes) or scene not in inventory:
        raise ValueError('inconsistent Level1 scene inventory')
    return inventory


def _scene_members(archive, pattern, kind):
    members = {}
    for info in archive.infolist():
        if info.header_offset < 0 or info.header_offset + 30 + info.compress_size > archive.start_dir:
            raise ValueError('ZIP member offsets exceed the supplied bytes; check archive parts and order')
        if info.is_dir():
            continue
        name = info.filename
        if not info.flag_bits & 0x800:
            try:
                name = name.encode('cp437').decode('gbk')
            except UnicodeError:
                pass
        match = pattern.search(name)
        if match is None:
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError(f'unsafe ZIP {kind} member name')
        if set(_LEVEL.findall(str(path.parent.parent))) != {'1'}:
            raise ValueError(f'ZIP {kind} members must declare Level1')
        sid = str(int(match.group(1)))
        if sid in members:
            raise ValueError(f'duplicate {kind} members for scene {sid}')
        members[sid] = info
    return members


def read_archive_rows(parts, manifest_path, scene, *, required_split=None):
    """Read one Level1 development measurement member from ordered ZIP parts."""
    payload = _read_development_member(
        parts, manifest_path, scene, _MEASUREMENT_PATH, 'measurement', required_split=required_split)
    return csv_rows(payload)


def _read_development_member(parts, manifest_path, scene, pattern, kind, *, required_split=None):
    frozen = _development_inventory(_LEVEL1_MANIFEST, scene, required_split)
    requested = _development_inventory(manifest_path, scene, required_split)
    paths = [Path(path) for path in parts]
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError('archive requires distinct, ordered ZIP parts')

    with ExitStack() as resources:
        handles = [resources.enter_context(path.open('rb')) for path in paths]
        source = handles[0] if len(handles) == 1 else _SplitZipView(handles)
        archive = resources.enter_context(ZipFile(source))
        members = _scene_members(archive, pattern, kind)
        if members.keys() != frozen or members.keys() != requested:
            raise ValueError(f'ZIP {kind} scene inventory differs from the frozen Level1 roster')
        return archive.read(members[scene])
