from pathlib import Path
import hashlib
import json
import lzma
import subprocess


def blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


assert blob('sparse_tetris_4d_optimized.py') == '8aaab5a7ce5bc11f23fa129634e69093f0335170'
assert blob('tetris_benchmark_logging.py') == 'b52f8a60b2944ef7806fcb4f23e9e28a3fd43b0a'
archive = Path('.ci/all-sections-frameskip.xz').read_bytes()
assert hashlib.sha256(archive).hexdigest() == '5626ab1636c197d7be8ab84cd95365c9b3cfcac6dd6b9726c869a21ec0b7c9ff'
data = json.loads(lzma.decompress(archive))
subprocess.run(['git', 'apply', '--check', '-'], input=data['source_patch'], text=True, check=True)
subprocess.run(['git', 'apply', '-'], input=data['source_patch'], text=True, check=True)
assert hashlib.sha256(Path('sparse_tetris_4d_optimized.py').read_bytes()).hexdigest() == data['source_sha256']
for path, key in (('tetris_benchmark_logging.py', 'logger_replacements'), ('README.md', 'readme_replacements')):
    target = Path(path)
    text = target.read_text(encoding='utf-8')
    for before, after in data[key]:
        assert text.count(before) == 1, (path, before[:100])
        text = text.replace(before, after)
    target.write_text(text, encoding='utf-8')
for name, content in data['files'].items():
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
assert blob('native/tetris_native.c') == '4b5e8351abd1c433e5b31cf72ba50e9ad2533b48'
assert blob('models/trained-sigmoid.json') == '9c2e7937cd2aef5841d83b2309715d68f9b329ef'
