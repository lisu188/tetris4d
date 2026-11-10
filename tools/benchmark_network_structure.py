import argparse
import dataclasses
import importlib.util
import json
import random
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

BASE = 'b3f2e8b4eea26f2f3fdeb18a2c7c4666103ffc0a'
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'sparse_tetris_4d_optimized.py'


def module(name, source):
    result = types.ModuleType(name)
    sys.modules[name] = result
    exec(compile(source, name + '.py', 'exec'), result.__dict__)
    return result


def size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        total += sum(size(item, seen) for item in value)
    elif dataclasses.is_dataclass(value):
        total += sum(size(getattr(value, field.name), seen) for field in dataclasses.fields(value))
    elif isinstance(value, types.ModuleType):
        raise ValueError('Measure owned structures, not modules.')
    elif hasattr(value, '__slots__'):
        total += sum(size(getattr(value, name), seen) for name in value.__slots__)
    return total


def measure(call, heights):
    times = []
    for _ in range(5):
        start = time.perf_counter()
        for h in heights:
            call(h)
        times.append((time.perf_counter() - start) * 1000 / len(heights))
    return {'median_ms': statistics.median(times), 'repetitions_ms': times}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--backend', choices=('python', 'numpy'), default='numpy')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.baseline.read_text(encoding='utf-8') if args.baseline else subprocess.run(
        ['git', 'show', BASE + ':' + SCRIPT], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    old = module('tetris_baseline', source)
    new = module('tetris_compact', (ROOT / SCRIPT).read_text(encoding='utf-8'))
    old.configure_backend(args.backend)
    new.configure_backend(args.backend)
    rng = random.Random(8303)
    heights = [tuple(rng.randrange(17) for _ in range(new.AREA)) for _ in range(20)]
    matched = 0
    for h in heights:
        a, b = old.network_forward(h, 10, 0), new.network_forward(h, 10, 0)
        assert a == b
        assert old.sigmoid_roughness(h) == new.sigmoid_roughness(h)
        matched += len(a)

    def cold(mod, h, lazy=False):
        mod.sigmoid_roughness.cache_clear()
        if hasattr(mod, 'pooled_histograms'):
            mod.pooled_histograms.cache_clear()
        return mod.network_trace(h, 10, 0) if lazy else mod.network_forward(h, 10, 0)

    report = {'base_commit': BASE, 'python': sys.version, 'backend': args.backend,
              'fixture_seed': 8303, 'fixtures': len(heights), 'repetitions': 5,
              'exact_activation_matches': matched,
              'timings': {'baseline_full_trace_cold': measure(lambda h: cold(old, h), heights),
                          'compact_full_trace_cold': measure(lambda h: cold(new, h), heights),
                          'compact_lazy_trace_cold': measure(lambda h: cold(new, h, True), heights)},
              'stored_graph_bytes': {'baseline': size(old.NETWORK), 'compact': size(new.NETWORK)},
              'note': 'Graph memory excludes shared adjacency and response tables. Cold clears feature caches, not imports. Timings are network tracing, not whole-game FPS.'}
    for h in heights[:3]:
        new.network_trace(h, 10, 0)
    report['timings']['compact_lazy_trace_warm'] = measure(lambda h: new.network_trace(h, 10, 0), heights[:3] * 20)
    report['trace_bytes'] = {'baseline': size(old.network_forward(heights[0], 10, 0)),
                             'compact': size(new.network_trace(heights[0], 10, 0))}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
