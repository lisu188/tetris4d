from __future__ import annotations

import argparse
import bisect
import itertools
import json
import math
import random
import statistics
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Generator, Iterator

WIDTH = DEPTH = HEIGHT = FOURTH = 16
SLICE_AREA = WIDTH * DEPTH
AREA = SLICE_AREA * FOURTH
INPUT_COUNT = AREA + 2
EMPTY_BOARD = (0,) * AREA
HEIGHT_WEIGHT = -0.510066
LAYERS_WEIGHT = 0.760666
HOLES_WEIGHT = -0.356630
BUMP_WEIGHT = -0.184483
SIGMOID_SLOPE = 1.0
SIGMOID_THRESHOLD = 2.0
ROUGHNESS_SCALE = 6.0
DEFAULT_BEAM = 4
DEFAULT_CANDIDATES = 2048
BATCH_SIZE = 512
BACKEND = "auto"
_NUMPY = None
_NUMPY_TRIED = False
_NATIVE = None
_NATIVE_INFO = {'available': False, 'reason': 'Not initialized'}
_NATIVE_USE_AVX2 = False
_NATIVE_USE_BMI2 = False
AXIS_NAMES = 'XYZW'
PLANES = tuple(itertools.combinations(range(4), 2))
PLANE_NAMES = tuple(AXIS_NAMES[a] + AXIS_NAMES[b] for a, b in PLANES)
Board = tuple[int, ...]
Cells = tuple[tuple[int, int, int, int], ...]
RULES = ('16x16x16x16 placement-based falling blocks; X,Y,W are horizontal and Z is gravity. '
         'Rigid pieces contain 4 to 16 cells; eight new shapes have four-dimensional center geometry. '
         'Rotate in the XY, XZ, XW, YZ, YW, ZW planes. '
         'Clear a Z hyperlayer only when all 4096 X/Y/W cells are occupied. '
         'No independent cell gravity, wall kicks, hold, slides, or timed input constraints.')


def column_index(x: int, y: int, w: int) -> int:
    return (w * DEPTH + y) * WIDTH + x


def column_coordinates(index: int) -> tuple[int, int, int]:
    return index % WIDTH, index // WIDTH % DEPTH, index // SLICE_AREA


X_PAIRS = tuple((column_index(x, y, w), column_index(x + 1, y, w))
                for w in range(FOURTH) for y in range(DEPTH) for x in range(WIDTH - 1))
Y_PAIRS = tuple((column_index(x, y, w), column_index(x, y + 1, w))
                for w in range(FOURTH) for y in range(DEPTH - 1) for x in range(WIDTH))
W_PAIRS = tuple((column_index(x, y, w), column_index(x, y, w + 1))
                for w in range(FOURTH - 1) for y in range(DEPTH) for x in range(WIDTH))
NEIGHBORS = X_PAIRS + Y_PAIRS + W_PAIRS
PAIR_INDEX = {pair: i for i, pair in enumerate(NEIGHBORS)}
_neighbors = [[] for _ in range(AREA)]
for _a, _b in NEIGHBORS:
    _neighbors[_a].append(_b)
    _neighbors[_b].append(_a)
ADJACENT = tuple(tuple(row) for row in _neighbors)
del _neighbors, _a, _b


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    edges: tuple[tuple[int, float], ...]
    bias: float
    activation: str


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


SIGMOID_BASELINE = 2 * sigmoid(-SIGMOID_THRESHOLD)
HIDDEN_OUTPUT_WEIGHT = BUMP_WEIGHT * ROUGHNESS_SCALE
OUTPUT_BIAS = -HIDDEN_OUTPUT_WEIGHT * len(NEIGHBORS) * SIGMOID_BASELINE


def roughness_term(difference: float) -> float:
    return ROUGHNESS_SCALE * (sigmoid(difference - SIGMOID_THRESHOLD)
                              + sigmoid(-difference - SIGMOID_THRESHOLD) - SIGMOID_BASELINE)


ROUGHNESS_TABLE = tuple(roughness_term(d) for d in range(-HEIGHT, HEIGHT + 1))


SIGMOID_TABLE = tuple(sigmoid(d - SIGMOID_THRESHOLD) for d in range(-HEIGHT, HEIGHT + 1))


def _integer_table(values: tuple[float, ...]) -> tuple[tuple[int, ...], int]:
    ratios = tuple(value.as_integer_ratio() for value in values)
    denominator = max(d for _, d in ratios)
    return tuple(n * (denominator // d) for n, d in ratios), denominator


ROUGHNESS_TICKS, ROUGHNESS_DENOMINATOR = _integer_table(ROUGHNESS_TABLE[HEIGHT:])
_OUTPUT_SIGNALS = tuple(h * HEIGHT_WEIGHT for h in range(HEIGHT + 1)) + tuple(
    value * HIDDEN_OUTPUT_WEIGHT for value in SIGMOID_TABLE)
_OUTPUT_TICKS, OUTPUT_DENOMINATOR = _integer_table(_OUTPUT_SIGNALS)
HEIGHT_OUTPUT_TICKS = _OUTPUT_TICKS[:HEIGHT + 1]
PAIR_OUTPUT_TICKS = tuple(_OUTPUT_TICKS[HEIGHT + 1 + HEIGHT + d]
                          + _OUTPUT_TICKS[HEIGHT + 1 + HEIGHT - d] for d in range(HEIGHT + 1))
PAIR_ACTIVE_COUNTS = tuple(int(SIGMOID_TABLE[HEIGHT + d] >= .5)
                           + int(SIGMOID_TABLE[HEIGHT - d] >= .5) for d in range(HEIGHT + 1))
del _OUTPUT_SIGNALS, _OUTPUT_TICKS


@lru_cache(maxsize=16)
def pooled_histograms(heights: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(heights) != AREA or any(not isinstance(h, int) or not 0 <= h <= HEIGHT for h in heights):
        raise ValueError('Histogram pooling requires 4096 integer column heights in [0, 16].')
    if BACKEND in ('numpy', 'native'):
        np, a, b, _ = numpy_tables()
        values = np.asarray(heights, dtype=np.int16)
        counts = np.bincount(values, minlength=HEIGHT + 1)
        differences = np.bincount(np.abs(values[b] - values[a]), minlength=HEIGHT + 1)
        return tuple(map(int, counts)), tuple(map(int, differences))
    counts, differences = [0] * (HEIGHT + 1), [0] * (HEIGHT + 1)
    for h in heights:
        counts[h] += 1
    for a, b in NEIGHBORS:
        differences[abs(heights[b] - heights[a])] += 1
    return tuple(counts), tuple(differences)


@lru_cache(maxsize=16)
def sigmoid_roughness(heights: tuple[int, ...]) -> float:
    if BACKEND not in ('numpy', 'native'):
        return math.fsum(ROUGHNESS_TABLE[heights[b] - heights[a] + HEIGHT] for a, b in NEIGHBORS)
    counts = pooled_histograms(heights)[1]
    return sum(count * tick for count, tick in zip(counts, ROUGHNESS_TICKS)) / ROUGHNESS_DENOMINATOR


class SharedSigmoidNetwork(Sequence):
    __slots__ = ()

    def __len__(self) -> int:
        return 2 * len(NEIGHBORS) + 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if not isinstance(index, int):
            raise TypeError('Neuron indices must be integers or slices.')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('Neuron index out of range.')
        if index == len(self) - 1:
            edges = tuple((i, HEIGHT_WEIGHT) for i in range(AREA))
            edges += ((AREA, HOLES_WEIGHT), (AREA + 1, LAYERS_WEIGHT))
            edges += tuple((INPUT_COUNT + i, HIDDEN_OUTPUT_WEIGHT) for i in range(index))
            return Node('score', edges, OUTPUT_BIAS, 'linear')
        pair, direction = divmod(index, 2)
        a, b = NEIGHBORS[pair]
        axis = 'X' if pair < len(X_PAIRS) else 'Y' if pair < len(X_PAIRS) + len(Y_PAIRS) else 'W'
        x, y, w = column_coordinates(a)
        sign = 1.0 if direction == 0 else -1.0
        return Node(f'{axis}{"+" if direction == 0 else "-"}_{x},{y},{w}',
                    ((b, sign), (a, -sign)), -SIGMOID_THRESHOLD, 'sigmoid')


def make_network() -> SharedSigmoidNetwork:
    return SharedSigmoidNetwork()


NETWORK = make_network()
OUTPUT_INDEX = INPUT_COUNT + len(NETWORK) - 1


def _pooled_output(histograms, holes: float, layers: float) -> float:
    heights, differences = histograms
    numerator = sum(count * tick for count, tick in zip(heights, HEIGHT_OUTPUT_TICKS))
    numerator += sum(count * tick for count, tick in zip(differences, PAIR_OUTPUT_TICKS))
    denominator = OUTPUT_DENOMINATOR
    for signal in (holes * HOLES_WEIGHT, layers * LAYERS_WEIGHT):
        n, d = signal.as_integer_ratio()
        if d > denominator:
            numerator *= d // denominator
            denominator = d
        numerator += n * (denominator // d)
    return OUTPUT_BIAS + numerator / denominator


class NetworkTrace(Sequence):
    __slots__ = ('inputs', 'histograms', 'output', 'active_count', 'revision')

    def __init__(self, heights: tuple[int, ...], holes: int, layers: int) -> None:
        if len(heights) != AREA:
            raise ValueError(f'Expected {AREA} column heights.')
        self.revision = MODEL_REVISION
        self.inputs = tuple(heights) + (float(holes), float(layers))
        if not math.isfinite(self.inputs[AREA]) or not math.isfinite(self.inputs[AREA + 1]):
            raise ValueError('Network inputs must be finite.')
        try:
            self.histograms = pooled_histograms(tuple(heights))
        except ValueError:
            self.histograms = None
        if self.histograms is not None:
            self.output = _pooled_output(self.histograms, self.inputs[AREA], self.inputs[AREA + 1])
            self.active_count = sum(n * active for n, active in zip(self.histograms[1], PAIR_ACTIVE_COUNTS))
        else:
            self.inputs = tuple(map(float, self.inputs))
            if not all(map(math.isfinite, self.inputs)):
                raise ValueError('Network inputs must be finite.')
            signals = [h * HEIGHT_WEIGHT for h in self.inputs[:AREA]]
            signals.extend((self.inputs[AREA] * HOLES_WEIGHT, self.inputs[AREA + 1] * LAYERS_WEIGHT))
            active = 0
            for a, b in NEIGHBORS:
                d = self.inputs[b] - self.inputs[a]
                u, v = sigmoid(d - SIGMOID_THRESHOLD), sigmoid(-d - SIGMOID_THRESHOLD)
                signals.extend((u * HIDDEN_OUTPUT_WEIGHT, v * HIDDEN_OUTPUT_WEIGHT))
                active += (u >= .5) + (v >= .5)
            self.output, self.active_count = OUTPUT_BIAS + math.fsum(signals), active

    def __len__(self) -> int:
        return OUTPUT_INDEX + 1

    def __getitem__(self, index):
        if self.revision != MODEL_REVISION:
            raise RuntimeError('The model changed; create a new activation trace.')
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not isinstance(index, int):
            raise TypeError('Activation indices must be integers or slices.')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('Activation index out of range.')
        if index < INPUT_COUNT:
            return float(self.inputs[index])
        if index == OUTPUT_INDEX:
            return self.output
        pair, direction = divmod(index - INPUT_COUNT, 2)
        a, b = NEIGHBORS[pair]
        d = self.inputs[b] - self.inputs[a]
        if direction:
            d = -d
        if self.histograms is not None:
            return SIGMOID_TABLE[int(d) + HEIGHT]
        return sigmoid(d - SIGMOID_THRESHOLD)

    def expand(self) -> list[float]:
        if self.revision != MODEL_REVISION:
            raise RuntimeError('The model changed; create a new activation trace.')
        values = list(map(float, self.inputs))
        if self.histograms is not None and BACKEND in ('numpy', 'native'):
            np, a, b, _ = numpy_tables()
            inputs = np.asarray(self.inputs[:AREA], dtype=np.int16)
            differences = inputs[b] - inputs[a]
            table = numpy_network_data()
            hidden = np.empty((len(NEIGHBORS), 2), dtype=np.float64)
            hidden[:, 0] = table[differences + HEIGHT]
            hidden[:, 1] = table[HEIGHT - differences]
            values.extend(hidden.ravel().tolist())
        elif self.histograms is not None:
            for a, b in NEIGHBORS:
                d = int(self.inputs[b] - self.inputs[a])
                values.extend((SIGMOID_TABLE[d + HEIGHT], SIGMOID_TABLE[HEIGHT - d]))
        else:
            for a, b in NEIGHBORS:
                d = self.inputs[b] - self.inputs[a]
                values.extend((sigmoid(d - SIGMOID_THRESHOLD), sigmoid(-d - SIGMOID_THRESHOLD)))
        values.append(self.output)
        return values


def network_trace(heights: tuple[int, ...], holes: int, layers: int) -> NetworkTrace:
    return NetworkTrace(heights, holes, layers)


def network_forward(heights: tuple[int, ...], holes: int, layers: int) -> list[float]:
    return network_trace(heights, holes, layers).expand()


def network_score(heights: tuple[int, ...], holes: int, layers: int) -> float:
    return HEIGHT_WEIGHT * sum(heights) + HOLES_WEIGHT * holes + LAYERS_WEIGHT * layers + BUMP_WEIGHT * sigmoid_roughness(heights)


BASELINE_PARAMETERS = {'height': HEIGHT_WEIGHT, 'holes': HOLES_WEIGHT,
                       'layers': LAYERS_WEIGHT, 'roughness': BUMP_WEIGHT,
                       'threshold': SIGMOID_THRESHOLD}
MODEL_TRAINING = {'method': 'none', 'description': 'Untrained inherited heuristic coefficients.'}
MODEL_REVISION = 0


def network_parameters() -> dict[str, float]:
    return {'height': HEIGHT_WEIGHT, 'holes': HOLES_WEIGHT, 'layers': LAYERS_WEIGHT,
            'roughness': BUMP_WEIGHT, 'threshold': SIGMOID_THRESHOLD}


def configure_network(parameters: dict[str, float], training: dict | None = None) -> None:
    global HEIGHT_WEIGHT, HOLES_WEIGHT, LAYERS_WEIGHT, BUMP_WEIGHT, SIGMOID_THRESHOLD
    global SIGMOID_BASELINE, HIDDEN_OUTPUT_WEIGHT, OUTPUT_BIAS, ROUGHNESS_TABLE, SIGMOID_TABLE
    global ROUGHNESS_TICKS, ROUGHNESS_DENOMINATOR, HEIGHT_OUTPUT_TICKS, PAIR_OUTPUT_TICKS
    global OUTPUT_DENOMINATOR, PAIR_ACTIVE_COUNTS, MODEL_TRAINING, MODEL_REVISION
    if not isinstance(parameters, dict) or set(parameters) != set(BASELINE_PARAMETERS):
        raise ValueError('Model parameters must contain height, holes, layers, roughness and threshold.')
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in parameters.values()):
        raise ValueError('Model parameters must be finite numbers, not booleans or strings.')
    values = {key: float(value) for key, value in parameters.items()}
    if not (-100 <= values['height'] < 0 and -100 <= values['holes'] <= 0
            and 0 <= values['layers'] <= 100 and -100 <= values['roughness'] <= 0
            and .25 <= values['threshold'] <= 6):
        raise ValueError('Model parameters exceed supported bounds or have invalid signs.')
    if training is not None and not isinstance(training, dict):
        raise ValueError('Training metadata must be an object.')
    metadata = json.loads(json.dumps(training or {'method': 'custom'}, allow_nan=False))
    HEIGHT_WEIGHT, HOLES_WEIGHT, LAYERS_WEIGHT = values['height'], values['holes'], values['layers']
    BUMP_WEIGHT, SIGMOID_THRESHOLD = values['roughness'], values['threshold']
    SIGMOID_BASELINE = 2 * sigmoid(-SIGMOID_THRESHOLD)
    HIDDEN_OUTPUT_WEIGHT = BUMP_WEIGHT * ROUGHNESS_SCALE
    OUTPUT_BIAS = -HIDDEN_OUTPUT_WEIGHT * len(NEIGHBORS) * SIGMOID_BASELINE
    ROUGHNESS_TABLE = tuple(roughness_term(d) for d in range(-HEIGHT, HEIGHT + 1))
    SIGMOID_TABLE = tuple(sigmoid(d - SIGMOID_THRESHOLD) for d in range(-HEIGHT, HEIGHT + 1))
    ROUGHNESS_TICKS, ROUGHNESS_DENOMINATOR = _integer_table(ROUGHNESS_TABLE[HEIGHT:])
    signals = tuple(h * HEIGHT_WEIGHT for h in range(HEIGHT + 1)) + tuple(
        value * HIDDEN_OUTPUT_WEIGHT for value in SIGMOID_TABLE)
    ticks, OUTPUT_DENOMINATOR = _integer_table(signals)
    HEIGHT_OUTPUT_TICKS = ticks[:HEIGHT + 1]
    PAIR_OUTPUT_TICKS = tuple(ticks[HEIGHT + 1 + HEIGHT + d] + ticks[HEIGHT + 1 + HEIGHT - d]
                             for d in range(HEIGHT + 1))
    PAIR_ACTIVE_COUNTS = tuple(int(SIGMOID_TABLE[HEIGHT + d] >= .5) + int(SIGMOID_TABLE[HEIGHT - d] >= .5)
                               for d in range(HEIGHT + 1))
    for function in (pooled_histograms, sigmoid_roughness, evaluate_board, numpy_tables, numpy_network_data):
        function.cache_clear()
    MODEL_TRAINING = metadata
    MODEL_REVISION += 1


def load_checkpoint(path: Path | str) -> dict:
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError('Checkpoint exceeds the 2 MB limit.')
    checkpoint = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(checkpoint, dict) or checkpoint.get('schema') != 1
            or checkpoint.get('kind') != 'tetris4d-shared-sigmoid'
            or checkpoint.get('well') != [WIDTH, DEPTH, HEIGHT, FOURTH]):
        raise ValueError('Incompatible Tetris 4D checkpoint.')
    configure_network(checkpoint.get('parameters'), checkpoint.get('training'))
    return checkpoint


def network_metadata(include_nodes: bool = False) -> dict:
    result = {'well': [WIDTH, DEPTH, HEIGHT, FOURTH], 'axis_order': 'X,Y,Z,W', 'gravity': '-Z',
              'input_count': INPUT_COUNT, 'height_inputs': AREA, 'sigmoid_units': len(NETWORK) - 1,
              'linear_output_units': 1, 'neighbor_pairs': len(NEIGHBORS),
              'computational_units': len(NETWORK), 'displayed_nodes': OUTPUT_INDEX + 1,
              'nonzero_connections': INPUT_COUNT + 6 * len(NEIGHBORS), 'nonzero_biases': len(NETWORK),
              'execution': {'representation': 'shared sigmoid stencil with exact histogram pooling',
                            'shared_sigmoid_filters': 2, 'sigmoid_lookup_entries': len(SIGMOID_TABLE),
                            'filters': [{'weights': [1.0, -1.0], 'bias': -SIGMOID_THRESHOLD, 'activation': 'sigmoid'},
                                        {'weights': [-1.0, 1.0], 'bias': -SIGMOID_THRESHOLD, 'activation': 'sigmoid'}],
                            'pooling': 'Sum responses weighted by their integer occurrence counts.',
                            'absolute_difference_bins': HEIGHT + 1, 'height_bins': HEIGHT + 1,
                            'stored_neuron_objects': 0, 'stored_per_edge_weights': 0,
                            'logical_hidden_activations': len(NETWORK) - 1,
                            'expanded_graph': 'Logical nodes and edges are reconstructed on demand.',
                            'rounding': 'Integer accumulation of binary float terms, then one rounding; no reassociation drift.',
                            'ui': 'Only visible activations are reconstructed; the score covers every W page.'},
              'coefficients': {'height': HEIGHT_WEIGHT, 'holes': HOLES_WEIGHT,
                               'layers': LAYERS_WEIGHT, 'roughness': BUMP_WEIGHT},
              'sigmoid': {'threshold': SIGMOID_THRESHOLD, 'slope': SIGMOID_SLOPE,
                          'roughness_scale': ROUGHNESS_SCALE, 'hidden_output_weight': HIDDEN_OUTPUT_WEIGHT,
                          'output_bias': OUTPUT_BIAS},
              'training': dict(MODEL_TRAINING),
              'search': {'default_depth': 2, 'default_beam': DEFAULT_BEAM,
                         'candidate_limit_per_board': DEFAULT_CANDIDATES,
                         'sampling': 'Deterministic stratified sampling of orientations and origins; 0 means exhaustive.'},
              'rules': RULES, 'rotation_planes': PLANE_NAMES,
              'shape_orientations': {name: len(shapes) for name, shapes in SHAPES.items()},
              'piece_sizes': PIECE_SIZES, 'piece_sets': PIECE_SETS,
              'piece_note': 'Seven classic shapes plus eight connected shapes whose cell centers span four dimensions.'}
    if include_nodes:
        result['inputs'] = [f'height_{x},{y},{w}' for w in range(FOURTH) for y in range(DEPTH) for x in range(WIDTH)] + ['holes', 'hyperlayers_cleared']
        result['nodes'] = [asdict(node) for node in NETWORK]
    return result


def normalize(cells: Cells) -> Cells:
    minimum = tuple(min(cell[i] for cell in cells) for i in range(4))
    return tuple(sorted(tuple(cell[i] - minimum[i] for i in range(4)) for cell in cells))


def rotate_cells(cells: Cells, plane: int, direction: int = 1) -> Cells:
    if not 0 <= plane < len(PLANES) or direction not in (-1, 1):
        raise ValueError('Invalid rotation plane or direction.')
    a, b = PLANES[plane]
    result = []
    for cell in cells:
        point = list(cell)
        point[a], point[b] = -direction * cell[b], direction * cell[a]
        result.append(tuple(point))
    return normalize(tuple(result))


@dataclass(frozen=True, slots=True)
class Shape:
    cells: Cells
    extents: tuple[int, int, int, int]
    columns: tuple[tuple[int, int, int], ...]
    levels: tuple[tuple[int, int], ...]


def rotations(cells: Cells) -> tuple[Shape, ...]:
    first = normalize(cells)
    pending, seen, result = deque([first]), {first}, []
    while pending:
        cells = pending.popleft()
        extents = tuple(max(cell[i] for cell in cells) + 1 for i in range(4))
        masks = {}
        levels = {}
        for x, y, z, w in cells:
            index = column_index(x, y, w)
            masks[index] = masks.get(index, 0) | (1 << z)
            levels[z] = levels.get(z, 0) + 1
        columns = tuple((index, mask, (mask & -mask).bit_length() - 1) for index, mask in sorted(masks.items()))
        result.append(Shape(cells, extents, columns, tuple(sorted(levels.items()))))
        for plane in range(6):
            following = rotate_cells(cells, plane)
            if following not in seen:
                seen.add(following)
                pending.append(following)
    return tuple(result)


CLASSIC_PIECE_CELLS = {
    'I': ((0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0)),
    'O': ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)),
    'T': ((0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (1, 1, 0, 0)),
    'L': ((0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (0, 1, 0, 0)),
    'S': ((0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (2, 1, 0, 0)),
    'B': ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
    'C': ((0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0)),
}
NEW_PIECE_CELLS = {
    'NOVA': ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    'STAIR': ((0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 1, 1)),
    'FORK': ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 1, 1)),
    'HOOK': ((0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (2, 1, 0, 0), (2, 1, 1, 0), (2, 1, 1, 1)),
    'BRIDGE': ((0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0), (3, 0, 0, 0),
               (3, 1, 0, 0), (3, 1, 1, 0), (3, 1, 1, 1), (3, 1, 1, 2)),
    'RING': ((0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 0),
             (1, 1, 1, 1), (0, 1, 1, 1), (0, 0, 1, 1), (0, 0, 0, 1)),
    'CROSS': ((0, 0, 0, 0), (1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0),
              (0, -1, 0, 0), (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1)),
    'HYPER': tuple(itertools.product(range(2), repeat=4)),
}
PIECE_DESCRIPTIONS = {
    'I': 'Straight four-cell bar.', 'O': 'Flat square.', 'T': 'Flat T junction.',
    'L': 'Flat elbow.', 'S': 'Flat zigzag.', 'B': 'Three-axis corner.', 'C': 'Three-axis staircase.',
    'NOVA': 'A center with one arm on each of the four axes.',
    'STAIR': 'A five-cell path stepping through X, Y, Z and W.',
    'FORK': 'A three-arm junction with its last arm bent into W.',
    'HOOK': 'An extended bar turning successively through the remaining axes.',
    'BRIDGE': 'A long bridge with a three-plane bend and an extended W tail.',
    'RING': 'An eight-cell closed cycle spanning all four axes.',
    'CROSS': 'A center with arms in both directions of every axis.',
    'HYPER': 'A solid 2 by 2 by 2 by 2 block of sixteen hypercubes.',
}
PIECE_CELLS = {name: normalize(cells) for name, cells in (CLASSIC_PIECE_CELLS | NEW_PIECE_CELLS).items()}
PIECE_NAMES = tuple(PIECE_CELLS)
PIECE_SIZES = {name: len(cells) for name, cells in PIECE_CELLS.items()}
PIECE_SETS = {'classic': tuple(CLASSIC_PIECE_CELLS), 'expanded': PIECE_NAMES, '4d': tuple(NEW_PIECE_CELLS)}
SHAPES = {name: rotations(cells) for name, cells in PIECE_CELLS.items()}
SHAPE_INDEX = {name: {s.cells: i for i, s in enumerate(shapes)} for name, shapes in SHAPES.items()}
ROTATION_NEXT = {name: tuple(tuple(SHAPE_INDEX[name][rotate_cells(s.cells, p)] for p in range(6)) for s in shapes)
                 for name, shapes in SHAPES.items()}


COLUMN_TOPS = {name: tuple(tuple((offset, bottom, mask.bit_length()) for offset, mask, bottom in shape.columns)
                          for shape in shapes) for name, shapes in SHAPES.items()}


def affine_dimension(cells: Cells) -> int:
    rows = [[Fraction(value - origin) for value, origin in zip(cell, cells[0])] for cell in cells[1:]]
    rank = 0
    for column in range(4):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][column]
        rows[rank] = [value / divisor for value in rows[rank]]
        for i in range(rank + 1, len(rows)):
            factor = rows[i][column]
            rows[i] = [a - factor * b for a, b in zip(rows[i], rows[rank])]
        rank += 1
    return rank


def cells_connected(cells: Cells) -> bool:
    if not cells:
        return False
    remaining = set(cells)
    pending = [remaining.pop()]
    while pending:
        cell = pending.pop()
        for axis in range(4):
            for direction in (-1, 1):
                point = tuple(value + (direction if i == axis else 0) for i, value in enumerate(cell))
                if point in remaining:
                    remaining.remove(point)
                    pending.append(point)
    return not remaining


def piece_catalog() -> dict:
    return {'piece_sets': PIECE_SETS, 'new_piece_count': len(NEW_PIECE_CELLS),
            'total_piece_count': len(PIECE_NAMES), 'coordinate_order': AXIS_NAMES,
            'pieces': {name: {'new': name in NEW_PIECE_CELLS, 'cell_count': PIECE_SIZES[name],
                             'center_affine_dimension': affine_dimension(cells),
                             'orientations': len(SHAPES[name]), 'cells': cells,
                             'description': PIECE_DESCRIPTIONS[name]}
                       for name, cells in PIECE_CELLS.items()}}


def clear_points(layers: int) -> int:
    if not 0 <= layers <= HEIGHT:
        raise ValueError('Cleared hyperlayer count is outside the board.')
    return (0, 100, 300, 500, 800)[layers] if layers <= 4 else 800 + 400 * (layers - 4)


def board_features(board: Board) -> tuple[tuple[int, ...], int]:
    heights = tuple(map(int.bit_length, board))
    return heights, sum(heights) - sum(map(int.bit_count, board))


def full_layer_mask(board: Board) -> int:
    full = (1 << HEIGHT) - 1
    for column in board:
        full &= column
        if not full:
            break
    return full


def clear_layers(board: Board) -> tuple[Board, int]:
    full = full_layer_mask(board)
    if not full:
        return board, 0
    count = full.bit_count()
    if BACKEND == 'native':
        columns = _NUMPY.asarray(board, dtype=_NUMPY.uint16)
        result = _NUMPY.empty(AREA, dtype=_NUMPY.uint16)
        _native_check(_NATIVE.t4_compact(columns.ctypes.data, result.ctypes.data, full, int(_NATIVE_USE_BMI2)))
        return tuple(map(int, result)), count
    columns = list(board)
    while full:
        z = full.bit_length() - 1
        low = (1 << z) - 1
        columns = [(column & low) | ((column >> (z + 1)) << z) for column in columns]
        full ^= 1 << z
    return tuple(columns), count


@dataclass(slots=True)
class BoardEvaluation:
    board: Board
    heights: tuple[int, ...]
    height_sum: int
    holes: int
    roughness: float
    fill_counts: tuple[int, ...]
    score: float
    near_clear: bool = False
    array_heights: object = None
    native_board: object = None
    native_heights: object = None
    native_counts: object = None


@lru_cache(maxsize=8)
def evaluate_board(board: Board) -> BoardEvaluation:
    if len(board) != AREA or any(not isinstance(v, int) or v < 0 or v >= 1 << HEIGHT for v in board):
        raise ValueError('Invalid 4D board.')
    native = _native_features(board) if BACKEND == 'native' else None
    if native is not None:
        heights = tuple(map(int, native[1]))
        holes = int(native[3][1])
        counts = tuple(map(int, native[2]))
    else:
        heights, holes = board_features(board)
        counts = [0] * HEIGHT
        for mask in board:
            while mask:
                bit = mask & -mask
                counts[bit.bit_length() - 1] += 1
                mask ^= bit
    roughness = sigmoid_roughness(heights)
    height_sum = sum(heights)
    score = HEIGHT_WEIGHT * height_sum + HOLES_WEIGHT * holes + BUMP_WEIGHT * roughness
    context = BoardEvaluation(board, heights, height_sum, holes, roughness, tuple(counts), score,
                              max(counts) + max(PIECE_SIZES.values()) >= AREA)
    if native is not None:
        context.native_board, context.native_heights, context.native_counts = native[:3]
    return context


@dataclass(frozen=True, slots=True)
class Candidate:
    piece: str
    rotation: int
    x: int
    y: int
    w: int
    z: int
    layers: int
    score: float


@dataclass(frozen=True, slots=True)
class Move:
    candidate: Candidate
    before: Board
    board: Board
    heights: tuple[int, ...]
    holes: int

    @property
    def piece(self) -> str:
        return self.candidate.piece

    @property
    def score(self) -> float:
        return self.candidate.score

    @property
    def layers(self) -> int:
        return self.candidate.layers

    @property
    def roughness(self) -> float:
        return sigmoid_roughness(self.heights)


@dataclass(frozen=True, slots=True)
class LockedPiece:
    identifier: int
    piece: str
    cells: Cells


def placed_cells(move: Move | Candidate) -> Cells:
    p = move.candidate if isinstance(move, Move) else move
    return tuple((p.x + x, p.y + y, p.z + z, p.w + w) for x, y, z, w in SHAPES[p.piece][p.rotation].cells)


def animated_piece_cells(move: Move, progress: float) -> tuple[tuple[int, int, float, int], ...]:
    if not math.isfinite(progress):
        raise ValueError('Animation progress must be finite.')
    p = move.candidate
    shape = SHAPES[p.piece][p.rotation]
    start = HEIGHT - shape.extents[2]
    z = start + (p.z - start) * min(1.0, max(0.0, progress))
    return tuple((p.x + dx, p.y + dy, z + dz, p.w + dw) for dx, dy, dz, dw in shape.cells)


def merge_rigid_piece(board: Board, cells: Cells, piece: str | None = None) -> Board:
    if not cells or len(set(cells)) != len(cells):
        raise ValueError('A piece must contain distinct hypercubes.')
    if piece is not None and (piece not in SHAPE_INDEX or normalize(cells) not in SHAPE_INDEX[piece]):
        raise ValueError('The complete registered rigid shape must be preserved.')
    if len(board) != AREA:
        raise ValueError('Invalid 4D board size.')
    result = list(board)
    for x, y, z, w in cells:
        if not all(isinstance(v, int) and 0 <= v < 16 for v in (x, y, z, w)):
            raise ValueError('The entire piece must fit before locking.')
        i, bit = column_index(x, y, w), 1 << z
        if result[i] & bit:
            raise ValueError('A rigid piece cannot overlap occupied cells.')
        result[i] |= bit
    return tuple(result)


def compact_locked_pieces(pieces: tuple[LockedPiece, ...], cleared: int) -> tuple[LockedPiece, ...]:
    if not cleared:
        return pieces
    result = []
    for piece in pieces:
        cells = tuple((x, y, z - (cleared & ((1 << z) - 1)).bit_count(), w)
                      for x, y, z, w in piece.cells if not cleared & (1 << z))
        if cells:
            result.append(LockedPiece(piece.identifier, piece.piece, cells))
    return tuple(result)


def score_candidate(context: BoardEvaluation, piece: str, rotation: int, x: int, y: int, w: int) -> Candidate | None:
    shape = SHAPES[piece][rotation]
    sx, sy, sz, sw = shape.extents
    if not (0 <= x <= WIDTH - sx and 0 <= y <= DEPTH - sy and 0 <= w <= FOURTH - sw):
        return None
    base = (w * DEPTH + y) * WIDTH + x
    heights = context.heights
    columns = COLUMN_TOPS[piece][rotation]
    z = -HEIGHT
    for offset, bottom, top in columns:
        landing = heights[base + offset] - bottom
        if landing > z:
            z = landing
    if z + sz > HEIGHT:
        return None
    changed = {}
    delta = 0
    for offset, bottom, top in columns:
        i = base + offset
        value = z + top
        changed[i] = value
        delta += value - heights[i]
    if context.near_clear:
        if any(context.fill_counts[z + dz] + count == AREA for dz, count in shape.levels):
            provisional = Candidate(piece, rotation, x, y, w, z, 0, 0.0)
            board, layers = clear_layers(merge_rigid_piece(context.board, placed_cells(provisional), piece))
            new_heights, holes = board_features(board)
            return Candidate(piece, rotation, x, y, w, z, layers, network_score(new_heights, holes, layers))
    rough_delta = 0.0
    table = ROUGHNESS_TABLE
    get = changed.get
    for i, value in changed.items():
        old = heights[i]
        for j in ADJACENT[i]:
            target = get(j)
            if target is not None:
                if j < i:
                    continue
            else:
                target = heights[j]
            rough_delta += table[target - value + HEIGHT] - table[heights[j] - old + HEIGHT]
    score = context.score + (HEIGHT_WEIGHT + HOLES_WEIGHT) * delta - PIECE_SIZES[piece] * HOLES_WEIGHT + BUMP_WEIGHT * rough_delta
    return Candidate(piece, rotation, x, y, w, z, 0, score)


def materialize_move(board: Board, candidate: Candidate) -> Move:
    after, layers = clear_layers(merge_rigid_piece(board, placed_cells(candidate), candidate.piece))
    evaluated = evaluate_board(after)
    heights, holes = evaluated.heights, evaluated.holes
    score = (HEIGHT_WEIGHT * evaluated.height_sum + HOLES_WEIGHT * holes
             + LAYERS_WEIGHT * layers + BUMP_WEIGHT * evaluated.roughness)
    if layers != candidate.layers or not math.isclose(score, candidate.score, abs_tol=1e-8):
        raise AssertionError('Incremental and literal evaluations disagree.')
    return Move(Candidate(candidate.piece, candidate.rotation, candidate.x, candidate.y, candidate.w,
                          candidate.z, layers, score), board, after, heights, holes)


def make_move(board: Board, piece: str, rotation: int, x: int, y: int, w: int) -> Move | None:
    if piece not in SHAPES or not 0 <= rotation < len(SHAPES[piece]):
        return None
    candidate = score_candidate(evaluate_board(board), piece, rotation, x, y, w)
    return None if candidate is None else materialize_move(board, candidate)


@lru_cache(maxsize=32)
def placement_count(piece: str) -> int:
    return sum((WIDTH - s.extents[0] + 1) * (DEPTH - s.extents[1] + 1) * (FOURTH - s.extents[3] + 1)
               for s in SHAPES[piece])


def specifications(piece: str, limit: int = DEFAULT_CANDIDATES, salt: int = 0) -> Iterator[tuple[int, int, int, int]]:
    shapes = SHAPES[piece]
    for r, shape in enumerate(shapes):
        nx, ny, nw = WIDTH - shape.extents[0] + 1, DEPTH - shape.extents[1] + 1, FOURTH - shape.extents[3] + 1
        total = nx * ny * nw
        take = total if not limit else min(total, limit // len(shapes) + int(r < limit % len(shapes)))
        if take == total:
            positions = range(total)
        else:
            rng = random.Random((salt + 1) * 1000003 + r * 9176)
            start, stride = rng.randrange(total), rng.randrange(1, total)
            while math.gcd(stride, total) != 1:
                stride = stride % (total - 1) + 1
            positions = ((start + k * stride) % total for k in range(take))
        for index in positions:
            yield r, index % nx, index // nx % ny, index // (nx * ny)


def has_legal_move(board: Board, piece: str) -> bool:
    heights = evaluate_board(board).heights if BACKEND == 'native' else tuple(mask.bit_length() for mask in board)
    for r, x, y, w in specifications(piece, 0):
        shape = SHAPES[piece][r]
        base = column_index(x, y, w)
        if max(heights[base + offset] - bottom for offset, _, bottom in shape.columns) + shape.extents[2] <= HEIGHT:
            return True
    return False


@dataclass(frozen=True, slots=True)
class Decision:
    move: Move
    search_score: float
    evaluations: int
    milliseconds: float
    depth: int
    beam: int
    candidates: int
    sampled: bool
    policy: str = 'network'
    target: int | None = None
    coverage_gain: int = 0
    exhaustive_fallback: bool = False


def candidate_key(candidate: Candidate) -> tuple:
    return round(candidate.score, 9), -candidate.rotation, -candidate.w, -candidate.y, -candidate.x


def configure_backend(name: str = 'auto') -> str:
    global BACKEND, _NUMPY, _NUMPY_TRIED, _NATIVE_INFO
    if name not in ('auto', 'python', 'numpy', 'native'):
        raise ValueError('Backend must be auto, python, numpy, or native.')
    if name != 'python' and not _NUMPY_TRIED:
        _NUMPY_TRIED = True
        try:
            import numpy
            _NUMPY = numpy
        except ImportError:
            _NUMPY = None
    if name in ('numpy', 'native') and _NUMPY is None:
        raise ValueError('NumPy is not installed. Install numpy or use --backend python.')
    selected = 'numpy' if name != 'python' and _NUMPY is not None else 'python'
    if name in ('auto', 'native') and _NUMPY is not None:
        try:
            if _load_native():
                selected = 'native'
            elif name == 'native':
                raise ValueError(_NATIVE_INFO.get('reason', 'Native library unavailable.'))
        except (OSError, ValueError, AttributeError) as error:
            _NATIVE_INFO = {'available': False, 'reason': str(error)}
            if name == 'native':
                raise ValueError(f'Native backend unavailable: {error}') from error
            print(f'Native backend unavailable; using NumPy: {error}', file=sys.stderr)
    if selected != BACKEND:
        for function in (evaluate_board, target_layer_status, pooled_histograms, sigmoid_roughness):
            function.cache_clear()
    BACKEND = selected
    return BACKEND

def native_backend_info() -> dict:
    return dict(_NATIVE_INFO, selected=BACKEND,
                avx2_enabled=_NATIVE_USE_AVX2, bmi2_enabled=_NATIVE_USE_BMI2)


def _load_native() -> bool:
    global _NATIVE, _NATIVE_INFO, _NATIVE_USE_AVX2, _NATIVE_USE_BMI2
    import ctypes
    import hashlib
    import os
    extension = '.dylib' if sys.platform == 'darwin' else '.so'
    library_path = Path(__file__).resolve().parent / 'native' / ('libtetris4d_native' + extension)
    if _NATIVE is None:
        if not library_path.is_file():
            _NATIVE_INFO = {'available': False, 'reason': 'Build with python tools/build_native.py'}
            return False
        source = library_path.with_name('tetris_native.c')
        manifest = json.loads(library_path.with_suffix(extension + '.json').read_text(encoding='utf-8'))
        if (manifest.get('source_sha256') != hashlib.sha256(source.read_bytes()).hexdigest()
                or manifest.get('binary_sha256') != hashlib.sha256(library_path.read_bytes()).hexdigest()):
            raise ValueError('Native library is stale or altered; rebuild with python tools/build_native.py.')
        library = ctypes.CDLL(str(library_path))
        library.t4_abi.argtypes = []
        library.t4_abi.restype = ctypes.c_int
        library.t4_shape_stride.argtypes = []
        library.t4_shape_stride.restype = ctypes.c_int
        if library.t4_abi() != 1 or library.t4_shape_stride() != 182:
            raise ValueError('Incompatible native ABI; rebuild with python tools/build_native.py.')
        pointer, integer, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
        signatures = {
            't4_capabilities': [],
            't4_validate_shapes': [pointer, integer],
            't4_features': [pointer, pointer, pointer, pointer],
            't4_buried': [pointer, integer],
            't4_compact': [pointer, pointer, ctypes.c_uint, integer],
            't4_scores': [pointer, pointer, pointer, pointer, integer, pointer, integer, integer,
                          real, real, real, real, pointer, pointer, pointer, pointer, pointer, integer],
        }
        for name, arguments in signatures.items():
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, integer
        _NATIVE = library
    capabilities = _NATIVE.t4_capabilities()
    requested = os.environ.get('TETRIS_NATIVE_ISA', 'auto').lower()
    if requested not in ('auto', 'scalar', 'avx2'):
        raise ValueError('TETRIS_NATIVE_ISA must be auto, scalar, or avx2.')
    if requested == 'avx2' and not capabilities & 1:
        raise ValueError('AVX2 was requested but is not usable on this CPU/OS.')
    _NATIVE_USE_AVX2 = requested != 'scalar' and bool(capabilities & 1)
    _NATIVE_USE_BMI2 = requested != 'scalar' and bool(capabilities & 2)
    _NATIVE_INFO = {'available': True, 'library': str(library_path), 'abi': 1,
                    'isa': requested, 'avx2_available': bool(capabilities & 1),
                    'bmi2_available': bool(capabilities & 2)}
    return True


def _native_check(result: int) -> None:
    if result:
        raise ValueError(f'Native kernel rejected its input (code {result}).')


def _native_features(board: Board):
    np = _NUMPY
    columns = np.asarray(board, dtype=np.uint16)
    heights = np.empty(AREA, dtype=np.int32)
    counts = np.empty(HEIGHT, dtype=np.int32)
    totals = np.empty(2, dtype=np.int32)
    _native_check(_NATIVE.t4_features(columns.ctypes.data, heights.ctypes.data,
                                      counts.ctypes.data, totals.ctypes.data))
    for array in (columns, heights, counts):
        array.flags.writeable = False
    return columns, heights, counts, totals


def _native_context(context: BoardEvaluation) -> None:
    if context.native_board is None:
        board, heights, counts, _ = _native_features(context.board)
        context.native_board, context.native_heights, context.native_counts = board, heights, counts


@lru_cache(maxsize=16)
def native_shapes(piece: str):
    np = _NUMPY
    shapes = SHAPES[piece]
    records = np.zeros((len(shapes), 182), dtype=np.int32)
    for r, shape in enumerate(shapes):
        row = records[r]
        row[:6] = (*shape.extents, len(shape.columns), PIECE_SIZES[piece])
        for dz, count in shape.levels:
            row[6+dz] = count
        lookup = {offset: i for i, (offset, _, _) in enumerate(shape.columns)}
        for i, (offset, mask, bottom) in enumerate(shape.columns):
            start = 22 + i*10
            row[start:start+4] = (offset, mask, bottom, mask.bit_length())
            row[start+4:start+10] = -1
            coordinates = column_coordinates(offset)
            for direction, step in enumerate((-1, 1, -WIDTH, WIDTH, -SLICE_AREA, SLICE_AREA)):
                coord = coordinates[direction // 2]
                if 0 <= coord + (-1 if direction % 2 == 0 else 1) < 16:
                    row[start+4+direction] = lookup.get(offset+step, -1)
    _native_check(_NATIVE.t4_validate_shapes(records.ctypes.data, len(shapes)))
    records.flags.writeable = False
    return records


@lru_cache(maxsize=2)
def native_roughness_deltas(table: tuple[float, ...]):
    if any(table[HEIGHT+d] != table[HEIGHT-d] for d in range(HEIGHT+1)):
        raise ValueError('Native roughness requires a symmetric sigmoid response.')
    positive = _NUMPY.asarray(table[HEIGHT:], dtype=_NUMPY.float64)
    result = positive[:, None] - positive[None, :]
    result.flags.writeable = False
    return result


def native_scores(context: BoardEvaluation, piece: str, specs, target: int = -1):
    np = _NUMPY
    if type(target) is not int or not -1 <= target < HEIGHT:
        raise ValueError('Native target must be an integer in [-1, 15].')
    raw = np.asarray(specs)
    if raw.ndim != 2 or raw.shape[1] != 4 or not 1 <= len(raw) <= BATCH_SIZE or raw.dtype.kind not in 'iu':
        raise ValueError('Native scoring requires 1..512 four-integer placement specifications.')
    if raw.dtype != np.int32 and (np.any(raw < 0) or np.any(raw > (1 << 31) - 1)):
        raise ValueError('Native placement indices exceed their valid integer range.')
    raw = np.ascontiguousarray(raw, dtype=np.int32)
    _native_context(context)
    records = native_shapes(piece)
    table = native_roughness_deltas(ROUGHNESS_TABLE)
    scores = np.empty(len(raw), dtype=np.float64)
    z, gains = np.empty(len(raw), dtype=np.int32), np.empty(len(raw), dtype=np.int32)
    clear_flags = np.empty(len(raw), dtype=np.uint8)
    _native_check(_NATIVE.t4_scores(context.native_board.ctypes.data, context.native_heights.ctypes.data,
        context.native_counts.ctypes.data, records.ctypes.data, len(records), raw.ctypes.data,
        len(raw), target, context.score, HEIGHT_WEIGHT, HOLES_WEIGHT, BUMP_WEIGHT,
        table.ctypes.data, scores.ctypes.data, z.ctypes.data, gains.ctypes.data,
        clear_flags.ctypes.data, int(_NATIVE_USE_AVX2)))
    clears = {}
    for row in np.flatnonzero(clear_flags):
        candidate = score_candidate(context, piece, *map(int, raw[row]))
        if candidate is None:
            raise AssertionError('Native clear candidate must be legal.')
        clears[int(row)] = candidate
        scores[row] = candidate.score
    return scores, z, clears, gains


@lru_cache(maxsize=1)
def numpy_tables():
    np = _NUMPY
    a, b = np.asarray(NEIGHBORS, dtype=np.int32).T
    return np, a, b, np.asarray(ROUGHNESS_TABLE)


@lru_cache(maxsize=1)
def numpy_network_data():
    return _NUMPY.asarray(SIGMOID_TABLE)


@lru_cache(maxsize=16)
def numpy_shape_data(piece: str):
    np = _NUMPY
    shapes = SHAPES[piece]
    k = max(len(shape.columns) for shape in shapes)
    offsets = np.zeros((len(shapes), k), dtype=np.int32)
    bottoms = np.full((len(shapes), k), HEIGHT * 4, dtype=np.int16)
    tops = np.zeros((len(shapes), k), dtype=np.int16)
    valid = np.zeros((len(shapes), k), dtype=bool)
    partners = np.full((len(shapes), k, 6), -1, dtype=np.int16)
    steps = (-1, 1, -WIDTH, WIDTH, -SLICE_AREA, SLICE_AREA)
    for r, shape in enumerate(shapes):
        lookup = {offset: j for j, (offset, _, _) in enumerate(shape.columns)}
        for i, (offset, mask, bottom) in enumerate(shape.columns):
            offsets[r, i], bottoms[r, i], tops[r, i], valid[r, i] = offset, bottom, mask.bit_length(), True
            cx, cy, cw = column_coordinates(offset)
            for j, step in enumerate(steps):
                coordinates = (cx, cy, cw)
                axis, sign = j // 2, -1 if j % 2 == 0 else 1
                if 0 <= coordinates[axis] + sign < 16:
                    partners[r, i, j] = lookup.get(offset + step, -1)
    dimensions = np.asarray([shape.extents for shape in shapes], dtype=np.int16)
    return offsets, bottoms, tops, valid, partners, dimensions


class PlacementBatch:
    __slots__ = ('piece', 'specs', 'data', 'native')

    def __init__(self, piece: str, specs: tuple[tuple[int, int, int, int], ...]) -> None:
        self.piece, self.specs, self.data = piece, specs, None
        self.native = BACKEND == 'native' and bool(specs)
        if self.native:
            self.data = _NUMPY.asarray(specs, dtype=_NUMPY.int32)
            return
        if BACKEND not in ('numpy', 'native') or not specs:
            return
        np = _NUMPY
        offsets, bottoms, tops, valid, partners, dimensions = numpy_shape_data(piece)
        raw = np.asarray(specs, dtype=np.int32)
        r, x, y, w = raw.T
        indices = ((w * DEPTH + y) * WIDTH + x)[:, None] + offsets[r]
        present = valid[r]
        source = indices[:, :, None]
        steps = np.asarray((-1, 1, -WIDTH, WIDTH, -SLICE_AREA, SLICE_AREA), dtype=np.int32)
        neighbors = source + steps
        coordinates = (indices % WIDTH, indices // WIDTH % DEPTH, indices // SLICE_AREA)
        edge_valid = np.stack(tuple(test for c in coordinates for test in (c > 0, c < 15)), axis=2)
        local = partners[r]
        local_safe = np.maximum(local, 0)
        edge_valid &= present[:, :, None]
        edge_valid &= (local < 0) | (neighbors > source)
        neighbors = np.clip(neighbors, 0, AREA - 1)
        self.data = (indices, bottoms[r], tops[r], present, local, local_safe, neighbors,
                     edge_valid, dimensions[r, 2], np.arange(len(specs))[:, None, None])

    def scores(self, context: BoardEvaluation):
        if self.native:
            scores, z, clears, _ = native_scores(context, self.piece, self.data)
            return scores, z, clears
        np = _NUMPY
        indices, bottoms, tops, present, local, local_safe, neighbors, edge_valid, heights4d, rows = self.data
        if context.array_heights is None:
            context.array_heights = np.asarray(context.heights, dtype=np.int16)
        heights = context.array_heights
        old = heights[indices]
        z = np.max(old - bottoms, axis=1)
        legal = z + heights4d <= HEIGHT
        new = z[:, None] + tops
        delta = np.sum(np.where(present, new - old, 0), axis=1)
        neighbor_old = heights[neighbors]
        neighbor_new = np.where(local >= 0, new[rows, local_safe], neighbor_old)
        table = numpy_tables()[3]
        before = table[np.clip(neighbor_old - old[:, :, None] + HEIGHT, 0, 2 * HEIGHT)]
        after = table[np.clip(neighbor_new - new[:, :, None] + HEIGHT, 0, 2 * HEIGHT)]
        terms = np.where(edge_valid, after - before, 0.0).reshape(len(self.specs), -1)
        rough = np.cumsum(terms, axis=1)[:, -1]
        scores = (context.score + (HEIGHT_WEIGHT + HOLES_WEIGHT) * delta
                  - PIECE_SIZES[self.piece] * HOLES_WEIGHT + BUMP_WEIGHT * rough)
        scores[~legal] = -math.inf
        clears = {}
        if context.near_clear:
            for i in np.flatnonzero(legal):
                r, x, y, w = self.specs[i]
                if any(context.fill_counts[int(z[i]) + dz] + count == AREA
                       for dz, count in SHAPES[self.piece][r].levels):
                    candidate = score_candidate(context, self.piece, r, x, y, w)
                    scores[i] = candidate.score
                    clears[int(i)] = candidate
        return scores, z, clears

    def evaluate(self, context: BoardEvaluation) -> tuple[list[Candidate | None], int]:
        if self.data is None:
            return [score_candidate(context, self.piece, *spec) for spec in self.specs], len(self.specs)
        scores, z, clears = self.scores(context)
        result = [clears.get(i) or Candidate(self.piece, *spec, int(z[i]), 0, value)
                  if math.isfinite(value) else None for i, (spec, value) in enumerate(zip(self.specs, scores.tolist()))]
        return result, len(self.specs)

    def maximum(self, context: BoardEvaluation) -> tuple[float, int]:
        if self.data is not None:
            scores, _, _ = self.scores(context)
            return float(scores.max()), len(self.specs)
        best = -math.inf
        for spec in self.specs:
            candidate = score_candidate(context, self.piece, *spec)
            if candidate is not None and candidate.score > best:
                best = candidate.score
        return best, len(self.specs)


class PlacementPlan:
    __slots__ = ('piece', 'limit', 'salt', 'batches', 'complete')

    def __init__(self, piece: str, limit: int, salt: int) -> None:
        self.piece, self.limit, self.salt, self.batches = piece, limit, salt, []
        self.complete = False

    def __iter__(self):
        if self.complete:
            yield from self.batches
            return
        self.batches.clear()
        iterator = iter(specifications(self.piece, self.limit, self.salt))
        while specs := tuple(itertools.islice(iterator, BATCH_SIZE)):
            batch = PlacementBatch(self.piece, specs)
            if 0 < self.limit <= 16384:
                self.batches.append(batch)
            yield batch
        self.complete = 0 < self.limit <= 16384


def decision_steps(board: Board, piece: str, next_piece: str | None = None, depth: int = 2,
                   beam: int = DEFAULT_BEAM, candidates: int = DEFAULT_CANDIDATES,
                   salt: int = 0) -> Generator[int, None, Decision | None]:
    if depth not in (1, 2) or beam < 1 or candidates < 0:
        raise ValueError('Depth must be 1 or 2; beam positive; candidates nonnegative.')
    if piece not in SHAPES or (depth == 2 and next_piece not in SHAPES):
        raise ValueError('Unknown current or preview piece.')
    if BACKEND == 'auto':
        configure_backend()
    started = time.perf_counter()
    context, first, evaluations = evaluate_board(board), [], 0
    for batch in PlacementPlan(piece, candidates, salt):
        results, count = batch.evaluate(context)
        evaluations += count
        first.extend(candidate for candidate in results if candidate is not None)
        yield evaluations
    if not first and candidates:
        for spec in specifications(piece, 0):
            candidate = score_candidate(context, piece, *spec)
            evaluations += 1
            if candidate:
                first.append(candidate)
                break
            if evaluations % 128 == 0:
                yield evaluations
    if not first:
        return None
    first.sort(key=candidate_key, reverse=True)
    sampled = bool(candidates and candidates < placement_count(piece))
    best_move, best_key, best_score = None, None, -math.inf
    if depth == 1:
        best_move = materialize_move(board, first[0])
        best_score = best_move.score
    else:
        future_plan = PlacementPlan(next_piece, candidates, salt + 65537)
        for i, candidate in enumerate(first):
            if i >= beam and math.isfinite(best_score):
                break
            move = materialize_move(board, candidate)
            future_context, future = evaluate_board(move.board), -math.inf
            for batch in future_plan:
                value, count = batch.maximum(future_context)
                evaluations += count
                if value > future:
                    future = value
                yield evaluations
            if not math.isfinite(future) and candidates:
                for spec in specifications(next_piece, 0):
                    following = score_candidate(future_context, next_piece, *spec)
                    evaluations += 1
                    if following:
                        future = following.score
                        break
                    if evaluations % 128 == 0:
                        yield evaluations
            value = LAYERS_WEIGHT * candidate.layers + future
            key = (round(value, 9), candidate_key(candidate))
            if best_key is None or key > best_key:
                best_move, best_key, best_score = move, key, value
            sampled |= bool(candidates and candidates < placement_count(next_piece))
    return Decision(best_move, best_score, evaluations, (time.perf_counter() - started) * 1000,
                    depth, beam, candidates, sampled)


def choose_move(*args, **kwargs) -> Decision | None:
    steps = decision_steps(*args, **kwargs)
    while True:
        try:
            next(steps)
        except StopIteration as result:
            return result.value


@dataclass(frozen=True, slots=True)
class TargetLayerStatus:
    target: int
    filled: int
    buried: int


@lru_cache(maxsize=16, typed=True)
def target_layer_status(board: Board, target: int = 1) -> TargetLayerStatus:
    if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < HEIGHT:
        raise ValueError(f'Target Z must be an integer in [0, {HEIGHT - 1}].')
    if BACKEND == 'native':
        context = evaluate_board(board)
        _native_context(context)
        buried = _NATIVE.t4_buried(context.native_board.ctypes.data, target)
        if buried < 0:
            _native_check(buried)
        return TargetLayerStatus(target, context.fill_counts[target], buried)
    if len(board) != AREA or any(not isinstance(mask, int) or not 0 <= mask < (1 << HEIGHT) for mask in board):
        raise ValueError('Expected 4096 valid vertical column masks.')
    bit = 1 << target
    return TargetLayerStatus(target, sum(bool(mask & bit) for mask in board),
                             sum(not (mask & bit) and bool(mask >> (target + 1)) for mask in board))


def next_hyperlayer_target(board: Board, preferred: int = 1) -> int | None:
    target_layer_status(board, preferred)
    for target in sorted(range(HEIGHT), key=lambda level: (abs(level - preferred), level)):
        status = target_layer_status(board, target)
        if not status.buried and status.filled < AREA:
            return target
    return None


def target_candidate_metrics(board: Board, candidate: Candidate, target: int) -> tuple[int, int]:
    bit, gain, buried = 1 << target, 0, 0
    base = column_index(candidate.x, candidate.y, candidate.w)
    for offset, mask, _ in SHAPES[candidate.piece][candidate.rotation].columns:
        old = board[base + offset]
        added = mask << candidate.z
        if not old & bit:
            gain += bool(added & bit)
            buried += bool((old | added) >> (target + 1)) and not bool(added & bit)
    return gain, buried


@lru_cache(maxsize=16)
def numpy_column_masks(piece: str):
    masks = _NUMPY.zeros_like(numpy_shape_data(piece)[0])
    for r, shape in enumerate(SHAPES[piece]):
        for i, (_, mask, _) in enumerate(shape.columns):
            masks[r, i] = mask
    return masks


def hyperlayer_decision_steps(board: Board, piece: str, target: int = 1,
                              candidates: int = DEFAULT_CANDIDATES,
                              salt: int = 0) -> Generator[int, None, Decision | None]:
    if piece not in SHAPES or not isinstance(candidates, int) or candidates < 0:
        raise ValueError('Unknown piece or invalid candidate budget.')
    status = target_layer_status(board, target)
    if status.buried or status.filled == AREA:
        return None
    if BACKEND == 'auto':
        configure_backend()
    started = time.perf_counter()
    context, evaluations, best, best_key = evaluate_board(board), 0, None, None
    missing = (i for i, mask in enumerate(board) if not mask & (1 << target))
    vertical = tuple(r for r, shape in enumerate(SHAPES[piece]) if len(shape.columns) == 1) if piece == 'I' else ()
    anchored = ((r, *column_coordinates(i)) for i in missing for r in vertical)
    initial = itertools.chain(specifications(piece, candidates, salt), anchored)
    fallback = False
    for iterator in (initial, specifications(piece, 0)):
        if fallback and (best is not None or not candidates):
            break
        while specs := tuple(itertools.islice(iterator, BATCH_SIZE)):
            batch = PlacementBatch(piece, specs)
            if batch.native:
                np = _NUMPY
                scores, z, clears, gains = native_scores(context, piece, batch.data, target)
                valid = np.flatnonzero(np.isfinite(scores))
                if len(valid):
                    index = int(valid[np.argmax(scores[valid])])
                    key = (int(gains[index]), float(scores[index]))
                    if best_key is None or key > best_key:
                        best_key = key
                        best = clears.get(index) or Candidate(piece, *specs[index], int(z[index]), 0, key[1])
            elif batch.data is not None:
                np = _NUMPY
                scores, z, clears = batch.scores(context)
                indices, _, tops, present, *_ = batch.data
                old = context.array_heights[indices]
                rotations = np.asarray([spec[0] for spec in specs], dtype=np.int32)
                relative = target - z
                on_target = ((numpy_column_masks(piece)[rotations] >> np.clip(relative, 0, HEIGHT)[:, None]) & 1).astype(bool)
                on_target &= (relative >= 0)[:, None] & (relative < HEIGHT)[:, None] & present
                gap = old <= target
                gains = np.sum(gap & on_target, axis=1)
                blocked = np.any(gap & (z[:, None] + tops > target) & ~on_target & present, axis=1)
                valid = np.flatnonzero(np.isfinite(scores) & ~blocked)
                if len(valid):
                    high = int(gains[valid].max())
                    valid = valid[gains[valid] == high]
                    index = int(valid[np.argmax(scores[valid])])
                    key = (high, float(scores[index]))
                    if best_key is None or key > best_key:
                        best_key = key
                        best = clears.get(index) or Candidate(piece, *specs[index], int(z[index]), 0, key[1])
            else:
                for spec in specs:
                    candidate = score_candidate(context, piece, *spec)
                    if candidate is None:
                        continue
                    gain, buried = target_candidate_metrics(board, candidate, target)
                    key = (gain, candidate.score)
                    if not buried and (best_key is None or key > best_key):
                        best_key, best = key, candidate
            evaluations += len(specs)
            yield evaluations
            if fallback and best is not None:
                break
        if best is not None:
            move = materialize_move(board, best)
            return Decision(move, move.score, evaluations, (time.perf_counter() - started) * 1000,
                            1, 1, candidates, bool(candidates and candidates < placement_count(piece)),
                            'hyperlayer', target, best_key[0], fallback)
        fallback = True
    return None


def choose_hyperlayer_move(*args, **kwargs) -> Decision | None:
    steps = hyperlayer_decision_steps(*args, **kwargs)
    while True:
        try:
            next(steps)
        except StopIteration as result:
            return result.value


class PieceStream:
    def __init__(self, seed: int = 7, piece_set: str = 'expanded') -> None:
        if piece_set not in PIECE_SETS:
            raise ValueError(f'Unknown piece set: {piece_set}')
        self.names = PIECE_SETS[piece_set]
        self.rng, self.bag = random.Random(seed), []

    def pop(self) -> str:
        if not self.bag:
            self.bag = list(self.names)
            self.rng.shuffle(self.bag)
        return self.bag.pop()


class Game:
    def __init__(self, seed: int = 7, piece_set: str = 'expanded') -> None:
        if piece_set not in PIECE_SETS:
            raise ValueError(f'Unknown piece set: {piece_set}')
        self.seed, self.piece_set = seed, piece_set
        self.reset()

    def reset(self) -> None:
        self.stream = PieceStream(self.seed, self.piece_set)
        self.board = EMPTY_BOARD
        self.current, self.next_piece = self.stream.pop(), self.stream.pop()
        self.pieces = self.layers = self.score = self.cells_placed = 0
        self.piece_counts = {name: 0 for name in PIECE_NAMES}
        self.game_over = False
        self.locked_pieces: tuple[LockedPiece, ...] = ()
        self.last_move: Move | None = None

    def commit(self, move: Move) -> None:
        if self.game_over or move.before != self.board or move.piece != self.current:
            raise ValueError('This move is stale or does not belong to the active piece.')
        p = move.candidate
        if not 0 <= p.rotation < len(SHAPES[p.piece]):
            raise ValueError('Invalid rotation.')
        shape = SHAPES[p.piece][p.rotation]
        sx, sy, sz, sw = shape.extents
        if not (0 <= p.x <= WIDTH - sx and 0 <= p.y <= DEPTH - sy and 0 <= p.w <= FOURTH - sw):
            raise ValueError('Invalid placement origin.')
        base = column_index(p.x, p.y, p.w)
        landing = max(self.board[base + offset].bit_length() - bottom for offset, _, bottom in shape.columns)
        if p.z != landing or p.z + sz > HEIGHT:
            raise ValueError('The entire rigid piece must land on the stack or floor.')
        solid = merge_rigid_piece(self.board, placed_cells(move), move.piece)
        compacted, layers = clear_layers(solid)
        if compacted != move.board or layers != move.layers:
            raise ValueError('Invalid rigid-piece placement or hyperlayer compaction.')
        cleared = full_layer_mask(solid)
        locked = LockedPiece(self.pieces + 1, move.piece, placed_cells(move))
        self.locked_pieces = compact_locked_pieces(self.locked_pieces + (locked,), cleared)
        self.board = move.board
        self.pieces += 1
        self.cells_placed += PIECE_SIZES[move.piece]
        self.piece_counts[move.piece] += 1
        self.layers += move.layers
        self.score += clear_points(move.layers)
        self.last_move = move
        self.current, self.next_piece = self.next_piece, self.stream.pop()
        self.game_over = not has_legal_move(self.board, self.current)

    def summary(self) -> dict:
        heights, holes = board_features(self.board)
        return {'seed': self.seed, 'piece_set': self.piece_set, 'pieces': self.pieces,
                'cells_placed': self.cells_placed, 'piece_counts': dict(self.piece_counts),
                'hyperlayers': self.layers, 'score': self.score,
                'game_over': self.game_over, 'max_height': max(heights), 'holes': holes,
                'occupied_cells': sum(mask.bit_count() for mask in self.board)}


PALETTE = {"bg": "#101721", "panel": "#182331", "board": "#080F19", "text": "#F4F0E6",
           "muted": "#9EAEBD", "grid": "#223346", "accent": "#67D8E5", "locked": "#647D9A"}
PIECE_COLORS = {"I": "#65D9E8", "O": "#E5C467", "T": "#AF8CE0", "L": "#E5AA70",
                "S": "#7ACD9A", "B": "#E78286", "C": "#719FE4", "D": "#D69FCE",
                "NOVA": "#F3D56B", "STAIR": "#8F9EFF", "FORK": "#F4A1CE", "HOOK": "#F49367",
                "BRIDGE": "#71CDAC", "RING": "#C3A0F3", "CROSS": "#F1738A", "HYPER": "#62D7ED"}


@lru_cache(maxsize=4096)
def blend(start: str, end: str, amount: float) -> str:
    amount = min(1.0, max(0.0, amount))
    return "#" + "".join(f"{round(int(start[i:i + 2], 16) * (1 - amount) + int(end[i:i + 2], 16) * amount):02x}" for i in (1, 3, 5))


FACE_GEOMETRY = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1))),
    ((0, 1, 0), ((1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0))),
)
CUBE_EDGES = tuple((a, tuple(a[i] + (1 if i == axis else 0) for i in range(3)))
                   for a in ((x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1))
                   for axis in range(3) if a[axis] == 0)


class WellView:
    def __init__(self, canvas, compact: bool = False) -> None:
        self.canvas = canvas
        self.compact = compact
        self.section_revision = 0
        self.scene_rebuilds = 0
        self.draw_skips = 0
        self.yaw, self.elevation, self.zoom = 0.68, 0.57, 1.0
        self.drag_origin = None
        self.last_key = None
        self.last_args = None
        self.piece_cache = None
        self.owners = {}
        self.rendered_cubes = ()
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.canvas.configure(cursor=""))
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom(1.08))
        self.canvas.bind("<Button-5>", lambda event: self._zoom(1 / 1.08))
        self.canvas.bind("<Configure>", lambda event: self.redraw())

    def reset_camera(self) -> None:
        self.yaw, self.elevation, self.zoom = 0.68, 0.57, 1.0
        self.redraw()

    def _press(self, event) -> None:
        self.drag_origin = (event.x, event.y, self.yaw, self.elevation)
        self.canvas.configure(cursor="fleur")

    def _drag(self, event) -> None:
        if self.drag_origin:
            x, y, yaw, elevation = self.drag_origin
            self.yaw = yaw + (event.x - x) * 0.009
            self.elevation = min(1.40, max(0.18, elevation + (event.y - y) * 0.007))
            self.redraw()

    def _wheel(self, event) -> str:
        if event.delta:
            self._zoom(1.08 if event.delta > 0 else 1 / 1.08)
        return "break"

    def _zoom(self, multiplier: float) -> None:
        self.zoom = min(1.0 if self.compact else 1.65, max(0.65, self.zoom * multiplier))
        self.redraw()

    def redraw(self) -> None:
        if self.last_args:
            self.draw(*self.last_args)

    def _camera(self, w: int, h: int) -> None:
        ca, sa, ce, se = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.elevation), math.sin(self.elevation)
        self.right = (-sa, ca, 0.0)
        self.up = (-ca * se, -sa * se, ce)
        self.toward = (ca * ce, sa * ce, se)
        corners = [self._raw(x, y, z) for x in (0, WIDTH) for y in (0, DEPTH) for z in (0, HEIGHT)]
        low_x, high_x = min(p[0] for p in corners), max(p[0] for p in corners)
        low_y, high_y = min(p[1] for p in corners), max(p[1] for p in corners)
        margin_x, margin_y = (12, 32) if self.compact else (76, 86)
        self.scale = self.zoom * min((w - margin_x) / (high_x - low_x), (h - margin_y) / (high_y - low_y))
        self.scale = max(.1 if self.compact else 1.0, self.scale)
        self.ox = w / 2 - (low_x + high_x) * self.scale / 2
        self.oy = h / 2 + (0 if self.compact else 5) - (low_y + high_y) * self.scale / 2

    def _raw(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        return (x * self.right[0] + y * self.right[1],
                -(x * self.up[0] + y * self.up[1] + z * self.up[2]),
                x * self.toward[0] + y * self.toward[1] + z * self.toward[2])

    def project(self, x: float, y: float, z: float) -> tuple[float, float]:
        px, py, _ = self._raw(x, y, z)
        return self.ox + self.scale * px, self.oy + self.scale * py

    def _line(self, a, b, **options) -> None:
        self.canvas.create_line(*self.project(*a), *self.project(*b), **options)

    def _faces(self, cubes, occupied):
        visible = [(normal, vertices) for normal, vertices in FACE_GEOMETRY
                   if sum(a * b for a, b in zip(normal, self.toward)) > 1e-9]
        faces = []
        for x, y, z, color, owner in cubes:
            for normal, vertices in visible:
                adjacent = (x + normal[0], y + normal[1], round(z + normal[2], 9))
                if adjacent in occupied:
                    continue
                points = [(x + dx, y + dy, z + dz) for dx, dy, dz in vertices]
                depth = sum(self._raw(*point)[2] for point in points) / 4
                shade = blend(color, '#FFFFFF', .18) if normal[2] else blend(color, '#000000', .18 if normal[0] else .35)
                borders = []
                for i, a in enumerate(vertices):
                    b = vertices[(i + 1) % 4]
                    step = tuple(a[j] + b[j] - 1 if not normal[j] else 0 for j in range(3))
                    beside = (x + step[0], y + step[1], round(z + step[2], 9))
                    beyond = (beside[0] + normal[0], beside[1] + normal[1], round(beside[2] + normal[2], 9))
                    if not owner or occupied.get(beside) != owner or beyond in occupied:
                        borders.append((points[i], points[(i + 1) % 4]))
                faces.append((depth, points, shade, borders, owner))
        faces.sort(key=lambda face: face[0])
        return faces

    def _face(self, face, tag):
        _, points, color, borders, owner = face
        first = self.canvas.create_polygon(*[v for point in points for v in self.project(*point)], fill=color,
                             outline=blend(color, '#162536', .3), width=.5,
                             tags=(tag, 'voxel_face', 'active_piece' if owner == -1 else 'locked_piece',
                                   'active_dynamic' if owner == -1 else 'static_mesh'))
        for a, b in borders:
            self._line(a, b, fill='#162536', width=1.8,
                       tags=(tag, 'active_dynamic' if owner == -1 else 'static_mesh'))
        return first

    def draw(self, game: Game, move: Move | None, progress: float = 0.0,
             w_slice: int = 0, cutoff: int = HEIGHT - 1) -> None:
        self.last_args = (game, move, progress, w_slice, cutoff)
        c = self.canvas
        width, height = ((max(1, c.winfo_width()), max(1, c.winfo_height())) if self.compact else
                         (max(280, c.winfo_width()), max(250, c.winfo_height())))
        if width < 28 or height < 40:
            self.last_key = None
            return
        if game.locked_pieces is not self.piece_cache:
            self.piece_cache = game.locked_pieces
            self.owners = {cell: (piece.identifier, piece.piece) for piece in game.locked_pieces for cell in piece.cells}
            self.section_revision += 1
        if game.board is not getattr(self, '_board_identity', None) or w_slice != getattr(self, '_board_slice', None):
            self._board_identity, self._board_slice = game.board, w_slice
            self._slice_masks = game.board[w_slice * SLICE_AREA:(w_slice + 1) * SLICE_AREA]
        stack_key = (self._slice_masks, self.section_revision, w_slice, cutoff)
        key = (stack_key, move, round(progress, 3), width, height, self.yaw, self.elevation, self.zoom)
        if key == self.last_key:
            self.draw_skips += 1
            return
        self.last_key = key
        self._camera(width, height)
        if stack_key != getattr(self, '_stack_key', None):
            self._stack_key = stack_key
            stack = []
            mask_limit = (1 << (cutoff + 1)) - 1
            for i in range(w_slice * SLICE_AREA, (w_slice + 1) * SLICE_AREA):
                x, y, _ = column_coordinates(i)
                mask = game.board[i] & mask_limit
                while mask:
                    bit = mask & -mask
                    z = bit.bit_length() - 1
                    owner, name = self.owners.get((x, y, z, w_slice), (0, ''))
                    stack.append((x, y, z, PIECE_COLORS.get(name, PALETTE['locked']), owner))
                    mask ^= bit
            self._stack = tuple(stack)
            self._occupied = {(x, y, z): owner for x, y, z, _, owner in stack}
        active = () if move is None else animated_piece_cells(move, progress)
        moving = tuple((x, y, z, PIECE_COLORS[move.piece], -1) for x, y, z, w in active if w == w_slice)
        self.rendered_cubes = self._stack + moving
        occupied = self._occupied.copy()
        occupied.update(((x, y, round(z, 9)), -1) for x, y, z, _, _ in moving)
        grid_contacts = tuple((x, y, round(z, 9)) for x, y, z, _, _ in moving if round(z, 9).is_integer())
        scene_key = (stack_key, grid_contacts, width, height, self.yaw, self.elevation, self.zoom)
        if scene_key != getattr(self, '_scene_key', None):
            self._scene_key = scene_key
            self.scene_rebuilds += 1
            self._overlay_key = None
            c.delete('all')
            c.create_text(6 if self.compact else 14, 10 if self.compact else 16, anchor='w',
                          text=f'W {w_slice:02d}' if self.compact else f'3D SECTION  /  W = {w_slice:02d} of 00–15',
                          fill=PALETTE['text'], font=('TkDefaultFont', 8 if self.compact else 10, 'bold'))
            if not self.compact:
                c.create_text(14, 36, anchor='w', text='X/Y horizontal · Z gravity · W fixed' if width < 400 else 'X / Y horizontal · Z gravity · other W sections remain separate',
                              fill=PALETTE['muted'], font=('TkDefaultFont', 8))
            floor = [self.project(x, y, 0) for x, y in ((0, 0), (WIDTH, 0), (WIDTH, DEPTH), (0, DEPTH))]
            c.create_polygon(*[v for point in floor for v in point], fill='#142334', outline=PALETTE['grid'])
            for x in range(0, WIDTH + 1, 4 if self.compact else 1):
                self._line((x, 0, 0), (x, DEPTH, 0), fill='#36526B')
            for y in range(0, DEPTH + 1, 4 if self.compact else 1):
                self._line((0, y, 0), (WIDTH, y, 0), fill='#36526B')
            bx, by = (0 if self.toward[0] > 0 else WIDTH), (0 if self.toward[1] > 0 else DEPTH)
            for z in range(0, HEIGHT + 1, 4):
                self._line((bx, 0, z), (bx, DEPTH, z), fill=PALETTE['grid'], dash=(2, 5))
                self._line((0, by, z), (WIDTH, by, z), fill=PALETTE['grid'], dash=(2, 5))
            for x in (0, WIDTH):
                for y in (0, DEPTH):
                    self._line((x, y, 0), (x, y, HEIGHT), fill='#58738B', dash=(3, 5))
            static_faces = self._faces(self._stack, occupied)
            self._static_depths, self._static_items = [], []
            for i, face in enumerate(static_faces):
                self._static_depths.append(face[0])
                self._static_items.append(self._face(face, f'static_face:{i}'))
        c.delete('active_dynamic')
        for i, face in enumerate(self._faces(moving, occupied)):
            tag = f'active_face:{i}'
            self._face(face, tag)
            position = bisect.bisect_right(self._static_depths, face[0])
            if position < len(self._static_items):
                c.tag_lower(tag, self._static_items[position])
        overlay_key = (id(move), w_slice, cutoff, width, height, self.yaw, self.elevation, self.zoom)
        if overlay_key != getattr(self, '_overlay_key', None):
            self._overlay_key = overlay_key
            c.delete('overlay')
            if move:
                for x, y, z, w in placed_cells(move):
                    if w == w_slice:
                        for a, b in CUBE_EDGES:
                            self._line((x + a[0], y + a[1], z + a[2]), (x + b[0], y + b[1], z + b[2]),
                                       fill=PIECE_COLORS[move.piece], dash=(3, 3), width=1.3, tags=('landing_ghost', 'overlay'))
            for endpoint, label, color in (((WIDTH + .5, 0, 0), '+X', '#E5AA70'), ((0, DEPTH + .5, 0), '+Y', '#7ACD9A')):
                self._line((0, 0, 0), endpoint, fill=color, arrow='last', width=2, tags=('overlay',))
                px, py = self.project(*endpoint)
                if not self.compact:
                    c.create_text(px, py + 10, text=label, fill=color, font=('TkFixedFont', 8, 'bold'), tags=('overlay',))
            if cutoff < HEIGHT - 1:
                c.create_text(width - 5 if self.compact else 14, 10 if self.compact else 55,
                              anchor='e' if self.compact else 'w',
                              text=f'Z≤{cutoff}' if self.compact else f'Stack cutaway Z ≤ {cutoff}; active section is not clipped',
                              fill='#E5C467', font=('TkDefaultFont', 8), tags=('overlay',))
            count, total = len(moving), 0 if move is None else PIECE_SIZES[move.piece]
            if self.compact:
                c.create_text(6, height - 9, anchor='w', text=f'Active {count}',
                              fill=PALETTE['accent'], font=('TkDefaultFont', 8), tags=('overlay',))
            else:
                c.create_text(14, height - 35, anchor='w', text=f'Active section: {count}/{total} · whole piece below' if width < 400 else f'Active cells in this section: {count}/{total}  ·  full piece shown below',
                              fill=PALETTE['accent'], font=('TkDefaultFont', 9), tags=('overlay',))
                c.create_text(14, height - 16, anchor='w', text='Drag: orbit · wheel: zoom · dashed: landing' if width < 400 else 'Drag to orbit · wheel to zoom · dashed cells = landing',
                              fill=PALETTE['muted'], font=('TkDefaultFont', 8), tags=('overlay',))
        c.tag_raise('overlay')


class AllWellView:
    def __init__(self, parent, select_slice) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.frame = ttk.Frame(parent, style='Tetris.TFrame')
        self.frame.grid(row=0, column=0, sticky='nsew')
        self.frame.rowconfigure(1, weight=1)
        self.frame.columnconfigure(0, weight=1)
        self.select_slice = select_slice
        self.last_args = None
        self.piece_cache = None
        self.owners = {}
        self.visible_w_sections = tuple(range(FOURTH))
        self.section_tokens = [None] * FOURTH
        self.section_revisions = [0] * FOURTH
        self.selection_label = tk.StringVar(value='All 16 sections · selected W 00')
        ttk.Label(self.frame, textvariable=self.selection_label, style='Muted.Tetris.TLabel').grid(row=0, column=0, sticky='w')
        self.grid = ttk.Frame(self.frame, style='Tetris.TFrame')
        self.grid.grid(row=1, column=0, sticky='nsew')
        self.views = []
        for w in self.visible_w_sections:
            canvas = tk.Canvas(self.grid, width=100, height=80, bg=PALETTE['board'],
                               highlightthickness=2, highlightbackground=PALETTE['grid'])
            view = WellView(canvas, compact=True)
            self.views.append(view)
            canvas.bind('<ButtonPress-1>', lambda event, index=w: self._select(index), add='+')
            canvas.grid(row=w // 4, column=w % 4, sticky='nsew', padx=2, pady=2)
        for row in range(4):
            self.grid.rowconfigure(row, weight=1, uniform='well_rows')
        for column in range(4):
            self.grid.columnconfigure(column, weight=1, uniform='well_columns')

    def _select(self, index: int) -> None:
        self.select_slice(index)

    def reset_camera(self) -> None:
        for view in self.views:
            view.reset_camera()

    def draw(self, game: Game, move: Move | None, progress: float = 0.0,
             w_slice: int = 0, cutoff: int = HEIGHT - 1) -> None:
        self.last_args = (game, move, progress, w_slice, cutoff)
        label = f'All 16 sections · selected W {w_slice:02d}'
        if self.selection_label.get() != label:
            self.selection_label.set(label)
        if game.locked_pieces is not self.piece_cache:
            self.piece_cache = game.locked_pieces
            self.owners = {}
            tokens = [[] for _ in range(FOURTH)]
            for piece in game.locked_pieces:
                for cell in piece.cells:
                    identity = (piece.identifier, piece.piece)
                    self.owners[cell] = identity
                    tokens[cell[3]].append((cell, identity))
            for w, cells in enumerate(tokens):
                token = tuple(cells)
                if token != self.section_tokens[w]:
                    self.section_tokens[w] = token
                    self.section_revisions[w] += 1
        active_slices = set() if move is None else {cell[3] for cell in placed_cells(move)}
        for w, view in enumerate(self.views):
            selected = w == w_slice
            if selected != getattr(view, '_selected', None):
                view._selected = selected
                view.canvas.configure(highlightbackground=PALETTE['accent'] if selected else PALETTE['grid'])
            view.piece_cache, view.owners = self.piece_cache, self.owners
            view.section_revision = self.section_revisions[w]
            visible_move = move if w in active_slices else None
            view.draw(game, visible_move, progress if visible_move else 0.0, w, cutoff)



def draw_piece_sections(canvas, piece: str, rotation: int = 0, w_origin: int = 0, title: str = 'WHOLE ACTIVE PIECE') -> None:
    canvas.delete('all')
    shape = SHAPES[piece][rotation]
    levels = sorted({p[3] for p in shape.cells})
    width, height = max(60, canvas.winfo_width()), max(60, canvas.winfo_height())
    label = 'PIECE' if title == 'WHOLE ACTIVE PIECE' else title
    canvas.create_text(10, 13, anchor='w', text=f'{label}: {piece} · {len(shape.cells)} cells',
                       fill=PIECE_COLORS[piece], font=('TkDefaultFont', 8, 'bold'), tags=('preview_header',))
    section = width / len(levels)
    points = [(x + dx - y - dy, .48 * (x + dx + y + dy) - z - dz)
              for x, y, z, _ in shape.cells for dx, dy, dz in itertools.product((0, 1), repeat=3)]
    low_x, high_x = min(p[0] for p in points) - .2, max(p[0] for p in points) + .2
    low_y, high_y = min(p[1] for p in points) - .2, max(p[1] for p in points) + .2
    for wi, w in enumerate(levels):
        cells = [(x, y, z) for x, y, z, cw in shape.cells if cw == w]
        scale = max(3, min((section - 16) / (high_x - low_x), (height - 49) / (high_y - low_y)))
        ox, oy = section * (wi + .5) - (low_x + high_x) * scale / 2, (height + 29) / 2 - (low_y + high_y) * scale / 2
        canvas.create_text(section * (wi + .5), height - 10, text=f'W{w_origin + w}:{len(cells)}' if section < 95 else f'W={w_origin + w} · {len(cells)} cells',
                           fill=PIECE_COLORS[piece], font=('TkFixedFont', 8))
        if wi:
            canvas.create_line(section * wi, 30, section * wi, height - 25, fill=PALETTE['grid'], dash=(2, 4))
        occupied, faces = set(cells), []
        for x, y, z in cells:
            for normal, vertices in FACE_GEOMETRY:
                if sum(normal) <= 0 or (x + normal[0], y + normal[1], z + normal[2]) in occupied:
                    continue
                vertices = [(x + dx, y + dy, z + dz) for dx, dy, dz in vertices]
                shade = blend(PIECE_COLORS[piece], '#FFFFFF', .15) if normal[2] else blend(PIECE_COLORS[piece], '#000000', .3)
                projected = [(ox + (px - py) * scale, oy + (.48 * (px + py) - pz) * scale) for px, py, pz in vertices]
                faces.append((sum(sum(point) for point in vertices), projected, shade))
        for _, points, shade in sorted(faces):
            canvas.create_polygon(*[v for p in points for v in p], fill=shade, outline=PALETTE['board'], tags=('preview_face',))


class SliceAtlas:
    def __init__(self, canvas, on_select) -> None:
        self.canvas, self.on_select = canvas, on_select
        self.boxes = []
        self.last_key = None
        canvas.bind('<Button-1>', self.click)

    def click(self, event) -> None:
        for w, left, top, size in self.boxes:
            if left <= event.x < left + size and top <= event.y < top + size:
                self.on_select(w)
                break

    def draw(self, game: Game, move: Move | None, selected: int) -> None:
        c = self.canvas
        width, height = max(140, c.winfo_width()), max(140, c.winfo_height())
        key = (game.board, move, selected, width, height)
        if key == self.last_key:
            return
        self.last_key = key
        c.delete('all')
        self.boxes.clear()
        c.create_text(10, 14, anchor='w', text='ALL 16 W SECTIONS', fill=PALETTE['text'], font=('TkDefaultFont', 9, 'bold'))
        c.create_text(10, 33, anchor='w', text='Top views · click a section', fill=PALETTE['muted'], font=('TkDefaultFont', 8))
        size = max(8, min((width - 32) / 4, (height - 125) / 4))
        gap_x = (width - size * 4) / 5
        top_y = 57
        heights, _ = board_features(game.board)
        pending = () if move is None else placed_cells(move)
        for w in range(FOURTH):
            x0 = gap_x + (w % 4) * (size + gap_x)
            y0 = top_y + (w // 4) * (size + 21)
            self.boxes.append((w, x0, y0, size))
            cell = size / 16
            c.create_rectangle(x0 - 2, y0 - 2, x0 + size + 2, y0 + size + 2,
                               fill=PALETTE['board'], outline=PALETTE['accent'] if w == selected else PALETTE['grid'], width=2 if w == selected else 1)
            c.create_text(x0, y0 - 10, anchor='w', text=f'W {w:02}', fill=PALETTE['accent'] if w == selected else PALETTE['muted'], font=('TkFixedFont', 8))
            for y in range(DEPTH):
                for x in range(WIDTH):
                    value = heights[column_index(x, y, w)]
                    if value:
                        c.create_rectangle(x0 + x * cell, y0 + (15 - y) * cell, x0 + (x + 1) * cell, y0 + (16 - y) * cell,
                                           fill=blend(PALETTE['panel'], '#A9B5F5', .2 + .8 * value / HEIGHT), outline='')
            for x, y, z, cw in pending:
                if cw == w:
                    c.create_rectangle(x0 + x * cell, y0 + (15 - y) * cell, x0 + (x + 1) * cell, y0 + (16 - y) * cell,
                                       fill=PIECE_COLORS[move.piece], outline='#FFFFFF', width=.4, tags=('atlas_active',))


class NetworkView:
    def __init__(self, parent) -> None:
        import tkinter as tk
        self.canvas = tk.Canvas(parent, bg=PALETTE['board'], highlightthickness=0, width=630, height=520)
        self.canvas.pack(fill='both', expand=True)
        self.detail_var = tk.StringVar()
        self.detail = tk.Label(parent, textvariable=self.detail_var, bg=PALETTE['bg'], fg=PALETTE['muted'],
                               font=('TkFixedFont', 9), justify='left', anchor='nw', height=4, wraplength=620)
        self.detail.pack(fill='x', pady=(6, 0))
        self.values = None
        self.move = None
        self.w = 0
        self.hovered = self.pinned = None
        self.positions = {}
        self.node_items = {}
        self.layout_key = None
        self.paint_state = {}
        self.active_count = 0
        self.canvas.bind('<Configure>', lambda event: self.layout())
        self.canvas.bind('<Motion>', self.motion)
        self.canvas.bind('<Leave>', lambda event: self.leave())
        self.canvas.bind('<Button-1>', self.click)

    def update(self, move: Move | None, w: int) -> None:
        changed = move is not self.move
        if changed:
            self.move = move
            self.values = None if move is None else network_trace(move.heights, move.holes, move.layers)
            self.active_count = 0 if self.values is None else self.values.active_count
        if w != self.w:
            self.w = w
            self.layout_key = None
            self.hovered = self.pinned = None
        if self.layout_key is None:
            self.layout()
        elif changed:
            self.paint()

    def name(self, index: int) -> str:
        if index < AREA:
            x, y, w = column_coordinates(index)
            return f'h({x},{y},{w})'
        if index < INPUT_COUNT:
            return 'holes' if index == AREA else 'cleared hyperlayers'
        return NETWORK[index - INPUT_COUNT].name

    def layout(self) -> None:
        c = self.canvas
        width, height = max(360, c.winfo_width()), max(300, c.winfo_height())
        key = (width, height, self.w)
        if key == self.layout_key:
            return
        self.layout_key = key
        c.delete('all')
        self.positions.clear()
        self.node_items.clear()
        self.paint_state.clear()
        self.detail.configure(wraplength=max(350, c.winfo_width() - 4))
        c.create_text(12, 17, anchor='w', text=f'SIGMOID ACTIVATIONS / W {self.w:02d}' if width < 480 else f'LIVE SIGMOID ACTIVATIONS  /  W PAGE {self.w:02d}',
                      fill=PALETTE['text'], font=('TkDefaultFont', 10, 'bold'))
        c.create_text(12, 37, anchor='w', text='Whole-board score; one activation page.' if width < 480 else 'Whole-board score; heatmaps show one W page of actual neuron values.',
                      fill=PALETTE['muted'], font=('TkDefaultFont', 8))
        group_width = (width - 38) / 4
        cell = min((group_width - 7) / 16, (height - 153) / 32)
        top_y = (77, 112 + 16 * cell)
        origins = [12 + i * (group_width + 5) for i in range(4)]
        def grid(title: str, col: int, row: int, entries: list[tuple[int, int, int]]) -> None:
            left, top = origins[col], top_y[row]
            c.create_text(left, top - 15, anchor='w', text=title.split()[0] if width < 480 else title, fill='#A9B5F5' if col == 0 else PALETTE['accent'], font=('TkDefaultFont', 8, 'bold'))
            for gx, gy, index in entries:
                x, y = left + (gx + .5) * cell, top + (gy + .5) * cell
                item = c.create_rectangle(x - cell / 2 + .3, y - cell / 2 + .3, x + cell / 2 - .3, y + cell / 2 - .3,
                                          fill=PALETTE['panel'], outline=PALETTE['grid'], width=.5, tags=(f'node:{index}', 'unit'))
                self.positions[index] = (x, y)
                self.node_items[index] = (item, None)
        w = self.w
        grid('HEIGHTS 16×16', 0, 0, [(x, y, column_index(x, y, w)) for y in range(16) for x in range(16)])
        for col, axis, dx, dy, dw in ((1, 'X', 1, 0, 0), (2, 'Y', 0, 1, 0), (3, 'W', 0, 0, 1)):
            for row in (0, 1):
                entries = []
                if w + dw < FOURTH:
                    for y in range(DEPTH - dy):
                        for x in range(WIDTH - dx):
                            pair = (column_index(x, y, w), column_index(x + dx, y + dy, w + dw))
                            index = INPUT_COUNT + 2 * PAIR_INDEX[pair] + row
                            entries.append((x, y, index))
                grid(axis + ('+ sigmoid' if row == 0 else '− sigmoid'), col, row, entries)
                if not entries:
                    c.create_text(origins[col] + 6, top_y[row] + 35, anchor='w', text='Boundary\nNo W+1\nneighbor', fill=PALETTE['muted'], font=('TkDefaultFont', 8))
        x0, y0 = origins[0], top_y[1]
        for offset, name in ((0, 'holes'), (1, 'clears')):
            x, y = x0 + group_width * (.22 + offset * .52), y0 + 14
            index = AREA + offset
            item = c.create_oval(x - 13, y - 13, x + 13, y + 13, fill=PALETTE['panel'], outline=PALETTE['grid'], tags=(f'node:{index}', 'unit'))
            label = c.create_text(x, y, text='--', fill=PALETTE['text'], font=('TkFixedFont', 8), tags=(f'node:{index}', 'unit'))
            c.create_text(x, y + 25, text=name, fill=PALETTE['muted'], font=('TkDefaultFont', 8))
            self.positions[index], self.node_items[index] = (x, y), (item, label)
        x, y = x0 + group_width / 2, y0 + min(80, 16 * cell - 24)
        item = c.create_oval(x - 31, y - 25, x + 31, y + 25, fill=PALETTE['panel'], outline=PALETTE['accent'], tags=(f'node:{OUTPUT_INDEX}', 'unit'))
        label = c.create_text(x, y, text='--', fill=PALETTE['text'], font=('TkFixedFont', 9, 'bold'), tags=(f'node:{OUTPUT_INDEX}', 'unit'))
        c.create_text(x, y + 38, text='LINEAR SCORE', fill=PALETTE['muted'], font=('TkDefaultFont', 8))
        self.positions[OUTPUT_INDEX], self.node_items[OUTPUT_INDEX] = (x, y), (item, label)
        self.summary_item = c.create_text(12, height - 41, anchor='w', text='', fill=PALETTE['muted'], font=('TkDefaultFont', 8))
        c.create_text(12, height - 21, anchor='w', text='Brightness: sigmoid 0–1 · hover/pin · cyan + / amber −',
                      fill=PALETTE['muted'], font=('TkDefaultFont', 8))
        self.paint()

    def edges_for(self, index: int) -> list[tuple[int, int, float]]:
        if index == OUTPUT_INDEX:
            return []
        if index < INPUT_COUNT:
            result = [(index, OUTPUT_INDEX, HEIGHT_WEIGHT if index < AREA else HOLES_WEIGHT if index == AREA else LAYERS_WEIGHT)]
            if index < AREA:
                for j in ADJACENT[index]:
                    pair = (min(index, j), max(index, j))
                    p = PAIR_INDEX[pair]
                    for sign in (0, 1):
                        target = INPUT_COUNT + 2 * p + sign
                        weight = dict(NETWORK[target - INPUT_COUNT].edges)[index]
                        result.append((index, target, weight))
            return result
        node = NETWORK[index - INPUT_COUNT]
        return [(source, index, weight) for source, weight in node.edges] + [(index, OUTPUT_INDEX, HIDDEN_OUTPUT_WEIGHT)]

    def paint(self) -> None:
        if not self.node_items:
            return
        c, selected = self.canvas, self.pinned if self.pinned is not None else self.hovered
        c.delete('connection')
        related = {selected}
        edges = self.edges_for(selected) if selected is not None else []
        for source, target, weight in edges:
            related.update((source, target))
            if source not in self.positions or target not in self.positions:
                continue
            signal = 0 if self.values is None else self.values[source] * weight
            c.create_line(*self.positions[source], *self.positions[target], arrow='last',
                          fill='#67D8E5' if signal >= 0 else '#EDA87C',
                          width=1 + min(2, abs(signal) / 4), tags=('connection',),
                          dash=(3, 4) if source < INPUT_COUNT and target == OUTPUT_INDEX else ())
        c.tag_lower('connection')
        for index, (item, label) in self.node_items.items():
            value = 0 if self.values is None else self.values[index]
            scale = HEIGHT if index < AREA else AREA * HEIGHT if index in (AREA, OUTPUT_INDEX) else HEIGHT if index == AREA + 1 else 1
            amount = min(1, abs(value) / scale)
            if index < INPUT_COUNT or index == OUTPUT_INDEX:
                amount = math.sqrt(amount)
            hue = '#EDA87C' if value < 0 else '#A9B5F5' if index < INPUT_COUNT else PALETTE['accent']
            color = blend(PALETTE['panel'], hue, .07 + .8 * amount) if self.values is not None else PALETTE['panel']
            if selected is not None and index not in related:
                color = blend(PALETTE['board'], color, .4)
            outline = PALETTE['text'] if index in related else PALETTE['grid']
            text = ('--' if self.values is None else f'{value:.1f}' if index == OUTPUT_INDEX else f'{value:g}') if label else None
            state = (color, outline, text)
            previous = self.paint_state.get(index)
            if state != previous:
                if previous is None or state[:2] != previous[:2]:
                    c.itemconfigure(item, fill=color, outline=outline)
                if label and (previous is None or text != previous[2]):
                    c.itemconfigure(label, text=text)
                self.paint_state[index] = state
        active = self.active_count
        c.itemconfigure(self.summary_item, text=f'{active:,}/23,040 ≥ 0.50 · 2 shared filters / 17 bins · page: {len(self.positions):,}')
        if self.values is None:
            text = 'No selected legal drop. Values are unavailable, not zero.\nThe network updates after a search or a manual move.'
        elif selected is None:
            text = ('Shared sigmoid stencil: 2 filters, 33 lookup values, 17 pooled difference bins.\nHeatmap cells reconstruct the original logical activations on demand.\nThe output uses all 16 W pages; target constraints are outside the sigmoid network.')
        elif selected < INPUT_COUNT:
            text = f'{self.name(selected)} = {self.values[selected]:g}\n'
            text += 'Column height after drop and clears.' if selected < AREA else 'Total empty cells below occupied cells.' if selected == AREA else 'Full X/Y/W hyperlayers cleared by this move.'
            text += f'\n{len(edges)} incident connections. Hidden pages remain part of the computation.'
        else:
            value = self.values[selected]
            if selected == OUTPUT_INDEX:
                text = f'Linear score = {value:+.6f}\nExact pooled sum of 4098 inputs + 23040 logical sigmoid activations.\n2 shared filters; no stored dense graph. Output bias {OUTPUT_BIAS:.6f}.'
            else:
                node = NETWORK[selected - INPUT_COUNT]
                terms = ' '.join(f'{weight:+g}×{self.name(source)}[{self.values[source]:g}]' for source, weight in node.edges)
                pre = node.bias + sum(self.values[source] * weight for source, weight in node.edges)
                text = f'{node.name}: z = {terms} {node.bias:+g} = {pre:g}\nsigmoid(z) = {value:.6f}\nOutput contribution = {value:.6f} × {HIDDEN_OUTPUT_WEIGHT:.6f} = {value * HIDDEN_OUTPUT_WEIGHT:.6f}'
        self.detail_var.set(text + ('  [pinned]' if self.pinned is not None else ''))

    def target(self, event) -> int | None:
        for item in reversed(self.canvas.find_overlapping(event.x, event.y, event.x, event.y)):
            for tag in self.canvas.gettags(item):
                if tag.startswith('node:'):
                    return int(tag.split(':')[1])
        return None

    def motion(self, event) -> None:
        target = self.target(event)
        if target != self.hovered:
            self.hovered = target
            self.paint()

    def leave(self) -> None:
        self.hovered = None
        self.paint()

    def click(self, event) -> None:
        target = self.target(event)
        self.pinned = None if target == self.pinned else target
        self.paint()


class FramePacer:
    def __init__(self, fps: float = 30.0, enabled: bool = True) -> None:
        if not math.isfinite(fps) or not 1 <= fps <= 120:
            raise ValueError('UI FPS must be between 1 and 120.')
        self.enabled = enabled
        self.fps = float(fps)
        self.base_interval = 1.0 / fps
        self.interval = self.base_interval
        self.next_at = 0.0
        self.ema_seconds = 0.0
        self.requests = self.rendered = self.skipped = self.coalesced = 0

    def allow(self, now: float, force: bool = False) -> bool:
        self.requests += 1
        if force or not self.enabled or now >= self.next_at:
            return True
        self.skipped += 1
        return False

    def complete(self, started: float, finished: float) -> None:
        cost = max(0.0, finished - started)
        self.rendered += 1
        self.ema_seconds = cost if self.rendered == 1 else .8 * self.ema_seconds + .2 * cost
        self.interval = max(self.base_interval, min(.2, 2 * self.ema_seconds))
        self.next_at = max(started + self.interval, finished + .001)

    def report(self) -> dict:
        return {'adaptive_frameskip': self.enabled, 'requested_fps': self.fps,
                'requests': self.requests, 'rendered_frames': self.rendered,
                'deferred_requests': self.skipped, 'coalesced_requests': self.coalesced,
                'interval_ms': self.interval * 1000, 'render_ema_ms': self.ema_seconds * 1000}


PLAYBACK_SPEEDS = {'0.5/s': .5, '1/s': 1.0, '3/s': 3.0, '6/s': 6.0, '12/s': 12.0, 'Unlimited': 0.0}


class TetrisApp:
    def __init__(self, root, seed: int = 7, depth: int = 2, beam: int = DEFAULT_BEAM,
                 candidates: int = DEFAULT_CANDIDATES, piece_set: str = 'expanded',
                 policy: str = 'hyperlayer', target: int = 1, stop_after_clear: bool = False,
                 ui_fps: float = 30.0, frameskip: bool = True) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.root, self.beam, self.candidates = root, beam, candidates
        self.frame_pacer = FramePacer(ui_fps, frameskip)
        self._draw_id = None
        self._in_tick = False
        self._force_draw = False
        if policy not in ('network', 'hyperlayer'):
            raise ValueError('Unknown planner policy.')
        target_layer_status(EMPTY_BOARD, target)
        self.game = Game(seed, piece_set)
        self.policy = tk.StringVar(value=policy)
        self.target = tk.StringVar(value=str(target))
        self.stop_after_clear = stop_after_clear
        self.goal_complete = False
        self.planner_blocked = ''
        self.target_progress = tk.StringVar()
        self.running = self.closed = self.pending_step = False
        self.progress, self.previous = 0.0, time.perf_counter()
        self.search = None
        self.evaluations = 0
        self.planned: Move | None = None
        self.decision: Decision | None = None
        self.manual_rotation = self.manual_x = self.manual_y = self.manual_w = 0
        self.errors = []
        self.last_small_view = None
        self.human = tk.BooleanVar(value=False)
        self.piece_set = tk.StringVar(value=piece_set)
        self.gallery = None
        self.lookahead = tk.BooleanVar(value=depth == 2)
        self.follow = tk.BooleanVar(value=True)
        self.w_slice = tk.IntVar(value=0)
        self.cutoff = tk.IntVar(value=HEIGHT - 1)
        self.speed = tk.DoubleVar(value=3.0)
        self.speed_choice = tk.StringVar(value='3/s')
        self.stats = tk.StringVar()
        self.status = tk.StringVar()
        self.features = tk.StringVar()
        self.search_status = tk.StringVar()
        root.report_callback_exception = self.callback_error
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.title('Sparse Tetris 4D | Optimized | Rigid pieces | Sigmoid network')
        root.geometry(f'{min(1560, max(1120, root.winfo_screenwidth() - 40))}x{min(960, max(760, root.winfo_screenheight() - 60))}')
        root.minsize(1120, 760)
        root.configure(bg=PALETTE['bg'])
        style = ttk.Style(root)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure('Tetris.TFrame', background=PALETTE['bg'])
        style.configure('Tetris.TLabel', background=PALETTE['bg'], foreground=PALETTE['text'], font=('TkDefaultFont', 10))
        style.configure('Muted.Tetris.TLabel', foreground=PALETTE['muted'])
        style.configure('Tetris.TButton', padding=(9, 5))
        style.configure('Tetris.TCheckbutton', background=PALETTE['bg'], foreground=PALETTE['text'])
        style.map('Tetris.TCheckbutton', background=[('active', PALETTE['bg'])])
        outer = ttk.Frame(root, padding=14, style='Tetris.TFrame')
        outer.pack(fill='both', expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        header = ttk.Frame(outer, style='Tetris.TFrame')
        header.grid(row=0, column=0, sticky='ew')
        ttk.Label(header, text='Sparse Tetris 4D', font=('TkDefaultFont', 22, 'bold'), style='Tetris.TLabel').pack(side='left')
        ttk.Label(header, textvariable=self.stats, style='Tetris.TLabel', font=('TkFixedFont', 10)).pack(side='right')
        ttk.Label(outer, text='16⁴ = 65,536 cells  /  4,098 inputs → 23,040 sigmoid units → 1 linear score  /  73,218 connections',
                  style='Muted.Tetris.TLabel').grid(row=1, column=0, sticky='w', pady=(4, 10))
        self.content = ttk.Frame(outer, style='Tetris.TFrame')
        self.content.grid(row=2, column=0, sticky='nsew')
        self.content.rowconfigure(0, weight=1)
        for col, weight, minimum in ((0, 6, 420), (1, 2, 160), (2, 4, 420)):
            self.content.columnconfigure(col, weight=weight, minsize=minimum)
        left = ttk.Frame(self.content, style='Tetris.TFrame')
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.well = AllWellView(left, self.select_slice)
        self.piece_preview = tk.Canvas(left, bg=PALETTE['board'], height=124, highlightthickness=0)
        self.piece_preview.grid(row=1, column=0, sticky='ew', pady=(8, 0))
        center = ttk.Frame(self.content, style='Tetris.TFrame')
        center.grid(row=0, column=1, sticky='nsew', padx=(0, 10))
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.atlas_canvas = tk.Canvas(center, bg=PALETTE['board'], highlightthickness=0, width=260, height=350)
        self.atlas_canvas.grid(row=0, column=0, sticky='nsew')
        self.atlas = SliceAtlas(self.atlas_canvas, self.select_slice)
        self.next_preview = tk.Canvas(center, bg=PALETTE['board'], highlightthickness=0, height=105)
        self.next_preview.grid(row=1, column=0, sticky='ew', pady=(8, 6))
        ttk.Label(center, textvariable=self.features, style='Tetris.TLabel', font=('TkFixedFont', 9), justify='left').grid(row=2, column=0, sticky='w')
        right = ttk.Frame(self.content, style='Tetris.TFrame')
        right.grid(row=0, column=2, sticky='nsew')
        self.network = NetworkView(right)
        bars = ttk.Frame(outer, style='Tetris.TFrame')
        bars.grid(row=3, column=0, sticky='ew', pady=(9, 5))
        bars.columnconfigure(1, weight=1)
        bars.columnconfigure(3, weight=1)
        ttk.Label(bars, text='W section', style='Muted.Tetris.TLabel').grid(row=0, column=0, padx=(0, 8))
        self.w_scale = tk.Scale(bars, from_=0, to=15, orient='horizontal', variable=self.w_slice, showvalue=True,
                                command=self.slice_changed, bg=PALETTE['bg'], fg=PALETTE['text'], highlightthickness=0,
                                troughcolor=PALETTE['panel'], length=260)
        self.w_scale.grid(row=0, column=1, sticky='ew')
        ttk.Label(bars, text='Stack Z cutaway', style='Muted.Tetris.TLabel').grid(row=0, column=2, padx=10)
        tk.Scale(bars, from_=0, to=15, orient='horizontal', variable=self.cutoff, showvalue=True,
                 command=lambda value: self.draw(), bg=PALETTE['bg'], fg=PALETTE['text'], highlightthickness=0,
                 troughcolor=PALETTE['panel'], length=240).grid(row=0, column=3, sticky='ew')
        ttk.Checkbutton(bars, text='Follow active W', variable=self.follow, style='Tetris.TCheckbutton', command=self.follow_changed).grid(row=0, column=4, padx=(10, 0))
        controls = ttk.Frame(outer, style='Tetris.TFrame')
        controls.grid(row=4, column=0, sticky='ew', pady=(3, 6))
        self.start_button = ttk.Button(controls, text='Start', command=self.toggle, style='Tetris.TButton')
        self.start_button.pack(side='left')
        ttk.Button(controls, text='Drop / Step', command=self.step, style='Tetris.TButton').pack(side='left', padx=5)
        ttk.Button(controls, text='Restart', command=self.reset, style='Tetris.TButton').pack(side='left')
        ttk.Button(controls, text='New seed', command=lambda: self.reset(True), style='Tetris.TButton').pack(side='left', padx=5)
        ttk.Checkbutton(controls, text='Human', variable=self.human, command=self.change_mode, style='Tetris.TCheckbutton').pack(side='left', padx=5)
        self.lookahead_button = ttk.Checkbutton(controls, text='Lookahead', variable=self.lookahead, command=self.refresh_plan, style='Tetris.TCheckbutton')
        self.lookahead_button.pack(side='left', padx=5)
        ttk.Label(controls, text='Speed', style='Muted.Tetris.TLabel').pack(side='left', padx=(12, 5))
        self.speed_box = ttk.Combobox(controls, textvariable=self.speed_choice,
                                      values=tuple(PLAYBACK_SPEEDS), state='readonly', width=10)
        self.speed_box.pack(side='left')
        self.speed_box.bind('<<ComboboxSelected>>', self.change_speed)
        ttk.Label(controls, textvariable=self.search_status, style='Muted.Tetris.TLabel').pack(side='right')
        rotations_row = ttk.Frame(outer, style='Tetris.TFrame')
        rotations_row.grid(row=5, column=0, sticky='ew', pady=(0, 5))
        ttk.Label(rotations_row, text='Rotate plane (Human):', style='Muted.Tetris.TLabel').pack(side='left', padx=(0, 8))
        self.rotation_buttons = []
        for i, name in enumerate(PLANE_NAMES):
            button = ttk.Button(rotations_row, text=f'{i + 1} {name}', command=lambda p=i: self.rotate(p), style='Tetris.TButton', width=6)
            button.pack(side='left', padx=(0, 3))
            self.rotation_buttons.append(button)
        ttk.Label(rotations_row, text='Shift reverses · Q/E: W', style='Muted.Tetris.TLabel').pack(side='left', padx=(5, 8))
        ttk.Label(rotations_row, text='Pieces:', style='Muted.Tetris.TLabel').pack(side='left')
        self.piece_set_box = ttk.Combobox(rotations_row, textvariable=self.piece_set, values=tuple(PIECE_SETS),
                                         state='readonly', width=9)
        self.piece_set_box.pack(side='left', padx=5)
        self.piece_set_box.bind('<<ComboboxSelected>>', self.change_piece_set)
        ttk.Button(rotations_row, text='New pieces', command=self.show_gallery, style='Tetris.TButton').pack(side='left')
        policy_row = ttk.Frame(outer, style='Tetris.TFrame')
        policy_row.grid(row=6, column=0, sticky='ew', pady=(0, 5))
        ttk.Label(policy_row, text='Planner:', style='Muted.Tetris.TLabel').pack(side='left')
        self.policy_box = ttk.Combobox(policy_row, textvariable=self.policy, values=('network', 'hyperlayer'), state='readonly', width=12)
        self.policy_box.pack(side='left', padx=6)
        self.policy_box.bind('<<ComboboxSelected>>', self.change_policy)
        ttk.Label(policy_row, text='Target Z:', style='Muted.Tetris.TLabel').pack(side='left')
        self.target_box = ttk.Combobox(policy_row, textvariable=self.target, values=tuple(range(HEIGHT)), state='readonly', width=3)
        self.target_box.pack(side='left', padx=6)
        self.target_box.bind('<<ComboboxSelected>>', self.change_policy)
        ttk.Label(policy_row, textvariable=self.target_progress, style='Muted.Tetris.TLabel').pack(side='left', padx=8)
        ttk.Label(outer, textvariable=self.status, style='Muted.Tetris.TLabel', wraplength=1400).grid(row=7, column=0, sticky='w')
        def install_keys(widget) -> None:
            widget.bindtags(('Tetris4DKeys',) + widget.bindtags())
            for child in widget.winfo_children():
                install_keys(child)
        install_keys(root)
        root.bind_class('Tetris4DKeys', '<KeyPress>', self.key_down)
        for button in self.rotation_buttons:
            button.configure(state='disabled')
        root.bind('<Configure>', lambda event: self.redraw_small())
        self.refresh_plan()
        self.tick_id = root.after(16, self.tick)

    def change_speed(self, event=None) -> None:
        self.speed.set(PLAYBACK_SPEEDS[self.speed_choice.get()])
        self.previous = time.perf_counter()
        self.draw(force=True)

    def change_policy(self, event=None) -> None:
        self.running = self.pending_step = self.goal_complete = False
        self.start_button.configure(text='Start')
        self.refresh_plan()

    def change_piece_set(self, event=None) -> None:
        self.game.piece_set = self.piece_set.get()
        self.reset()

    def use_piece(self, name: str) -> None:
        if name not in SHAPES:
            raise ValueError('Unknown practice piece.')
        self.search = None
        self.game.current = name
        self.game.game_over = not has_legal_move(self.game.board, name)
        self.manual_rotation = self.manual_x = self.manual_y = self.manual_w = 0
        self.planned = None
        self.human.set(True)
        self.change_mode()
        if self.gallery is not None and self.gallery.winfo_exists():
            self.gallery.destroy()
        self.gallery = None
        self.root.focus_set()

    def show_gallery(self) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.running = False
        self.start_button.configure(text='Resume')
        if self.gallery is not None and self.gallery.winfo_exists():
            self.gallery.lift()
            return
        window = tk.Toplevel(self.root)
        self.gallery = window
        window.title('Eight new 4D pieces | Cell centers span all four dimensions')
        window.configure(bg=PALETTE['bg'])
        window.geometry('1080x660')
        window.minsize(960, 580)
        window.transient(self.root)
        ttk.Label(window, text='NEW 4D PIECES', font=('TkDefaultFont', 18, 'bold'), style='Tetris.TLabel').grid(
            row=0, column=0, columnspan=4, sticky='w', padx=14, pady=(12, 4))
        ttk.Label(window, text='All eight are connected and rigid. Each center geometry spans four dimensions. '
                              'Choose a card to replace the current piece in Human mode.',
                  style='Muted.Tetris.TLabel', wraplength=1030).grid(row=1, column=0, columnspan=4, sticky='w', padx=14)
        for column in range(4):
            window.columnconfigure(column, weight=1, uniform='cards')
        for row in (2, 3):
            window.rowconfigure(row, weight=1)
        for i, name in enumerate(NEW_PIECE_CELLS):
            card = ttk.Frame(window, style='Tetris.TFrame', padding=8)
            card.grid(row=2 + i // 4, column=i % 4, sticky='nsew', padx=5, pady=8)
            card.columnconfigure(0, weight=1)
            card.rowconfigure(0, weight=1)
            card.rowconfigure(2, minsize=56)
            preview = tk.Canvas(card, width=225, height=138, bg=PALETTE['board'], highlightthickness=0)
            preview.grid(row=0, column=0, sticky='nsew')
            preview.bind('<Configure>', lambda event, c=preview, n=name: draw_piece_sections(c, n, title='NEW'))
            ttk.Label(card, text=f'{PIECE_SIZES[name]} cells · {len(SHAPES[name])} orientations',
                      style='Muted.Tetris.TLabel').grid(row=1, column=0, sticky='w', pady=(4, 0))
            ttk.Label(card, text=PIECE_DESCRIPTIONS[name], style='Muted.Tetris.TLabel', wraplength=225).grid(
                row=2, column=0, sticky='w', pady=4)
            ttk.Button(card, text=f'Use {name}', command=lambda n=name: self.use_piece(n),
                       style='Tetris.TButton').grid(row=3, column=0, sticky='ew')
        window.update_idletasks()

    def callback_error(self, kind, value, traceback) -> None:
        import traceback as tb
        message = ''.join(tb.format_exception(kind, value, traceback))
        self.errors.append(message)
        print(message, file=sys.stderr)
        self.running = False
        self.status.set('UI error: ' + str(value))

    def close(self) -> None:
        self.closed = True
        self.search = None
        if getattr(self, 'tick_id', None):
            self.root.after_cancel(self.tick_id)
        if self._draw_id is not None:
            self.root.after_cancel(self._draw_id)
            self._draw_id = None
        self.root.destroy()

    def redraw_small(self) -> None:
        self.last_small_view = None

    def follow_changed(self) -> None:
        if self.follow.get() and self.planned:
            self.w_slice.set(self.planned.candidate.w)
            self.slice_changed()
        else:
            self.draw()

    def select_slice(self, w: int) -> None:
        self.follow.set(False)
        self.w_slice.set(max(0, min(15, w)))
        self.slice_changed()

    def slice_changed(self, value=None) -> None:
        if hasattr(self, 'network'):
            self.draw(force=True)

    def refresh_plan(self) -> None:
        self.search = None
        self.planned = self.decision = None
        self.progress = 0
        self.evaluations = 0
        self.planner_blocked = ''
        self.lookahead_button.configure(state='disabled' if self.policy.get() == 'hyperlayer' else 'normal')
        self.target_box.configure(state='readonly' if self.policy.get() == 'hyperlayer' else 'disabled')
        if self.goal_complete and not self.human.get():
            self.running = self.pending_step = False
            self.start_button.configure(text='Clear reached')
            self.draw(force=True)
            return
        if self.game.game_over:
            self.running = False
            self.start_button.configure(text='Start')
            self.draw(force=True)
            return
        if self.human.get():
            self.manual_rotation %= len(SHAPES[self.game.current])
            sx, sy, sz, sw = SHAPES[self.game.current][self.manual_rotation].extents
            self.manual_x = max(0, min(WIDTH - sx, self.manual_x))
            self.manual_y = max(0, min(DEPTH - sy, self.manual_y))
            self.manual_w = max(0, min(FOURTH - sw, self.manual_w))
            self.planned = make_move(self.game.board, self.game.current, self.manual_rotation,
                                     self.manual_x, self.manual_y, self.manual_w)
            self.finish_plan()
        else:
            if self.policy.get() == 'hyperlayer':
                self.search = hyperlayer_decision_steps(self.game.board, self.game.current, int(self.target.get()),
                                                        self.candidates, self.game.pieces)
            else:
                self.search = decision_steps(self.game.board, self.game.current, self.game.next_piece,
                                             2 if self.lookahead.get() else 1, self.beam, self.candidates,
                                             self.game.seed * 8191 + self.game.pieces)
            self.draw()

    def finish_plan(self) -> None:
        self.progress = 0
        if self.planned and self.follow.get():
            self.w_slice.set(self.planned.candidate.w)
        self.last_small_view = None
        self.draw(force=not self.running)

    def change_mode(self) -> None:
        self.running = self.pending_step = False
        self.start_button.configure(text='Start', state='disabled' if self.human.get() else 'normal')
        for button in self.rotation_buttons:
            button.configure(state='normal' if self.human.get() else 'disabled')
        if self.planned:
            p = self.planned.candidate
            self.manual_rotation, self.manual_x, self.manual_y, self.manual_w = p.rotation, p.x, p.y, p.w
        self.refresh_plan()

    def toggle(self) -> None:
        if self.human.get() or self.game.game_over or self.planner_blocked or self.goal_complete:
            return
        self.running = not self.running
        self.previous = time.perf_counter()
        self.start_button.configure(text='Pause' if self.running else 'Resume')
        self.draw(force=True)

    def reset(self, new_seed: bool = False) -> None:
        self.search = None
        self.running = self.pending_step = False
        if new_seed:
            self.game.seed += 1
        self.game.reset()
        self.goal_complete = False
        self.planner_blocked = ''
        self.manual_rotation = self.manual_x = self.manual_y = self.manual_w = 0
        self.start_button.configure(text='Start')
        self.last_small_view = None
        self.refresh_plan()

    def place(self) -> None:
        if self.planned is None or self.game.game_over:
            return
        cleared = self.planned.layers
        self.game.commit(self.planned)
        if cleared and not self.human.get():
            if self.stop_after_clear:
                self.goal_complete = True
                self.running = False
            elif self.policy.get() == 'hyperlayer':
                target = next_hyperlayer_target(self.game.board, int(self.target.get()))
                if target is not None:
                    self.target.set(str(target))
        self.manual_rotation = 0
        self.pending_step = False
        self.refresh_plan()

    def step(self) -> None:
        self.running = False
        self.start_button.configure(text='Resume')
        if self.search is not None:
            self.pending_step = True
        else:
            self.place()

    def rotate(self, plane: int, reverse: bool = False) -> None:
        if not self.human.get() or self.game.game_over:
            return
        count = 3 if reverse else 1
        for _ in range(count):
            self.manual_rotation = ROTATION_NEXT[self.game.current][self.manual_rotation][plane]
        self.refresh_plan()

    def key_down(self, event) -> str | None:
        if event.widget in (self.piece_set_box, self.policy_box, self.target_box, self.speed_box):
            return None
        key = event.keysym.lower()
        shifted_planes = {'exclam': 0, 'at': 1, 'numbersign': 2, 'dollar': 3, 'percent': 4, 'asciicircum': 5}
        if self.human.get() and not self.game.game_over and key in shifted_planes:
            self.rotate(shifted_planes[key], True)
            return 'break'
        if key == 'escape':
            self.running = False
            self.start_button.configure(text='Resume')
        elif key == 'space':
            self.step() if self.human.get() else self.toggle()
        elif key in ('return', 'kp_enter'):
            self.step()
        elif key == 'r':
            self.reset()
        elif key == 'home':
            self.well.reset_camera()
        elif key in ('bracketleft', 'bracketright'):
            self.select_slice(self.w_slice.get() + (1 if key == 'bracketright' else -1))
        elif self.human.get() and not self.game.game_over:
            if key in '123456' and len(key) == 1:
                self.rotate(int(key) - 1, bool(event.state & 1))
            elif key in ('left', 'right', 'up', 'down', 'w', 'a', 's', 'd', 'q', 'e'):
                if key in ('left', 'a'):
                    self.manual_x -= 1
                elif key in ('right', 'd'):
                    self.manual_x += 1
                elif key in ('up', 'w'):
                    self.manual_y += 1
                elif key in ('down', 's'):
                    self.manual_y -= 1
                elif key == 'q':
                    self.manual_w -= 1
                elif key == 'e':
                    self.manual_w += 1
                self.refresh_plan()
            else:
                return None
        else:
            return None
        return 'break'

    def draw(self, force: bool = False) -> None:
        if self.closed or not hasattr(self, 'well'):
            return
        if self._in_tick and self.frame_pacer.enabled:
            self.frame_pacer.coalesced += 1
            self._force_draw = getattr(self, '_force_draw', False) or force
            return
        now = time.perf_counter()
        if not self.frame_pacer.allow(now, force):
            if self._draw_id is None:
                delay = max(1, math.ceil((self.frame_pacer.next_at - now) * 1000))
                self._draw_id = self.root.after(delay, self._flush_draw)
            return
        if self._draw_id is not None:
            self.root.after_cancel(self._draw_id)
            self._draw_id = None
        started = time.perf_counter()
        try:
            self._draw_now()
        finally:
            self.frame_pacer.complete(started, time.perf_counter())

    def _flush_draw(self) -> None:
        self._draw_id = None
        self.draw()

    def render_metrics(self) -> dict:
        result = self.frame_pacer.report()
        rate = self.speed.get()
        result.update(playback_mode='unlimited' if rate == 0 else 'animated',
                      animation_rate_per_second=None if rate == 0 else rate)
        result.update(visible_w_sections=list(self.well.visible_w_sections),
                      scene_rebuilds=sum(view.scene_rebuilds for view in self.well.views),
                      unchanged_section_skips=sum(view.draw_skips for view in self.well.views))
        return result

    def _draw_now(self) -> None:
        w = self.w_slice.get()
        self.network.update(self.planned, w)
        if self.policy.get() == 'hyperlayer':
            layer = target_layer_status(self.game.board, int(self.target.get()))
            behavior = 'pause on clear requested' if self.stop_after_clear else 'continuous play'
            self.target_progress.set(f'Z={layer.target}: {layer.filled:,}/{AREA:,} filled · {layer.buried:,} buried gaps · {behavior}')
        else:
            self.target_progress.set('Neural score with optional next-piece lookahead')
        self.well.draw(self.game, self.planned, self.progress, w, self.cutoff.get())
        self.atlas.draw(self.game, self.planned, w)
        self.stats.set(f'Pieces {self.game.pieces:,} | Cells {self.game.cells_placed:,} | Clears {self.game.layers:,} | {self.game.piece_set}')
        if self.planned:
            m, p = self.planned, self.planned.candidate
            self.features.set(f'{m.piece} · {PIECE_SIZES[m.piece]} CELLS\nHeight sum {sum(m.heights):>7}\nHoles      {m.holes:>7}\nClears     {m.layers:>7}\nScore      {m.score:>+9.3f}\nX {p.x:02}  Y {p.y:02}  W {p.w:02}')
        else:
            self.features.set('Planning a legal drop…' if self.search is not None else 'No legal drop selected.')
        key = (self.game.current, self.game.next_piece, self.planned,
               self.piece_preview.winfo_width(), self.next_preview.winfo_width())
        if key != self.last_small_view:
            self.last_small_view = key
            p = self.planned.candidate if self.planned else None
            draw_piece_sections(self.piece_preview, self.game.current, p.rotation if p else self.manual_rotation if self.human.get() else 0,
                                p.w if p else self.manual_w if self.human.get() else 0)
            draw_piece_sections(self.next_preview, self.game.next_piece, title='NEXT')
        if self.search is not None:
            self.search_status.set(f'Searching: {self.evaluations:,} candidates')
        elif self.decision:
            label = f'Target +{self.decision.coverage_gain}' if self.decision.policy == 'hyperlayer' else f'Depth {self.decision.depth}'
            self.search_status.set(f'{label} · {self.decision.evaluations:,} evals · {self.decision.milliseconds:.0f} ms')
        else:
            self.search_status.set('Manual' if self.human.get() else '')
        if self.goal_complete and not self.human.get():
            self.status.set('HYPERLAYER CLEARED. Paused by --stop-after-clear. Restart or change planner to play again.')
        elif self.planner_blocked:
            self.status.set(self.planner_blocked)
        elif self.game.game_over:
            self.status.set('GAME OVER. No legal rigid placement remains. R restarts.')
        elif self.human.get():
            self.status.set('Human: arrows/WASD move X/Y · Q/E move W · 1–6 rotate XY/XZ/XW/YZ/YW/ZW · Shift reverses · Space/Enter drops · [ ] selects W view.')
        else:
            mode = ('hyperlayer planner, depth 1, protected target + sigmoid tie-break' if self.policy.get() == 'hyperlayer' else
                    f'{self.candidates:,} candidates/board, beam {self.beam}' if self.candidates else f'all placements/board, beam {self.beam}')
            pace = 'Unlimited · no animation delay' if self.speed.get() == 0 else f'{self.speed.get():g}/s animation'
            self.status.set(f'{"Running" if self.running else "Paused"} · {pace} · {mode} · Space start/pause · Enter step · [ ] W view · R restart. A clear needs 4,096 cells at one Z.')

    def _advance_search(self, deadline: float) -> None:
        while self.search is not None and time.perf_counter() < deadline:
            try:
                self.evaluations = next(self.search)
            except StopIteration as result:
                self.decision = result.value
                self.search = None
                self.planned = self.decision.move if self.decision else None
                if self.decision is None:
                    self.running = self.pending_step = False
                    self.game.game_over = not has_legal_move(self.game.board, self.game.current)
                    if not self.game.game_over:
                        layer = target_layer_status(self.game.board, int(self.target.get()))
                        reason = f'{layer.buried} buried target gaps' if layer.buried else 'no target-preserving placement'
                        self.planner_blocked = f'PLANNER PAUSED: {reason}. Choose another target, Human, network, or restart. Not game over.'
                    self.start_button.configure(text='Paused')
                self.finish_plan()
                if self.pending_step:
                    self.place()
                break

    def tick(self) -> None:
        if self.closed:
            return
        now = time.perf_counter()
        dt = min(.08, now - self.previous)
        self.previous = now
        self._in_tick = True
        try:
            if self.running and not self.human.get() and self.speed.get() == 0:
                deadline = now + .008
                while (self.running and not self.closed and not self.game.game_over
                       and time.perf_counter() < deadline):
                    self._advance_search(deadline)
                    if not self.running or self.search is not None or self.planned is None:
                        break
                    self.place()
            else:
                was_searching = self.search is not None
                self._advance_search(time.perf_counter() + .008)
                if self.running and not was_searching and self.search is None and self.planned:
                    self.progress += dt * self.speed.get()
                    if self.progress >= 1:
                        self.place()
        finally:
            self._in_tick = False
        if self.closed:
            return
        self.draw(force=self._force_draw)
        self._force_draw = False
        elapsed = time.perf_counter() - now
        delay = 1 if self.running and not self.human.get() and self.speed.get() == 0 else max(1, 16 - int(elapsed * 1000))
        self.tick_id = self.root.after(delay, self.tick)


def reference_drop(board: Board, piece: str, rotation: int, x: int, y: int, w: int) -> tuple[Board, int, int] | None:
    shape = SHAPES[piece][rotation]
    sx, sy, sz, sw = shape.extents
    if not (0 <= x <= WIDTH - sx and 0 <= y <= DEPTH - sy and 0 <= w <= FOURTH - sw):
        return None
    def blocked(z: int) -> bool:
        for dx, dy, dz, dw in shape.cells:
            zz = z + dz
            if zz < 0 or (zz < HEIGHT and board[column_index(x + dx, y + dy, w + dw)] & (1 << zz)):
                return True
        return False
    z = HEIGHT + sz
    while not blocked(z - 1):
        z -= 1
    if z + sz > HEIGHT:
        return None
    columns = list(board)
    for dx, dy, dz, dw in shape.cells:
        columns[column_index(x + dx, y + dy, w + dw)] |= 1 << (z + dz)
    retained = [level for level in range(HEIGHT) if not all(mask & (1 << level) for mask in columns)]
    result = tuple(sum(((mask >> old) & 1) << new for new, old in enumerate(retained)) for mask in columns)
    return result, HEIGHT - len(retained), z


def run_piece_tests() -> dict:
    rng, checks = random.Random(811), 0
    started = time.perf_counter()
    def check(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1
    check(len(NEW_PIECE_CELLS) == 8 and len(PIECE_NAMES) == 15, 'Expanded catalog sizes')
    canonical = set()
    for name, cells in PIECE_CELLS.items():
        check(cells_connected(cells), f'{name} must be face-connected')
        check(len(cells) == len(set(cells)) == PIECE_SIZES[name], 'Distinct cells and declared volume')
        check(affine_dimension(cells) == 4 if name in NEW_PIECE_CELLS else affine_dimension(cells) < 4,
              'New shapes must have intrinsically four-dimensional center geometry')
        key = min(shape.cells for shape in SHAPES[name])
        check(key not in canonical, 'Catalog shapes must not be duplicates under proper rotation')
        canonical.add(key)
        for shape in SHAPES[name]:
            check(cells_connected(shape.cells), 'Rotation preserves face connectivity')
            check(sum(mask.bit_count() for _, mask, _ in shape.columns) == PIECE_SIZES[name], 'Column masks preserve all cells')
            check(sum(count for _, count in shape.levels) == PIECE_SIZES[name], 'Level counts preserve all cells')
    for piece_set, names in PIECE_SETS.items():
        a, b = PieceStream(41, piece_set), PieceStream(41, piece_set)
        for _ in range(6):
            bag = tuple(a.pop() for _ in names)
            check(set(bag) == set(names) and len(set(bag)) == len(names), 'One of every selected shape in each bag')
            check(bag == tuple(b.pop() for _ in names), 'Repeatable piece stream')
    collision_cases = clear_cases = 0
    for name in NEW_PIECE_CELLS:
        for iteration in range(12):
            board = tuple(rng.getrandbits(rng.randrange(HEIGHT + 1)) if rng.random() < .06 else 0 for _ in range(AREA))
            rotation = rng.randrange(len(SHAPES[name]))
            sx, sy, _, sw = SHAPES[name][rotation].extents
            x, y, w = rng.randrange(WIDTH - sx + 1), rng.randrange(DEPTH - sy + 1), rng.randrange(FOURTH - sw + 1)
            move = make_move(board, name, rotation, x, y, w)
            reference = reference_drop(board, name, rotation, x, y, w)
            check((move is None) == (reference is None), 'New-piece independent collision feasibility')
            if move:
                check((move.board, move.layers, move.candidate.z) == reference, 'New-piece independent landing and clearing')
                check(sum(mask.bit_count() for mask in move.board) == sum(mask.bit_count() for mask in board)
                      + PIECE_SIZES[name] - AREA * move.layers, 'Variable-volume mass conservation')
                check(math.isclose(move.score, network_forward(move.heights, move.holes, move.layers)[-1], abs_tol=1e-8),
                      'Variable-volume incremental sigmoid equivalence')
            collision_cases += 1
        for rotation in sorted({0, len(SHAPES[name]) // 2, len(SHAPES[name]) - 1}):
            shape = SHAPES[name][rotation]
            x, y, w = 4, 5, 6
            board = [1] * AREA
            for dx, dy, dz, dw in shape.cells:
                if dz == 0:
                    board[column_index(x + dx, y + dy, w + dw)] = 0
            board = tuple(board)
            move = make_move(board, name, rotation, x, y, w)
            check(move is not None and move.candidate.z == 0 and move.layers == 1, 'New shape completes a hyperlayer')
            check((move.board, move.layers, move.candidate.z) == reference_drop(board, name, rotation, x, y, w),
                  'New-piece clear matches independent compaction')
            check(sum(mask.bit_count() for mask in move.board) == sum(mask.bit_count() for mask in board)
                  + PIECE_SIZES[name] - AREA, 'New-piece clear mass conservation')
            locked = (LockedPiece(1, name, placed_cells(move)),)
            compacted = compact_locked_pieces(locked, 1)
            remaining = tuple(cell for piece in compacted for cell in piece.cells)
            expected = tuple((cx, cy, cz - 1, cw) for cx, cy, cz, cw in placed_cells(move) if cz != 0)
            check(remaining == expected, 'Hyperlayer compaction preserves identity and W position')
            clear_cases += 1
        empty = make_move(EMPTY_BOARD, name, 0, 0, 0, 0)
        good = placed_cells(empty)
        for invalid in (good[:-1], good + (good[0],), tuple((cx + (2 if i == 0 else 0), cy, cz, cw)
                       for i, (cx, cy, cz, cw) in enumerate(good))):
            try:
                merge_rigid_piece(EMPTY_BOARD, invalid, name)
            except ValueError:
                check(True, 'Reject incomplete, duplicated or deformed registered shape')
            else:
                check(False, 'Malformed registered shape accepted')
        sx, sy, _, sw = SHAPES[name][0].extents
        for origin in ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (WIDTH - sx + 1, 0, 0),
                       (0, DEPTH - sy + 1, 0), (0, 0, FOURTH - sw + 1)):
            check(make_move(EMPTY_BOARD, name, 0, *origin) is None, 'Reject new-piece out-of-bounds origins')
    board = [3] * AREA
    for x, y, z, w in PIECE_CELLS['HYPER']:
        board[column_index(x, y, w)] &= ~(1 << z)
    move = make_move(tuple(board), 'HYPER', 0, 0, 0, 0)
    check(move is not None and move.layers == 2 and move.board == EMPTY_BOARD, 'HYPER completes two full hyperlayers')
    check((move.board, move.layers, move.candidate.z) == reference_drop(tuple(board), 'HYPER', 0, 0, 0, 0),
          'Independent double hyperlayer compaction')
    for count in range(HEIGHT + 1):
        check(clear_points(count) >= 0, 'Score supports any board-valid clear count')
    game = Game(31, '4d')
    for _ in range(len(NEW_PIECE_CELLS) * 2):
        decision = choose_move(game.board, game.current, game.next_piece, 2, 2, 256, game.pieces)
        check(decision is not None, 'New-only autoplay finds a legal move')
        game.commit(decision.move)
        check(game.cells_placed == sum(PIECE_SIZES[name] * count for name, count in game.piece_counts.items()),
              'Counters use actual piece volume')
        check(sum(mask.bit_count() for mask in game.board) == game.cells_placed - AREA * game.layers,
              'New-only game conserves cells')
    check(all(game.piece_counts[name] == 2 for name in NEW_PIECE_CELLS), 'Autoplay receives every new piece twice')
    check(not any(game.piece_counts[name] for name in CLASSIC_PIECE_CELLS), 'New-only mode excludes classic shapes')
    game.reset()
    check(game.cells_placed == game.pieces == 0 and not any(game.piece_counts.values()), 'Reset clears volume counters')
    return {'passed': True, 'assertions': checks, 'new_pieces': len(NEW_PIECE_CELLS),
            'catalog_has_no_rotation_duplicates': True, 'all_new_shapes_affine_dimension': 4,
            'independent_new_piece_drop_cases': collision_cases, 'new_piece_hyperlayer_clear_cases': clear_cases + 1,
            'wall_seconds': round(time.perf_counter() - started, 3)}


def run_self_tests() -> dict:
    rng, checks = random.Random(303), 0
    started = time.perf_counter()
    def check(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1
    check(len(EMPTY_BOARD) == 4096 and AREA * HEIGHT == 65536, '4D board size')
    check(INPUT_COUNT == 4098 and len(NETWORK) == 23041, 'Network dimensions')
    check(sum(len(n.edges) for n in NETWORK) == 73218, 'Connection count')
    check(len(NEIGHBORS) == 11520 == len(set(NEIGHBORS)), '3D surface adjacency')
    check(all(n.activation == 'sigmoid' for n in NETWORK[:-1]), 'Only sigmoid hidden units')
    check(NETWORK[-1].activation == 'linear', 'Linear output')
    check(sigmoid(-1000) == 0 and sigmoid(1000) == 1, 'Stable sigmoid')
    for i, node in enumerate(NETWORK):
        check(all(0 <= source < INPUT_COUNT + i for source, _ in node.edges), 'Acyclic graph')
    for i in range(AREA):
        check(column_index(*column_coordinates(i)) == i, 'Index round trip')
    for a, b in NEIGHBORS:
        check(sum(abs(x - y) for x, y in zip(column_coordinates(a), column_coordinates(b))) == 1, 'True 3D column neighbors')
    for _ in range(30):
        heights = tuple(rng.randrange(HEIGHT + 1) for _ in range(AREA))
        holes, layers = rng.randrange(AREA * (HEIGHT - 1)), rng.randrange(5)
        values = network_forward(heights, holes, layers)
        check(math.isclose(values[-1], network_score(heights, holes, layers), abs_tol=1e-8), 'Literal graph equivalence')
        check(all(0 <= value <= 1 for value in values[INPUT_COUNT:-1]), 'Sigmoid range')
    check(abs(network_forward((0,) * AREA, 0, 0)[-1]) < 1e-8, 'Flat baseline')
    rotation_cases = 0
    for piece, shapes in SHAPES.items():
        reference_distances = sorted(sum((a[k] - b[k]) ** 2 for k in range(4)) for a, b in itertools.combinations(shapes[0].cells, 2))
        check(len({s.cells for s in shapes}) == len(shapes), 'Unique orientations')
        for r, shape in enumerate(shapes):
            check(len(set(shape.cells)) == PIECE_SIZES[piece], 'Complete rigid shape')
            check(sorted(sum((a[k] - b[k]) ** 2 for k in range(4)) for a, b in itertools.combinations(shape.cells, 2)) == reference_distances, 'Rotation preserves distances')
            for plane in range(6):
                cells = shape.cells
                for _ in range(4):
                    cells = rotate_cells(cells, plane)
                check(cells == shape.cells, 'Quarter-turn cycle')
                check(rotate_cells(rotate_cells(shape.cells, plane), plane, -1) == shape.cells, 'Inverse rotation')
                check(rotate_cells(shape.cells, plane) in SHAPE_INDEX[piece], 'Rotation closure')
                rotation_cases += 1
            candidate = score_candidate(evaluate_board(EMPTY_BOARD), piece, r, 2, 3, 4)
            move = materialize_move(EMPTY_BOARD, candidate)
            for progress in (0, .1, .5, .9, 1):
                cells = animated_piece_cells(move, progress)
                distances = sorted(sum((a[k] - b[k]) ** 2 for k in range(4)) for a, b in itertools.combinations(cells, 2))
                check(all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(distances, reference_distances)), 'Rigid 4D animation')
            check(animated_piece_cells(move, 1) == placed_cells(move), 'Animation landing')
    d_cells = ((0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, -1, 0))
    check(normalize(d_cells) in SHAPE_INDEX['C'], 'C and D equivalent under proper 4D rotations')
    for name in PIECE_NAMES:
        specs = list(specifications(name, 2048, 77))
        check(len(specs) == len(set(specs)) == 2048, 'Unique bounded samples')
        check(set(r for r, *_ in specs) == set(range(len(SHAPES[name]))), 'All orientations sampled')
        check(set(w for r, x, y, w in specs) == set(range(max(FOURTH - shape.extents[3] + 1 for shape in SHAPES[name]))), 'All feasible W origins sampled')
    drop_cases = 0
    for iteration in range(80):
        board = tuple(rng.getrandbits(rng.randrange(HEIGHT + 1)) if rng.random() < .18 else 0 for _ in range(AREA))
        piece = rng.choice(PIECE_NAMES)
        r = rng.randrange(len(SHAPES[piece]))
        sx, sy, _, sw = SHAPES[piece][r].extents
        x, y, w = rng.randrange(WIDTH - sx + 1), rng.randrange(DEPTH - sy + 1), rng.randrange(FOURTH - sw + 1)
        move = make_move(board, piece, r, x, y, w)
        reference = reference_drop(board, piece, r, x, y, w)
        check((move is None) == (reference is None), 'Independent collision feasibility')
        if move:
            check((move.board, move.layers, move.candidate.z) == reference, 'Independent 4D drop and compaction')
            check(sum(m.bit_count() for m in move.board) == sum(m.bit_count() for m in board) + PIECE_SIZES[piece] - AREA * move.layers, 'Mass conservation')
            check(math.isclose(move.score, network_forward(move.heights, move.holes, move.layers)[-1], abs_tol=1e-8), 'Incremental sigmoid score')
        drop_cases += 1
    row = [0] * AREA
    for y in range(16):
        for x in range(16):
            row[column_index(x, y, 0)] = 1
    check(clear_layers(tuple(row))[1] == 0, 'A 3D slice plane is not a 4D hyperlayer')
    check(clear_layers((1,) * AREA) == (EMPTY_BOARD, 1), 'Complete hyperlayer clear')
    check(clear_layers((0b10101,) * AREA) == (EMPTY_BOARD, 3), 'Nonadjacent hyperlayer clears')
    for count in range(1, 5):
        board = [(1 << count) - 1] * AREA
        board[column_index(0, 0, 0)] = 0
        rotation = next(r for r, s in enumerate(SHAPES['I']) if s.extents[2] == 4)
        move = make_move(tuple(board), 'I', rotation, 0, 0, 0)
        check(move is not None and move.layers == count, 'Rigid piece completes hyperlayers')
        check(sum(m.bit_count() for m in move.board) == 4 - count, 'Hyperlayer clear cell conservation')
        reference = reference_drop(tuple(board), 'I', rotation, 0, 0, 0)
        check((move.board, move.layers, move.candidate.z) == reference, 'Independent hyperlayer reference')
    row = [0] * AREA
    row[column_index(0, 0, 0)] = 1 << 7
    rotation = next(r for r, s in enumerate(SHAPES['I']) if s.extents[3] == 4)
    move = make_move(tuple(row), 'I', rotation, 0, 0, 0)
    check(move is not None and all(z == 8 for _, _, z, _ in placed_cells(move)), 'Rigid bridge across W with one supported cell')
    move_other = make_move(tuple(row), 'I', 0, 0, 0, 1)
    check(move_other is not None and move_other.candidate.z == 0, 'Separate W cells do not collide')
    high = ((1 << HEIGHT) - 1,) * AREA
    check(not has_legal_move(high, 'O'), 'Full 4D board top-out')
    check(choose_move(high, 'O', 'I', candidates=32) is None, 'Sample fallback proves top-out')
    locked = (LockedPiece(1, 'I', ((0, 0, 0, 0), (0, 0, 1, 1), (0, 0, 2, 2), (0, 0, 3, 3))),)
    compacted = compact_locked_pieces(locked, 0b0101)
    check(compacted[0].cells == ((0, 0, 0, 1), (0, 0, 1, 3)), 'Only Z compacts; W identity stays fixed')
    for seed in (3, 7):
        game = Game(seed)
        for _ in range(12):
            decision = choose_move(game.board, game.current, game.next_piece, 2, 2, 256, game.pieces)
            check(decision is not None, 'Seeded decision exists')
            move = decision.move
            game.commit(move)
            check(sum(mask.bit_count() for mask in game.board) == game.cells_placed - AREA * game.layers, 'Game conservation')
            rebuilt = [0] * AREA
            for p in game.locked_pieces:
                for x, y, z, w in p.cells:
                    rebuilt[column_index(x, y, w)] |= 1 << z
            check(tuple(rebuilt) == game.board, 'Persistent piece identity')
        try:
            game.commit(move)
        except ValueError:
            check(True, 'Reject stale move')
        else:
            check(False, 'Stale move accepted')
    context = evaluate_board(EMPTY_BOARD)
    salt, limit = 91, 32
    first = sorted((score_candidate(context, 'T', *spec) for spec in specifications('T', limit, salt)), key=candidate_key, reverse=True)
    brute = -math.inf
    for c in first[:3]:
        board = materialize_move(EMPTY_BOARD, c).board
        ctx = evaluate_board(board)
        best = max(score_candidate(ctx, 'O', *spec).score for spec in specifications('O', limit, salt + 65537))
        brute = max(brute, LAYERS_WEIGHT * c.layers + best)
    decision = choose_move(EMPTY_BOARD, 'T', 'O', 2, 3, limit, salt)
    check(math.isclose(decision.search_score, brute, abs_tol=1e-8), 'Beam objective equals independent sampled enumeration')
    a = choose_move(EMPTY_BOARD, 'T', 'O', 1, 2, 128, 9)
    b = choose_move(EMPTY_BOARD, 'T', 'O', 1, 2, 128, 9)
    check(a.move == b.move, 'Deterministic sampling and policy')
    piece_tests = run_piece_tests()
    return {'passed': True, 'assertions': checks + piece_tests['assertions'], 'piece_tests': piece_tests, 'rotation_checks': rotation_cases,
            'unique_orientations': sum(len(s) for s in SHAPES.values()), 'independent_drop_cases': drop_cases,
            'literal_network_cases': 30, 'wall_seconds': round(time.perf_counter() - started, 3)}


def run_game(seed: int, pieces: int, depth: int, beam: int, candidates: int, policy: str = 'network',
             piece_set: str = 'expanded', target: int = 1, stop_after_clear: bool = False) -> dict:
    if policy not in ('network', 'random', 'hyperlayer'):
        raise ValueError('Unknown policy.')
    target_layer_status(EMPTY_BOARD, target)
    game, timings = Game(seed, piece_set), []
    first_clear = None
    active_target, clear_events = target, []
    stop_reason = 'piece_limit'
    rng = random.Random(seed ^ 0x5EEDBEEF)
    started = time.perf_counter()
    for _ in range(pieces):
        if game.game_over:
            break
        if policy in ('network', 'hyperlayer'):
            if policy == 'hyperlayer':
                decision = choose_hyperlayer_move(game.board, game.current, active_target, candidates, game.pieces)
            else:
                decision = choose_move(game.board, game.current, game.next_piece, depth, beam, candidates,
                                       seed * 8191 + game.pieces)
            if decision is None:
                game.game_over = not has_legal_move(game.board, game.current)
                stop_reason = 'top_out' if game.game_over else 'planner_blocked'
                break
            timings.append(decision.milliseconds)
            move = decision.move
        else:
            options = []
            context = evaluate_board(game.board)
            for spec in specifications(game.current, 512, seed * 8191 + game.pieces):
                candidate = score_candidate(context, game.current, *spec)
                if candidate:
                    options.append(candidate)
            if not options:
                for spec in specifications(game.current, 0):
                    candidate = score_candidate(context, game.current, *spec)
                    if candidate:
                        options.append(candidate)
                        break
            if not options:
                game.game_over = True
                break
            move = materialize_move(game.board, rng.choice(options))
        game.commit(move)
        if sum(v.bit_count() for v in game.board) != game.cells_placed - game.layers * AREA:
            raise AssertionError('Game mass conservation failed.')
        if move.layers:
            if first_clear is None:
                first_clear = game.pieces
            clear_events.append({'piece': game.pieces, 'layers': move.layers,
                                 'total_layers': game.layers, 'target': active_target if policy == 'hyperlayer' else None})
            if stop_after_clear:
                stop_reason = 'first_clear'
                break
            if policy == 'hyperlayer':
                following_target = next_hyperlayer_target(game.board, active_target)
                if following_target is not None:
                    active_target = following_target
    if game.game_over and stop_reason != 'first_clear':
        stop_reason = 'top_out'
    result = game.summary()
    result.update(backend=BACKEND, policy=policy, capped=not game.game_over and game.pieces == pieces,
                  piece_limit=pieces, depth=depth if policy == 'network' else 1 if policy == 'hyperlayer' else None,
                  beam=beam if policy == 'network' else None, candidates=candidates if policy != 'random' else 512,
                  target=target if policy == 'hyperlayer' else None, active_target=active_target if policy == 'hyperlayer' else None,
                  first_clear_piece=first_clear, clear_events=clear_events, stop_reason=stop_reason,
                  mean_decision_ms=round(statistics.fmean(timings), 3) if timings else None,
                  wall_seconds=round(time.perf_counter() - started, 3))
    return result


def launch_demo(args) -> int:
    try:
        import tkinter as tk
        root = tk.Tk()
    except (ImportError, RuntimeError) as error:
        print(f'Tkinter is unavailable: {error}. Use --test for headless validation.', file=sys.stderr)
        return 2
    except Exception as error:
        print(f'A desktop display is required: {error}. Use --test or --benchmark.', file=sys.stderr)
        return 2
    app = TetrisApp(root, args.seed, args.depth, args.beam, args.candidates, args.piece_set, args.policy, args.target, args.stop_after_clear,
                    args.ui_fps, not args.no_frameskip)
    if args.gui_smoke:
        def finish():
            result = {'passed': not app.errors, 'callback_errors': app.errors,
                      'selected_drop': app.planned is not None, 'visible_network_nodes': len(app.network.positions)}
            print(json.dumps(result, indent=2))
            app.close()
        root.after(2400, finish)
    root.mainloop()
    return 1 if app.errors else 0


def start_benchmark_logging(args):
    if args.no_benchmark_log:
        return None
    try:
        from tetris_benchmark_logging import BenchmarkLog
    except ImportError as error:
        print(f'Benchmark logging unavailable: {error}', file=sys.stderr)
        return None
    settings = {key: getattr(args, key) for key in ('seed', 'piece_set', 'policy', 'target', 'depth',
                 'beam', 'candidates', 'stop_after_clear', 'backend')}
    settings['mode'] = 'benchmark' if args.benchmark else 'gui'
    settings.update(ui_fps=args.ui_fps, frameskip=not args.no_frameskip)
    if args.benchmark:
        settings.update(pieces=args.pieces, games=args.games)
    recorder = BenchmarkLog(sys.modules[__name__], args.benchmark_log, settings).install()
    if recorder.enabled:
        print(f'Benchmark log: {recorder.path}', file=sys.stderr)
    return recorder


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Rigid 4D Tetris with a fixed sparse sigmoid evaluator and live activation slices.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--test', action='store_true')
    mode.add_argument('--catalog', action='store_true', help='Export names, cells, sizes and rotations of every piece.')
    mode.add_argument('--benchmark', action='store_true')
    mode.add_argument('--network', action='store_true', help='Export the compact shared-sigmoid architecture.')
    mode.add_argument('--network-expanded', action='store_true', help='Expand every logical neuron and connection for inspection.')
    parser.add_argument('--backend', choices=('auto', 'python', 'numpy', 'native'), default='auto',
                        help='Auto uses a locally built native scorer, then NumPy, then Python. Build native with tools/build_native.py.')
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument('--weights', type=Path,
                         default=Path(__file__).resolve().parent / 'models' / 'trained-sigmoid.json',
                         help='Load a validated JSON checkpoint; defaults to models/trained-sigmoid.json beside this script.')
    weights.add_argument('--baseline', action='store_true', help='Use the original untrained weights instead of a checkpoint.')
    parser.add_argument('--policy', choices=('network', 'hyperlayer'), default='hyperlayer',
                        help='Default: continuous hyperlayer search with depth 1 and automatic retargeting after clears; network uses neural-only search.')
    parser.add_argument('--target', type=int, choices=range(HEIGHT), default=1, help='Protected hyperlayer Z for --policy hyperlayer.')
    parser.add_argument('--stop-after-clear', action='store_true', help='Pause autoplay or stop benchmarks after the first clear; disabled by default.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--piece-set', choices=tuple(PIECE_SETS), default='expanded',
                        help='expanded: all 15 pieces; classic: original 7; 4d: only the 8 new shapes.')
    parser.add_argument('--depth', type=int, choices=(1, 2), default=2)
    parser.add_argument('--beam', type=int, default=DEFAULT_BEAM)
    parser.add_argument('--candidates', type=int, default=DEFAULT_CANDIDATES,
                        help='Candidate budget per board, stratified by orientation; 0 scans every origin/orientation.')
    parser.add_argument('--pieces', type=int, default=100)
    parser.add_argument('--games', type=int, default=3)
    parser.add_argument('--compare', action='store_true', help='Also test a random policy over sampled legal placements.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gui-smoke', action='store_true')
    parser.add_argument('--ui-fps', type=int, choices=range(1, 121), default=30, metavar='1..120',
                        help='Maximum scheduled UI redraw rate; default: 30. Does not change search or game rules.')
    parser.add_argument('--no-frameskip', action='store_true',
                        help='Disable adaptive UI pacing for rendering comparisons; default: enabled.')
    logs = parser.add_mutually_exclusive_group()
    logs.add_argument('--benchmark-log', type=Path, help='Write local benchmark JSONL here; default: a unique file under logs/. Existing files are not overwritten.')
    logs.add_argument('--no-benchmark-log', action='store_true', help='Disable all benchmark logging and instrumentation for clean timing comparisons.')
    args = parser.parse_args(argv)
    if args.beam < 1 or args.candidates < 0 or args.pieces < 1 or args.games < 1:
        parser.error('beam, pieces and games must be positive; candidates must be nonnegative.')
    recorder = None
    try:
        configure_backend(args.backend)
        if args.baseline:
            configure_network(BASELINE_PARAMETERS, {'method': 'none', 'description': 'Untrained inherited heuristic coefficients.'})
        else:
            load_checkpoint(args.weights)
        if args.test:
            result = run_self_tests()
        elif args.catalog:
            result = piece_catalog()
        elif args.network or args.network_expanded:
            result = network_metadata(args.network_expanded)
        elif args.benchmark:
            recorder = start_benchmark_logging(args)
            runs = []
            for policy in ((args.policy, 'random') if args.compare else (args.policy,)):
                for i in range(args.games):
                    run = run_game(args.seed + i, args.pieces, args.depth, args.beam, args.candidates, policy, args.piece_set, args.target, args.stop_after_clear)
                    runs.append(run)
                    print(f'{policy} seed={run["seed"]} pieces={run["pieces"]} hyperlayers={run["hyperlayers"]}', file=sys.stderr, flush=True)
            result = {'network': network_metadata(), 'runs': runs,
                      'note': 'Short capped runs do not establish long-term playing strength. A clear consumes 4096 cells; piece sizes vary from 4 to 16.'}
        else:
            recorder = start_benchmark_logging(args)
            return launch_demo(args)
        text = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(text + '\n', encoding='utf-8')
        print(text)
        return 0
    except (ValueError, AssertionError, OSError) as error:
        if recorder is not None:
            recorder.emit('error', phase='application', exception=type(error).__name__)
        print(f'Error: {error}', file=sys.stderr)
        return 1
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == '__main__':
    raise SystemExit(main())
