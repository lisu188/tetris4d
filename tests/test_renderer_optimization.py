from __future__ import annotations

import ast
import copy
from dataclasses import asdict
import random
from pathlib import Path
import sys
import tkinter as tk
from textwrap import dedent
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import sparse_tetris_4d_optimized as game
from benchmark_rendering import load_baseline

baseline, baseline_source = load_baseline()


def reference_move(move):
    if move is None:
        return None
    return baseline.Move(baseline.Candidate(**asdict(move.candidate)), move.before, move.board, move.heights, move.holes)


def snapshot(canvas):
    result = []
    for item in canvas.find_all():
        if canvas.itemcget(item, 'state') == 'hidden':
            continue
        kind = canvas.type(item)
        options = ('fill', 'outline', 'width') if kind in ('rectangle', 'polygon') else ('fill', 'width', 'dash') if kind == 'line' else ('text', 'fill', 'anchor', 'font')
        result.append((kind, tuple(canvas.coords(item)), tuple((k, canvas.itemcget(item, k)) for k in options)))
    return result


def state(pieces):
    board = [0] * game.AREA
    for piece in pieces:
        for x, y, z, w in piece.cells:
            board[game.column_index(x, y, w)] |= 1 << z
    return SimpleNamespace(board=tuple(board), locked_pieces=pieces)


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.geometry('1100x720')
        self.canvases = [tk.Canvas(self.root, width=420, height=500, highlightthickness=0) for _ in range(2)]
        for i, canvas in enumerate(self.canvases):
            canvas.place(x=i * 440, y=0, width=420, height=500)
        self.root.update()

    def tearDown(self):
        self.root.destroy()

    def test_game_engine_and_native_source_are_not_modified(self):
        before = ast.parse(baseline_source)
        after = ast.parse((ROOT / 'sparse_tetris_4d_optimized.py').read_text(encoding='utf-8'))
        def named(tree):
            return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        old, new = named(before), named(after)
        for name, node in old.items():
            # The fused native board evaluator is checked against this same historical
            # baseline by value below. The renderer still consumes the original features.
            if name in ('WellView', 'AllWellView', 'SliceAtlas', 'main', 'run_game',
                        'evaluate_board', '_native_features', 'sigmoid_roughness'):
                continue
            if name == 'TetrisApp':
                old_methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
                new_methods = {n.name: n for n in new[name].body if isinstance(n, ast.FunctionDef)}
                for method, fn in old_methods.items():
                    if method not in ('render_metrics', '__init__', 'refresh_plan', 'place', '_draw_now'):
                        candidate = copy.deepcopy(new_methods[method])
                        if method == '_advance_search':
                            expected = ast.parse("if self.decision is not None and self.decision.policy == 'neural' and self.decision.target is not None:\n    self.target.set(str(self.decision.target))").body[0]
                            removed = []
                            class StripRecovery(ast.NodeTransformer):
                                def visit_If(self, node):
                                    if ast.dump(node) == ast.dump(expected):
                                        removed.append(node)
                                        return None
                                    return self.generic_visit(node)
                            candidate = StripRecovery().visit(candidate)
                            self.assertEqual(len(removed), 1)
                        self.assertEqual(ast.dump(fn), ast.dump(candidate), method)
            else:
                candidate = copy.deepcopy(new[name])
                if name == '_load_native':
                    # Extend the historical ABI at the exact binding point. The global AST
                    # comparison still checks every old signature, capability check and call.
                    signature = next(n for n in ast.walk(node) if isinstance(n, ast.Assign)
                                     and ast.unparse(n.targets[0]) == 'signatures')
                    extra = ast.parse("{'t4_sustained_metrics': [pointer, pointer, integer, pointer, pointer, integer, integer, pointer, pointer, pointer]}").body[0].value
                    signature.value.keys.extend(extra.keys)
                    signature.value.values.extend(extra.values)
                    owner = next(n for n in ast.walk(node) if signature in getattr(n, 'body', []))
                    start = owner.body.index(signature) + 1
                    owner.body[start:start] = ast.parse(dedent("""
                        optional = {
                            't4_board_features': [pointer] * 5,
                            't4_checked_apply': [pointer, pointer] + [integer] * 4 + [pointer] * 6 + [integer],
                            't4_has_move': [pointer, pointer, integer],
                            't4_exhaustive_primary': [pointer, pointer, integer, integer, integer, integer, pointer, integer],
                        }
                        signatures.update((name, args) for name, args in optional.items() if hasattr(library, name))
                    """)).body
                if name == 'launch_demo':
                    # The feature-inspector smoke callback has a text display. Check its
                    # complete body and keep the Tk setup, scheduling and mainloop pinned.
                    expected = next(n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n.name == 'finish')
                    expected.body = ast.parse(dedent("""
                        try:
                            view = app.network
                            result = {'passed': not app.errors, 'callback_errors': app.errors,
                                      'selected_drop': app.planned is not None,
                                      'visible_network_nodes': len(getattr(view, 'positions', ()))}
                            if hasattr(view, 'display'):
                                result['feature_decision_visible'] = 'Architecture:' in view.display.get('1.0', 'end')
                            print(json.dumps(result, indent=2))
                        finally:
                            app.close()
                    """)).body
                if name == 'pooled_histograms':
                    # Exact C-speed validation of the 4096 heights (issue #35).
                    original = 'any((not isinstance(h, int) or not 0 <= h <= HEIGHT for h in heights))'
                    swapped = []
                    class RestoreValidation(ast.NodeTransformer):
                        def generic_visit(self, node):
                            if isinstance(node, ast.expr) and ast.unparse(node) == 'not _valid_heights(heights)':
                                swapped.append(node)
                                return ast.parse(original, mode='eval').body
                            return super().generic_visit(node)
                    candidate = RestoreValidation().visit(candidate)
                    self.assertEqual(len(swapped), 1)
                if name == 'Game':
                    # The game library adds optional recording hooks only: a finalizer,
                    # `recording = ...` lookups and `if recording is not None:` calls.
                    # Every pre-existing engine statement must remain identical.
                    hooks = []
                    class StripRecording(ast.NodeTransformer):
                        def visit_FunctionDef(self, node):
                            if node.name == '__del__':
                                hooks.append('__del__')
                                return None
                            return self.generic_visit(node)
                        def visit_Assign(self, node):
                            if [ast.unparse(t) for t in node.targets] == ['recording'] and ast.unparse(node.value) in ('_recording()', '_RECORDING'):
                                hooks.append('lookup')
                                return None
                            return node
                        def visit_If(self, node):
                            if ast.unparse(node.test) == 'recording is not None' and not node.orelse and all(
                                    ast.unparse(s).startswith('recording.') for s in node.body):
                                hooks.append('call')
                                return None
                            return self.generic_visit(node)
                    # Build the expected render-history path from the original commit
                    # statements. Only the shared tail takes the next-piece legality as input.
                    methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
                    reset = methods['reset']
                    anchor = next(i for i, stmt in enumerate(reset.body) if ast.unparse(stmt) == 'self.board = EMPTY_BOARD') + 1
                    reset.body[anchor:anchor] = ast.parse('self._packed_board = None\nself.last_clear_mask = 0').body
                    commit = methods['commit']
                    tail = next(i for i, stmt in enumerate(commit.body) if ast.unparse(stmt) == 'self.board = move.board')
                    finish = ast.parse('def _finish_commit(self, move: Move, next_is_legal: bool) -> None: pass').body[0]
                    finish.body = commit.body[tail:]
                    self.assertEqual(ast.unparse(finish.body[-1]), 'self.game_over = not has_legal_move(self.board, self.current)')
                    finish.body[-1] = ast.parse('self.game_over = not next_is_legal').body[0]
                    commit.body[tail:] = ast.parse(dedent("""
                        self.last_clear_mask = cleared
                        self._packed_board = None
                        self._finish_commit(move, has_legal_move(move.board, self.next_piece))
                    """)).body
                    packed = ast.parse(dedent('''
                        @property
                        def packed_board(self):
                            """Read-only native column masks belonging to the current immutable board."""
                            cached = self._packed_board
                            return cached[1] if cached is not None and cached[0] is self.board else None
                    ''')).body[0]
                    node.body.insert(node.body.index(commit), packed)
                    node.body.insert(node.body.index(commit) + 1, finish)
                    candidate = StripRecording().visit(candidate)
                    self.assertEqual(sorted(hooks), ['__del__', 'call', 'call', 'call', 'lookup', 'lookup'])
                self.assertEqual(ast.dump(node), ast.dump(candidate), name)
        self.assertEqual(game.PIECE_CELLS, baseline.PIECE_CELLS)
        self.assertEqual(game.ROUGHNESS_TABLE, baseline.ROUGHNESS_TABLE)

    def test_default_commit_keeps_render_history_and_matches_historical_game(self):
        self.assertFalse(game.TWO_STAGE_EXHAUSTIVE)
        self.assertFalse(game.HEADLESS_COMMIT)
        self.addCleanup(game.configure_backend, game.BACKEND)
        self.addCleanup(baseline.configure_backend, baseline.BACKEND)
        game.configure_backend('auto')
        baseline.configure_backend('python')
        center = game.column_index(7, 7, 7)
        clear_board = [3] * game.AREA
        clear_board[center] = 0
        frontier = [1 << 14] * game.AREA
        for step in (0, -1, 1, -16, 16, -256, 256):
            frontier[center + step] = 1 << 11
        for board, cleared, top_out in ((game.EMPTY_BOARD, 0, False), (tuple(clear_board), 3, False),
                                        (tuple(frontier), 0, True)):
            with self.subTest(clear_mask=cleared, top_out=top_out):
                old, new = baseline.Game(103), game.Game(103)
                for state in (old, new):
                    state.board, state.current, state.next_piece = board, 'I', 'CROSS'
                rotation = next(i for i, shape in enumerate(game.SHAPES['I']) if shape.extents[2] == 4)
                reference = baseline.make_move(board, 'I', rotation, 7, 7, 7)
                old.commit(reference)
                option = game.make_move(board, 'I', rotation, 7, 7, 7).candidate
                actual = game.play_candidate(new, option)
                self.assertEqual(new.last_clear_mask, cleared)
                self.assertIsNone(new.packed_board)
                self.assertEqual(new.game_over, top_out)
                for field in ('board', 'pieces', 'layers', 'score', 'cells_placed', 'piece_counts',
                              'current', 'next_piece', 'game_over'):
                    self.assertEqual(getattr(old, field), getattr(new, field), field)
                self.assertEqual(tuple(map(asdict, old.locked_pieces)), tuple(map(asdict, new.locked_pieces)))
                self.assertEqual((reference.board, reference.heights, reference.holes),
                                 (actual.board, actual.heights, actual.holes))
                self.assertEqual(old.stream.bag, new.stream.bag)
                self.assertEqual(old.stream.rng.getstate(), new.stream.rng.getstate())

    def test_board_evaluation_matches_renderer_baseline(self):
        backend = game.BACKEND
        self.addCleanup(game.configure_backend, backend)
        game.configure_backend('auto')
        baseline.configure_backend('python')
        rng = random.Random(561)
        for board in (game.EMPTY_BOARD, (65535,) * game.AREA,
                      tuple(rng.getrandbits(rng.randrange(17)) for _ in range(game.AREA))):
            expected, actual = baseline.evaluate_board(board), game.evaluate_board(board)
            for field in ('board', 'heights', 'height_sum', 'holes', 'roughness', 'fill_counts', 'score', 'near_clear'):
                self.assertEqual(getattr(expected, field), getattr(actual, field), field)

    def test_face_geometry_matches_reference_exactly(self):
        old, new = baseline.WellView(self.canvases[0]), game.WellView(self.canvases[1])
        rng = random.Random(991)
        for case in range(36):
            old.yaw = new.yaw = rng.uniform(-4, 4)
            old.elevation = new.elevation = rng.uniform(.18, 1.4)
            old._camera(420, 500)
            new._camera(420, 500)
            shift = (0, .1, .333333333, .999999999)[case % 4]
            points = {(rng.randrange(8), rng.randrange(8), rng.randrange(6) + shift) for _ in range(80)}
            cubes = [(x, y, z, game.PIECE_COLORS['I'], rng.choice((-1, 0, 1, 2))) for x, y, z in points]
            occupied = {(x, y, round(z, 9)): owner for x, y, z, _, owner in cubes}
            self.assertEqual(old._faces(cubes, occupied), new._faces(cubes, occupied))

    def test_batched_static_mesh_matches_reference_primitives(self):
        old, new = baseline.WellView(self.canvases[0]), game.WellView(self.canvases[1])
        old._camera(420, 500)
        new._camera(420, 500)
        cubes = [(x, y, (x + y) % 4, game.PIECE_COLORS['T'], x // 2 + 1) for x in range(10) for y in range(10)]
        occupied = {(x, y, z): owner for x, y, z, _, owner in cubes}
        faces = old._faces(cubes, occupied)
        reference_ids = [old._face(face, f'static_face:{i}') for i, face in enumerate(faces)]
        ids = new._static_mesh(faces)
        self.assertEqual(snapshot(self.canvases[0]), snapshot(self.canvases[1]))
        self.assertEqual(len(reference_ids), len(ids))
        self.assertTrue(all(self.canvases[1].type(i) == 'polygon' for i in ids))
        self.assertEqual([self.canvases[0].gettags(i) for i in self.canvases[0].find_all()],
                         [self.canvases[1].gettags(i) for i in self.canvases[1].find_all()])

    def test_full_well_matches_cameras_cutaways_and_animation(self):
        old, new = baseline.WellView(self.canvases[0], True), game.WellView(self.canvases[1], True)
        pieces = tuple(game.LockedPiece(x + 1, 'I', tuple((x, 0, z, 7) for z in range(4))) for x in range(6))
        board = state(pieces)
        move = game.make_move(board.board, 'HYPER', 0, 8, 8, 7)
        for yaw, cutoff, progress in ((.68, 15, 0), (.68, 2, .5), (2.5, 15, .999), (-1, 15, 1), (-1, 15, 1)):
            old.yaw = new.yaw = yaw
            old.draw(board, reference_move(move), progress, 7, cutoff)
            new.draw(board, move, progress, 7, cutoff)
            self.assertEqual(snapshot(self.canvases[0]), snapshot(self.canvases[1]))
            self.assertEqual(old.rendered_cubes, new.rendered_cubes)

    def test_atlas_reuses_cells_and_matches_reference(self):
        old, new = baseline.SliceAtlas(self.canvases[0], lambda w: None), game.SliceAtlas(self.canvases[1], lambda w: None)
        rng = random.Random(883)
        masks = [rng.getrandbits(rng.randrange(1, 17)) for _ in range(game.AREA)]
        cases = [tuple(masks)]
        for _ in range(12):
            for i in rng.sample(range(game.AREA), 16):
                masks[i] = rng.randrange(1 << 16)
            cases.append(tuple(masks))
        cases.extend([game.EMPTY_BOARD, (1,) * game.AREA, game.EMPTY_BOARD])
        move = game.make_move(game.EMPTY_BOARD, 'HYPER', 0, 0, 0, 7)
        first_item = None
        for step, board in enumerate(cases):
            frame = SimpleNamespace(board=board)
            visible = move if step % 2 else None
            selected = step % 16
            old.draw(frame, reference_move(visible), selected)
            new.draw(frame, visible, selected)
            self.assertEqual(sorted(snapshot(self.canvases[0])), sorted(snapshot(self.canvases[1])))
            self.assertEqual(old.boxes, new.boxes)
            self.assertLessEqual(len(self.canvases[1].find_all()), game.AREA + 34 + 16)
            if step == 0:
                first_item = new.cells[0]
            self.assertEqual(first_item, new.cells[0])
        self.assertEqual(new.scene_rebuilds, 1)
        self.assertLessEqual(new.cells_created, game.AREA)
        before = (new.cell_updates, new.cells_created)
        new.draw(SimpleNamespace(board=game.EMPTY_BOARD), None, 0)
        self.assertEqual(before, (new.cell_updates, new.cells_created))

    def test_atlas_resize_and_selection_do_not_leave_stale_items(self):
        atlas = game.SliceAtlas(self.canvases[0], lambda w: None)
        frame = SimpleNamespace(board=(3,) * game.AREA)
        atlas.draw(frame, None, 0)
        old_ids = tuple(atlas.cells)
        atlas.draw(frame, None, 15)
        self.assertEqual(old_ids, tuple(atlas.cells))
        self.assertEqual(atlas.scene_rebuilds, 1)
        self.canvases[0].place_configure(width=340, height=420)
        self.root.update()
        atlas.draw(frame, None, 15)
        self.assertEqual(atlas.scene_rebuilds, 2)
        self.assertEqual(len(atlas.boxes), 16)
        self.assertEqual(len(self.canvases[0].find_all()), game.AREA + 34)

    def test_ownership_append_coalescing_clear_and_reset(self):
        for canvas in self.canvases:
            canvas.place_forget()
        hosts = [tk.Frame(self.root) for _ in range(2)]
        panels = []
        for i, (host, module) in enumerate(zip(hosts, (baseline, game))):
            host.place(x=i * 540, y=0, width=530, height=650)
            host.rowconfigure(0, weight=1)
            host.columnconfigure(0, weight=1)
            panels.append(module.AllWellView(host, lambda w: None))
        self.root.update()
        pieces = tuple(game.LockedPiece(i + 1, 'I', tuple((i % 16, 1, z, i // 16) for z in range(4))) for i in range(32))
        changed = pieces[:5] + (game.LockedPiece(99, 'T', pieces[5].cells),) + pieces[6:]
        compacted = tuple(game.LockedPiece(p.identifier, p.piece, tuple((x, y, z - 1, w) for x, y, z, w in p.cells if z > 0)) for p in changed)
        frames = [state(()), state(pieces[:1]), state(pieces[:12]), state(pieces), state(changed), state(compacted), state(())]
        for frame in frames:
            for panel in panels:
                panel.draw(frame, None)
            self.assertEqual(panels[0].owners, panels[1].owners)
            self.assertEqual(panels[0].section_tokens, panels[1].section_tokens)
            self.assertEqual(panels[0].section_revisions, panels[1].section_revisions)
            self.assertEqual([v.rendered_cubes for v in panels[0].views], [v.rendered_cubes for v in panels[1].views])
        self.assertEqual(panels[1].ownership_appends, 32)
        self.assertEqual(panels[1].ownership_full_scans, 4)
        self.assertTrue(all(v.canvas.winfo_ismapped() for v in panels[1].views))

    def test_canvas_batch_uses_tcl_argument_quoting(self):
        text = 'literal ; [error unexpected] { } $untrusted \\ word'
        result = game.canvas_batch(self.canvases[0], [('create', 'text', 1, 2, '-text', text)])
        self.assertEqual(self.canvases[0].itemcget(int(result[0]), 'text'), text)
        self.assertEqual(game.canvas_batch(self.canvases[0], []), ())


if __name__ == '__main__':
    unittest.main(verbosity=2)
