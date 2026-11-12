from __future__ import annotations

import ast
import copy
from dataclasses import asdict
import random
from pathlib import Path
import sys
import tkinter as tk
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
            if name in ('WellView', 'AllWellView', 'SliceAtlas', 'main', 'run_game'):
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
                self.assertEqual(ast.dump(node), ast.dump(new[name]), name)
        self.assertEqual(game.PIECE_CELLS, baseline.PIECE_CELLS)
        self.assertEqual(game.ROUGHNESS_TABLE, baseline.ROUGHNESS_TABLE)

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
