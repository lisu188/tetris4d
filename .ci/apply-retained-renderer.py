from pathlib import Path
import ast
import hashlib
import textwrap

path = Path('sparse_tetris_4d_optimized.py')
raw = path.read_bytes()
assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == '1be83b3959db35f588b6f925640f23ba5aadc366'
source = raw.decode('utf-8')

def replace(old, new):
    global source
    assert source.count(old) == 1, (old[:100], source.count(old))
    source = source.replace(old, new)

def method(cls, name, code):
    global source
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    lines[fn.lineno - 1:fn.end_lineno] = [textwrap.indent(textwrap.dedent(code).strip(), '    ') + '\n']
    source = ''.join(lines)

replace('\n\nclass WellView:\n', '''

FACE_RENDER_GEOMETRY = tuple(
    (normal, vertices, tuple(tuple(a[j] + vertices[(i + 1) % 4][j] - 1 if not normal[j] else 0
                                  for j in range(3)) for i, a in enumerate(vertices)))
    for normal, vertices in FACE_GEOMETRY)


def canvas_batch(canvas, commands):
    if not commands:
        return ()
    name = '::tetris4d::canvas_batch_v1'
    if not canvas.tk.call('info', 'commands', name):
        canvas.tk.eval('namespace eval ::tetris4d {}; '
                       'proc ::tetris4d::canvas_batch_v1 {canvas commands} {'
                       'set result {}; foreach command $commands {'
                       'lappend result [$canvas {*}$command]}; return $result}')
    return canvas.tk.splitlist(canvas.tk.call(name, canvas._w, tuple(commands)))


class WellView:
''')
method('WellView', '_faces', '''
def _faces(self, cubes, occupied):
    tx, ty, tz = self.toward
    visible = [(normal, vertices, steps) for normal, vertices, steps in FACE_RENDER_GEOMETRY
               if normal[0] * tx + normal[1] * ty + normal[2] * tz > 1e-9]
    faces = []
    for x, y, z, color, owner in cubes:
        for (nx, ny, nz), vertices, steps in visible:
            if (x + nx, y + ny, round(z + nz, 9)) in occupied:
                continue
            points = [(x + dx, y + dy, z + dz) for dx, dy, dz in vertices]
            depth = sum(px * tx + py * ty + pz * tz for px, py, pz in points) / 4
            shade = blend(color, '#FFFFFF', .18) if nz else blend(color, '#000000', .18 if nx else .35)
            borders = []
            for i, (sx, sy, sz) in enumerate(steps):
                beside = (x + sx, y + sy, round(z + sz, 9))
                beyond = (beside[0] + nx, beside[1] + ny, round(beside[2] + nz, 9))
                if not owner or occupied.get(beside) != owner or beyond in occupied:
                    borders.append((points[i], points[(i + 1) % 4]))
            faces.append((depth, points, shade, borders, owner))
    faces.sort(key=lambda face: face[0])
    return faces
''')
replace('    def _face(self, face, tag):\n', '''    def _static_mesh(self, faces):
        commands, starts = [], []
        project = self.project
        for i, (_, points, color, borders, _) in enumerate(faces):
            tag = f'static_face:{i}'
            starts.append(len(commands))
            commands.append(('create', 'polygon', *[v for point in points for v in project(*point)],
                             '-fill', color, '-outline', blend(color, '#162536', .3), '-width', .5,
                             '-tags', (tag, 'voxel_face', 'locked_piece', 'static_mesh')))
            for a, b in borders:
                commands.append(('create', 'line', *project(*a), *project(*b), '-fill', '#162536',
                                 '-width', 1.8, '-tags', (tag, 'static_mesh')))
        created = canvas_batch(self.canvas, commands)
        self.mesh_batches = getattr(self, 'mesh_batches', 0) + bool(commands)
        return [int(created[i]) for i in starts]

    def _face(self, face, tag):
''')
replace("            self._static_depths, self._static_items = [], []\n            for i, face in enumerate(static_faces):\n                self._static_depths.append(face[0])\n                self._static_items.append(self._face(face, f'static_face:{i}'))", "            self._static_depths = [face[0] for face in static_faces]\n            self._static_items = self._static_mesh(static_faces)")
replace('        self.section_revisions = [0] * FOURTH\n', '        self.section_revisions = [0] * FOURTH\n        self.ownership_appends = self.ownership_full_scans = 0\n')
replace("        if game.locked_pieces is not self.piece_cache:\n            self.piece_cache = game.locked_pieces\n            self.owners = {}\n            tokens = [[] for _ in range(FOURTH)]\n            for piece in game.locked_pieces:\n                for cell in piece.cells:\n                    identity = (piece.identifier, piece.piece)\n                    self.owners[cell] = identity\n                    tokens[cell[3]].append((cell, identity))\n            for w, cells in enumerate(tokens):\n                token = tuple(cells)\n                if token != self.section_tokens[w]:\n                    self.section_tokens[w] = token\n                    self.section_revisions[w] += 1", """        if game.locked_pieces is not self.piece_cache:
            previous, current = self.piece_cache, game.locked_pieces
            append_only = (previous is not None and len(current) > len(previous)
                           and all(current[i] is piece for i, piece in enumerate(previous)))
            self.piece_cache = current
            tokens = [[] for _ in range(FOURTH)]
            if append_only:
                added = current[len(previous):]
                self.ownership_appends += len(added)
            else:
                self.owners = {}
                added = current
                self.ownership_full_scans += 1
            for piece in added:
                identity = (piece.identifier, piece.piece)
                for cell in piece.cells:
                    self.owners[cell] = identity
                    tokens[cell[3]].append((cell, identity))
            for w, cells in enumerate(tokens):
                if append_only and not cells:
                    continue
                token = (self.section_tokens[w] or ()) + tuple(cells) if append_only else tuple(cells)
                if token != self.section_tokens[w]:
                    self.section_tokens[w] = token
                    self.section_revisions[w] += 1""")
method('SliceAtlas', '__init__', '''
def __init__(self, canvas, on_select) -> None:
    self.canvas, self.on_select = canvas, on_select
    self.boxes = []
    self.last_key = None
    self.layout_key = None
    self.board_cache = self.move_cache = None
    self.cells = [None] * AREA
    self.heights = [0] * AREA
    self.borders, self.labels = [], []
    self.selected = None
    self.scene_rebuilds = self.cell_updates = self.cells_created = 0
    self.colors = tuple(blend(PALETTE['panel'], '#A9B5F5', .2 + .8 * h / HEIGHT)
                        for h in range(HEIGHT + 1))
    canvas.bind('<Button-1>', self.click)
''')
method('SliceAtlas', 'draw', '''
def draw(self, game: Game, move: Move | None, selected: int) -> None:
    c = self.canvas
    width, height = max(140, c.winfo_width()), max(140, c.winfo_height())
    key = (id(game.board), id(move), selected, width, height)
    if key == self.last_key:
        return
    self.last_key = key
    resized = self.layout_key != (width, height)
    if resized:
        self.layout_key = (width, height)
        self.scene_rebuilds += 1
        c.delete('all')
        self.boxes.clear()
        self.borders.clear()
        self.labels.clear()
        self.cells = [None] * AREA
        self.heights = [0] * AREA
        self.board_cache = None
        self.selected = None
        c.create_text(10, 14, anchor='w', text='ALL 16 W SECTIONS', fill=PALETTE['text'], font=('TkDefaultFont', 9, 'bold'))
        c.create_text(10, 33, anchor='w', text='Top views · click a section', fill=PALETTE['muted'], font=('TkDefaultFont', 8))
        size = max(8, min((width - 32) / 4, (height - 125) / 4))
        gap_x = (width - size * 4) / 5
        for w in range(FOURTH):
            x0 = gap_x + (w % 4) * (size + gap_x)
            y0 = 57 + (w // 4) * (size + 21)
            self.boxes.append((w, x0, y0, size))
            self.borders.append(c.create_rectangle(x0 - 2, y0 - 2, x0 + size + 2, y0 + size + 2,
                                fill=PALETTE['board'], outline=PALETTE['grid'], width=1))
            self.labels.append(c.create_text(x0, y0 - 10, anchor='w', text=f'W {w:02}',
                               fill=PALETTE['muted'], font=('TkFixedFont', 8)))
    commands, new_items = [], []
    if self.board_cache is not game.board:
        old_board = self.board_cache
        self.board_cache = game.board
        for i, mask in enumerate(game.board):
            if old_board is not None and mask == old_board[i]:
                continue
            value = mask.bit_length()
            if value == self.heights[i]:
                continue
            self.heights[i] = value
            item = self.cells[i]
            self.cell_updates += 1
            if item is None:
                if not value:
                    continue
                x, y, w = column_coordinates(i)
                _, x0, y0, size = self.boxes[w]
                cell = size / 16
                new_items.append((i, len(commands)))
                commands.append(('create', 'rectangle', x0 + x * cell, y0 + (15 - y) * cell,
                                 x0 + (x + 1) * cell, y0 + (16 - y) * cell,
                                 '-fill', self.colors[value], '-outline', '', '-tags', ('atlas_height',)))
            else:
                commands.append(('itemconfigure', item, '-fill', self.colors[value],
                                 '-state', 'normal' if value else 'hidden'))
    result = canvas_batch(c, commands)
    for i, position in new_items:
        self.cells[i] = int(result[position])
    self.cells_created += len(new_items)
    if self.selected != selected:
        for w in (self.selected, selected):
            if w is not None:
                active = w == selected
                c.itemconfigure(self.borders[w], outline=PALETTE['accent'] if active else PALETTE['grid'], width=2 if active else 1)
                c.itemconfigure(self.labels[w], fill=PALETTE['accent'] if active else PALETTE['muted'])
        self.selected = selected
    if resized or move is not self.move_cache:
        self.move_cache = move
        c.delete('atlas_active')
        if move is not None:
            for x, y, z, w in placed_cells(move):
                _, x0, y0, size = self.boxes[w]
                cell = size / 16
                c.create_rectangle(x0 + x * cell, y0 + (15 - y) * cell, x0 + (x + 1) * cell, y0 + (16 - y) * cell,
                                   fill=PIECE_COLORS[move.piece], outline='#FFFFFF', width=.4, tags=('atlas_active',))
    c.tag_raise('atlas_active')
''')
replace('        return result\n\n    def _draw_now(self) -> None:', '''        result.update(atlas_scene_rebuilds=self.atlas.scene_rebuilds,
                      atlas_cell_updates=self.atlas.cell_updates,
                      atlas_cells_created=self.atlas.cells_created,
                      ownership_appends=self.well.ownership_appends,
                      ownership_full_scans=self.well.ownership_full_scans,
                      static_mesh_batches=sum(getattr(view, 'mesh_batches', 0) for view in self.well.views))
        return result

    def _draw_now(self) -> None:''')
compile(source, str(path), 'exec')
path.write_text(source, encoding='utf-8')
print('Patched renderer SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
