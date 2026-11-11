from pathlib import Path
import ast
import hashlib

p = Path('sparse_tetris_4d_optimized.py')
raw = p.read_bytes()
assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == '4cd46774e87815096e9faa87145b2b25ba776898'
s = raw.decode('utf-8')
original = ast.parse(s)

def change(old, new, count=None):
    global s
    actual = s.count(old)
    assert actual and (count is None or actual == count), (old, actual)
    s = s.replace(old, new)

change('self.speed = tk.DoubleVar(value=3.0)', 'self.speed = tk.DoubleVar(value=0.0)', 1)
change("self.speed_choice = tk.StringVar(value='3/s')", "self.speed_choice = tk.StringVar(value='Unlimited')", 1)
change("policy: str = 'hyperlayer', target: int = 1", "policy: str = 'sustained', target: int = 1", 1)
change("if policy not in ('network', 'hyperlayer'):", "if policy not in ('network', 'hyperlayer', 'sustained'):", 1)
change("if policy not in ('network', 'random', 'hyperlayer'):", "if policy not in ('network', 'random', 'hyperlayer', 'sustained'):", 1)
change("if policy in ('network', 'hyperlayer'):", "if policy in ('network', 'hyperlayer', 'sustained'):", 1)
change("self.policy.get() == 'hyperlayer'", "self.policy.get() in TARGET_POLICIES")
change("self.decision.policy == 'hyperlayer'", "self.decision.policy in TARGET_POLICIES")
change("policy == 'hyperlayer'", "policy in TARGET_POLICIES")
change("choices=('network', 'hyperlayer'), default='hyperlayer'", "choices=('network', 'hyperlayer', 'sustained'), default='sustained'", 1)
change("values=('network', 'hyperlayer')", "values=('network', 'hyperlayer', 'sustained')", 1)
change('self.search = hyperlayer_decision_steps(', "self.search = (sustained_decision_steps if self.policy.get() == 'sustained' else hyperlayer_decision_steps)(", 1)
change('decision = choose_hyperlayer_move(', "decision = (choose_sustained_move if policy == 'sustained' else choose_hyperlayer_move)(", 1)
change("Default: continuous hyperlayer search with depth 1 and automatic retargeting after clears; network uses neural-only search.", "Default: sustained clearing with future-hole penalty; hyperlayer preserves the previous controller; network uses neural-only search.", 1)
change("hyperlayer planner, depth 1, protected target + sigmoid tie-break", "target planner, depth 1, protected target + sigmoid tie-break", 1)
block = Path('.ci/sustained-source.py').read_text(encoding='utf-8')
change('\n\nclass PieceStream:', "\n\nTARGET_POLICIES = ('hyperlayer', 'sustained')\n\n" + block + '\n\nclass PieceStream:', 1)
compile(s, str(p), 'exec')
critical = ('Game', 'Shape', 'PieceStream', 'score_candidate', 'materialize_move', 'clear_layers', 'merge_rigid_piece', 'configure_network', 'hyperlayer_decision_steps', 'choose_hyperlayer_move', 'native_scores')
before = {n.name: ast.dump(n) for n in original.body if hasattr(n, 'name')}
after = {n.name: ast.dump(n) for n in ast.parse(s).body if hasattr(n, 'name')}
for name in critical:
    if name in before:
        assert before[name] == after[name], name
p.write_text(s, encoding='utf-8')

p = Path('tetris_benchmark_logging.py')
s = p.read_text(encoding='utf-8')
old = "('decision_steps', 'hyperlayer_decision_steps')"
assert s.count(old) == 1
s = s.replace(old, "('decision_steps', 'hyperlayer_decision_steps', 'sustained_decision_steps')")
old = "policy='hyperlayer' if name.startswith('hyperlayer') else 'network',"
assert s.count(old) == 1
s = s.replace(old, "policy=decision.policy if decision is not None else ('sustained' if name.startswith('sustained') else 'hyperlayer' if name.startswith('hyperlayer') else 'network'),")
p.write_text(s, encoding='utf-8')

for name in ('tests/test_launch_defaults.py', 'tests/test_native.py', 'tests/test_frame_pacing.py'):
    p = Path(name)
    s = p.read_text(encoding='utf-8')
    s = s.replace("self.assertEqual(args.policy, 'hyperlayer')", "self.assertEqual(args.policy, 'sustained')")
    s = s.replace("self.assertEqual(run.call_args.args[5], 'hyperlayer')", "self.assertEqual(run.call_args.args[5], 'sustained')")
    s = s.replace("('hyperlayer', 1, False)", "('sustained', 1, False)")
    p.write_text(s, encoding='utf-8')
p = Path('tests/test_unlimited_speed_gui.py')
s = p.read_text(encoding='utf-8')
s = s.replace('game.choose_hyperlayer_move(', 'game.choose_sustained_move(')
old = "check(app.speed.get() == 3, 'Animated default changed unexpectedly')"
assert s.count(old) == 1
s = s.replace(old, "check(app.speed.get() == 0 and app.speed_choice.get() == 'Unlimited' and app.policy.get() == 'sustained', 'Unlimited sustained defaults missing')")
p.write_text(s, encoding='utf-8')
p = Path('tests/test_renderer_optimization.py')
s = p.read_text(encoding='utf-8')
s = s.replace("('WellView', 'AllWellView', 'SliceAtlas')", "('WellView', 'AllWellView', 'SliceAtlas', 'main', 'run_game')")
s = s.replace("if method != 'render_metrics':", "if method not in ('render_metrics', '__init__', 'refresh_plan', 'place', '_draw_now'):")
p.write_text(s, encoding='utf-8')
for name in ('README.md', 'docs/playback-speed.md'):
    p = Path(name)
    s = p.read_text(encoding='utf-8')
    s = s.replace('default is the hyperlayer planner', 'default is the sustained-clearing planner')
    s = s.replace('The default remains the previous 3/s animation.', 'The default is Unlimited, with playback initially paused.')
    s = s.replace('The previous 3/s animated default is retained.', 'Unlimited is selected by default; Start/Space begins autoplay.')
    s = s.replace('the default 3/s animation', 'the optional 3/s animation')
    s += '\n\n## Sustained clearing and Unlimited defaults\n\nNormal argument-free launch now selects Unlimited and the sustained target planner. Start/Space begins playback. The planner retains target access but subtracts four times the newly created holes above the target from target-coverage gain, then uses the unchanged sigmoid score to break ties. It is a policy improvement, not newly trained neural weights. `--policy hyperlayer` reproduces the previous controller. All 16 views, frame skipping and logging remain enabled. Native source is unchanged. See `docs/sustained-policy.md` and `docs/gpu-neural-design.md`.\n'
    p.write_text(s, encoding='utf-8')
p = Path('tests/test_hyperlayer_gui.py')
s = p.read_text(encoding='utf-8').replace('app = game.TetrisApp(root)', "app = game.TetrisApp(root, policy='hyperlayer')")
p.write_text(s, encoding='utf-8')
