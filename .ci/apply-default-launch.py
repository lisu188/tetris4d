from pathlib import Path


def replace(path: str, before: str, after: str) -> None:
    file = Path(path)
    text = file.read_text(encoding='utf-8')
    if text.count(before) != 1:
        raise ValueError(f'Expected exactly one matching section in {path}: {before!r}')
    file.write_text(text.replace(before, after), encoding='utf-8')


source = 'sparse_tetris_4d_optimized.py'
replace(source, "                 policy: str = 'network', target: int = 1) -> None:",
        "                 policy: str = 'hyperlayer', target: int = 1) -> None:")
replace(source, 'def main() -> int:', 'def main(argv: Sequence[str] | None = None) -> int:')
replace(source,
        "    parser.add_argument('--weights', type=Path, help='Load a validated JSON checkpoint before evaluation or play.')\n    parser.add_argument('--policy', choices=('network', 'hyperlayer'), default='network',\n                        help='hyperlayer protects access and prioritizes target coverage; uses depth 1 and pauses after a clear.')",
        "    weights = parser.add_mutually_exclusive_group()\n    weights.add_argument('--weights', type=Path,\n                         default=Path(__file__).resolve().parent / 'models' / 'trained-sigmoid.json',\n                         help='Load a validated JSON checkpoint; defaults to models/trained-sigmoid.json beside this script.')\n    weights.add_argument('--baseline', action='store_true', help='Use the original untrained weights instead of a checkpoint.')\n    parser.add_argument('--policy', choices=('network', 'hyperlayer'), default='hyperlayer',\n                        help='Default: hyperlayer, protecting target access with depth 1 and pausing after a clear; network restores neural-only search.')")
replace(source, '    args = parser.parse_args()', '    args = parser.parse_args(argv)')
replace(source, '        if args.weights:\n            load_checkpoint(args.weights)',
        "        if args.baseline:\n            configure_network(BASELINE_PARAMETERS, {'method': 'none', 'description': 'Untrained inherited heuristic coefficients.'})\n        else:\n            load_checkpoint(args.weights)")
replace('tests/test_hyperlayer_gui.py', "        app = game.TetrisApp(root, policy='hyperlayer', target=1)",
        '        app = game.TetrisApp(root)')
replace('README.md', '## Run\n\n',
        '## Run\n\nNo arguments are needed: the default is the hyperlayer planner, target Z=1, and the bundled trained sigmoid checkpoint. The checkpoint is resolved relative to the script, so launching from another working directory works. Use Start/Space to run; the planner pauses after its first clear.\n\n')
replace('README.md', 'Default search uses depth 2, beam width 4, and 2,048 deterministically sampled candidate placements per board.',
        'Default hyperlayer search uses depth 1 and 2,048 sampled candidates plus targeted vertical-I finishing placements. Neural-only search (`--policy network`) uses depth 2, beam width 4, and 2,048 deterministically sampled candidate placements per board.')
replace('README.md', 'The baseline remains the default; load the checkpoint explicitly:',
        'The trained checkpoint is loaded automatically; this explicit form remains equivalent:')
replace('README.md', "Enable the target planner in the GUI's **Planner** selector, or launch it directly:",
        "The target planner and trained checkpoint are selected automatically. A plain launch is sufficient:")
replace('README.md', 'python sparse_tetris_4d_optimized.py --policy hyperlayer --target 1 --weights models/trained-sigmoid.json',
        'python sparse_tetris_4d_optimized.py')
replace('README.md', 'The original `--policy network` remains the default.',
        'Use `--policy network` for neural-only search, or `--policy network --baseline` to restore the original untrained controller.')
replace('README.md', 'The normal game still loads weights only when `--weights` is supplied.',
        'The normal game also loads this checkpoint automatically; `--weights PATH` overrides it and `--baseline` disables checkpoint loading.')
replace('docs/first-hyperlayer.md', 'The target policy is opt-in.',
        'The target policy is the default when launching the application.')
replace('docs/first-hyperlayer.md',
        'The helper loads the existing trained checkpoint by default. The main application retains its prior baseline-weight default; pass `--weights` to reproduce the published trajectories. `--policy network` preserves the original controller.',
        'Both the helper and main application load the shipped trained checkpoint by default. Running `python sparse_tetris_4d_optimized.py` selects hyperlayer mode and Z=1 without arguments. The main application resolves its default checkpoint relative to the script, not the working directory. `--policy network` selects neural-only search with trained weights; `--policy network --baseline` restores the original untrained controller. Explicit `--weights PATH` overrides the bundled checkpoint and cannot be combined with `--baseline`. A missing or invalid checkpoint is reported as an error rather than silently changing the model.')
replace('docs/training.md', 'Without `--weights`, the original baseline remains available unchanged.',
        'The main application now loads the bundled trained checkpoint by default and selects the hyperlayer planner at Z=1. Use `--baseline` for the original untrained weights, or `--policy network --baseline` for the original neural-only controller. Explicit `--weights PATH` overrides the bundled checkpoint.')
