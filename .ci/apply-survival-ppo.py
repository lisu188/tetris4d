from pathlib import Path
import hashlib


def load(name, digest):
    path = Path(name)
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, name
    return path, raw.decode('utf-8')


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


path, text = load('train_overnight.py', 'f78224785996879bcef8081eb21d922f2506252468483930c3b994860212a53f')
text = replace(text, 'class Trainer:\n', 'class Trainer:\n    def training_signature(self):\n        return {}\n\n    def checkpoint_metadata(self):\n        return {}\n\n')
text = replace(text, '        self.baseline = None\n', '        self.signature.update(self.training_signature())\n        self.baseline = None\n')
text = replace(text, "        temporary = self.root / '.next-checkpoint.pt'", "        extra.update(self.checkpoint_metadata())\n        temporary = self.root / '.next-checkpoint.pt'")
path.write_text(text, encoding='utf-8')

path, text = load('tetris_neural.py', '425a9cb52970f9a54481ad2f442a3863b48e157e2f93f7406eec5387809ed908')
old = "                         'updates': self.checkpoint['progress'].get('updates', 0), 'status': self.checkpoint['status']}"
new = old + "\n        extra = self.checkpoint.get('extra', {})\n        survival = extra.get('kind') == 'tetris4d-overnight-survival-v1'\n        self.metadata['value_semantics'] = extra.get('value_semantics', 'expected_remaining_placements_times_0.001' if survival else 'discounted_hyperlayers')\n        self.metadata['completion_semantics'] = extra.get('completion_semantics', 'imitation_auxiliary_not_recalibrated' if survival or self.checkpoint['progress'].get('mode') == 'a2c' else 'imitation_auxiliary')"
text = replace(text, old, new)
old = '        self.detail.set(f"{label}\\nLogit {trace[\'logit\']:.5f} · value {trace[\'value\']:.4f} · {trace[\'shortlist\']} legal candidates\\nEncoder input is the board BEFORE that decision. Hover for values.")'
new = "        value_units = 'remaining placements / 1000' if self.agent.metadata['value_semantics'] == 'expected_remaining_placements_times_0.001' else 'discounted clears'\n        auxiliary = 'Completion: imitation auxiliary; not recalibrated.' if 'not_recalibrated' in self.agent.metadata['completion_semantics'] else 'Completion: imitation auxiliary.'\n        self.detail.set(f\"{label}\\nLogit {trace['logit']:.5f} · value {trace['value']:.4f} ({value_units}) · {trace['shortlist']} legal candidates\\n{auxiliary}\\nEncoder input is the board BEFORE that decision. Hover for values.\")"
text = replace(text, old, new)
compile(text, str(path), 'exec')
path.write_text(text, encoding='utf-8')

path = Path('README.md')
text = path.read_text(encoding='utf-8')
text += '\n\n## Survival-first PPO training\n\nThe weight-audit follow-up is available as `train_survival.py`. It prefers the committed RL weights for initialization, resets the old clear-value output, uses clipped PPO with an exact categorical-KL early stop, and treats ties in the teacher primary objective as acceptable alternatives. It does not change the sigmoid architecture, native physics, stored models or no-argument GUI default.\n\n```bash\n.venv/bin/python -u train_survival.py --hours 8 --device cuda --record-games\n```\n\nRerunning resumes `runs/survival-ppo/last.pt`. Old A2C run directories remain separate. Checkpoints, replayable JSON games, evaluation and rank diagnostics are saved locally. A better training algorithm is not evidence of better play: only completed survival evaluations select `best.pt`, and there is no automatic promotion over sustained. See [survival PPO documentation](docs/survival-ppo.md).\n'
path.write_text(text, encoding='utf-8')
