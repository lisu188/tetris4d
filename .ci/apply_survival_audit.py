from pathlib import Path

root = Path('.')
p = root / 'tetris_neural.py'
s = p.read_text(encoding='utf-8')

def change(old, new):
    global s
    assert s.count(old) == 1, (old[:100], s.count(old))
    s = s.replace(old, new)

change('    protect_target: bool = True\n', "    protect_target: bool = True\n    recover_target: bool = False\n    policy_mode: str = 'neural'\n    residual_scale: float = 1.0\n    detach_critic: bool = False\n    centered_critic: bool = False\n")
change("        if type(self.protect_target) is not bool:\n            raise ValueError('protect_target must be boolean')", "        if any(type(getattr(self, n)) is not bool for n in ('protect_target', 'recover_target', 'detach_critic', 'centered_critic')):\n            raise ValueError('Target and critic settings must be boolean')\n        if self.policy_mode not in ('neural', 'teacher-residual'):\n            raise ValueError('Unsupported policy mode')\n        if not math.isfinite(self.residual_scale) or not 0 <= self.residual_scale <= 4:\n            raise ValueError('residual_scale must be between 0 and 4')")
change('        return spatial, pooled, self.value(joined).squeeze(-1), self.completion(joined), activation', "        value_input = joined.detach() if self.config.detach_critic else joined\n        return spatial, pooled, self.critic(value_input), self.completion(joined), activation\n\n    def critic(self, joined):\n        if self.config.centered_critic:\n            hidden = torch.sigmoid(self.value[0](joined)) - .5\n            return self.value[2](hidden).squeeze(-1)\n        return self.value(joined).squeeze(-1)")
change("        logits = torch.cat(pieces, dim=1).masked_fill(~batch['valid'].bool(), -torch.inf)", "        logits = torch.cat(pieces, dim=1)\n        if self.config.policy_mode == 'teacher-residual':\n            if 'teacher_prior' not in batch:\n                raise ValueError('Residual policy requires exact teacher ranking features')\n            logits = batch['teacher_prior'].float() + self.config.residual_scale * (2 * torch.sigmoid(logits.float()) - 1)\n        logits = logits.masked_fill(~batch['valid'].bool(), -torch.inf)")
change("        result = {'logits': logits, 'value': value, 'completion': completion}", "        result = {'logits': logits, 'value': value, 'completion': completion,\n                  'critic_input': torch.cat((pooled, batch['context']), dim=1).detach()}")
change('def prepare_steps(state, target: int, config: Config, candidates: int | None = None):', 'def _prepare_steps(state, target: int, config: Config, candidates: int | None = None):')
change('    options = tuple(options[i] for i in keep)\n    teacher = keep.index(best)', '    kept_keys = [keys[i] for i in keep]\n    options = tuple(options[i] for i in keep)\n    teacher = keep.index(best)')
change("              'column_mask': mask, 'valid': np.ones(len(options), dtype=np.bool_)}\n    return Prepared", "              'column_mask': mask, 'valid': np.ones(len(options), dtype=np.bool_)}\n    order = sorted(range(len(options)), key=lambda i: kept_keys[i], reverse=True)\n    rank = np.empty(len(options), dtype=np.float32)\n    rank[order] = np.arange(len(options), dtype=np.float32)\n    deficits = np.asarray([kept_keys[teacher][0] - k[0] for k in kept_keys], dtype=np.float32)\n    arrays['teacher_prior'] = -32 * deficits - rank / 8\n    return Prepared")
change('\n\ndef prepare(*args, **kwargs):', '''\n\ndef prepare_steps(state, target: int, config: Config, candidates: int | None = None):
    prepared = yield from _prepare_steps(state, target, config, candidates)
    if prepared is not None or not config.recover_target or not config.protect_target:
        return prepared
    if state.game_over or not game.has_legal_move(state.board, state.current):
        return None
    for alternative in sorted(range(game.HEIGHT), key=lambda z: (abs(z - target), z)):
        if alternative == target:
            continue
        status = game.target_layer_status(state.board, alternative)
        if status.buried or status.filled == game.AREA:
            continue
        prepared = yield from _prepare_steps(state, alternative, config, candidates)
        if prepared is not None:
            return prepared
    return None


def prepare(*args, **kwargs):''')
change("if name in ('features', 'indices', 'column_mask', 'valid'):", "if name in ('features', 'indices', 'column_mask', 'valid', 'teacher_prior'):")
change("    def __init__(self, path, device='auto', engine=None, allow_untrained=False):", "    def __init__(self, path, device='auto', engine=None, allow_untrained=False, capture_traces=True):")
change('        self.last_trace = None\n        self.last_metrics = {}', "        self.last_trace = None\n        self.pending_trace = None\n        self.capture_traces = capture_traces\n        self.last_metrics = {}")
change("        batch = collate([prepared.arrays], self.device)\n        with torch.inference_mode():\n            prediction = self.model(batch, capture=True)", "        requested_target = target\n        target = int(prepared.arrays['targets'])\n        batch = collate([prepared.arrays], self.device)\n        with torch.inference_mode():\n            prediction = self.model(batch, capture=self.capture_traces is True)")
a = s.index("            self.last_trace = {'activation':", s.index('class NeuralAgent:'))
b = s.index('        inferred = time.perf_counter()', a)
s = s[:a] + '''            trace_info = {'logit': score, 'piece': state.current, 'piece_number': state.pieces + 1,
                          'shortlist': len(prepared.options), 'selected': index}
            if self.capture_traces is True:
                self.last_trace = self._trace(prediction, trace_info)
            elif self.capture_traces == 'lazy':
                self.pending_trace = (prepared.arrays, trace_info)
''' + s[b:]
change("                             'teacher_agreement': index == prepared.teacher, **self.metadata}", "                             'teacher_agreement': index == prepared.teacher,\n                             'requested_target': requested_target, 'active_target': target,\n                             'target_recovered': target != requested_target,\n                             'trace_captured': self.capture_traces is True, **self.metadata}")
change("    def create_view(self, parent, legacy_class, policy):\n        return NeuralView(parent, self, legacy_class, policy)", '''    def _trace(self, prediction, info):
        return {**info, 'activation': prediction['activation'][0].float().cpu().numpy(),
                'hidden1': prediction['hidden1'][0].float().cpu().numpy(),
                'hidden2': prediction['hidden2'][0].float().cpu().numpy(),
                'completion': torch.sigmoid(prediction['completion'][0]).float().cpu().numpy(),
                'value': float(prediction['value'][0].item())}

    def get_trace(self):
        if self.pending_trace is not None:
            arrays, info = self.pending_trace
            with torch.inference_mode():
                prediction = self.model(collate([arrays], self.device), capture=True)
            self.last_trace = self._trace(prediction, info)
            self.pending_trace = None
        return self.last_trace

    def create_view(self, parent, legacy_class, policy):
        self.capture_traces = 'lazy'
        return NeuralView(parent, self, legacy_class, policy)''')
change('        trace = self.agent.last_trace', '        trace = self.agent.get_trace()')
change('    agent = NeuralAgent(args.checkpoint, args.device)', '    agent = NeuralAgent(args.checkpoint, args.device, capture_traces=False)')
change('                state.commit(decision.move)\n                decisions += 1', '                target = decision.target if decision.target is not None else target\n                state.commit(decision.move)\n                decisions += 1')
compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')

p = root / 'sparse_tetris_4d_optimized.py'
s = p.read_text(encoding='utf-8')
change('                self.planned = self.decision.move if self.decision else None\n', "                self.planned = self.decision.move if self.decision else None\n                if self.decision is not None and self.decision.policy == 'neural' and self.decision.target is not None:\n                    self.target.set(str(self.decision.target))\n")
compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')

p = root / 'tetris_benchmark_logging.py'
s = p.read_text(encoding='utf-8')
change('    def _search(self, name):\n        original = getattr(self.game, name)', '    def _search(self, name, owner=None):\n        owner = self.game if owner is None else owner\n        original = getattr(owner, name)')
change("piece=args[1] if len(args) > 1 else kwargs.get('piece'),", "piece=getattr(args[0], 'current', None) if owner is not self.game and args else (args[1] if len(args) > 1 else kwargs.get('piece')),")
change('        self._patch(self.game, name, steps)', '        self._patch(owner, name, steps)')
change("policy=decision.policy if decision is not None else ('sustained' if name.startswith('sustained') else 'hyperlayer' if name.startswith('hyperlayer') else 'network'),", "policy=decision.policy if decision is not None else ('neural' if owner is not self.game else 'sustained' if name.startswith('sustained') else 'hyperlayer' if name.startswith('hyperlayer') else 'network'),")
change('            self._search(name)\n        original_commit', "            self._search(name)\n        agent = getattr(self.game, 'NEURAL_AGENT', None)\n        if agent is not None:\n            self._search('steps', agent)\n        original_commit")
compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')

p = root / 'train_overnight.py'
s = p.read_text(encoding='utf-8')
change('import hashlib\n', 'import hashlib\nimport gzip\n')
change("        return {'number': self.number, 'seed': self.state.seed, 'piece_set': self.state.piece_set,\n                'target': self.target, 'moves': self.moves, 'phase': self.phase,\n                'board_sha256': board_hash(self.state.board), 'summary': self.state.summary()}", "        result = {'number': self.number, 'seed': self.state.seed, 'piece_set': self.state.piece_set,\n                  'target': self.target, 'moves': self.moves, 'phase': self.phase,\n                  'board_sha256': board_hash(self.state.board), 'summary': self.state.summary()}\n        pending = getattr(self, 'pending_retarget', None)\n        if pending is not None:\n            result['pending_retarget'] = pending\n        return result")
change('    episode.moves.append(row)\n    return move', "    pending = getattr(episode, 'pending_retarget', None)\n    if pending is not None:\n        row['retarget_from'] = pending\n        episode.pending_retarget = None\n    episode.moves.append(row)\n    return move")
change("        if row['target_before'] != episode.target:\n            raise ValueError('Replay target history does not match.')", "        if 'retarget_from' in row:\n            if row['retarget_from'] != episode.target:\n                raise ValueError('Replay recovery source does not match.')\n            status = game.target_layer_status(state.board, row['target_before'])\n            if status.buried or status.filled == game.AREA:\n                raise ValueError('Replay recovery target is inaccessible.')\n            episode.pending_retarget, episode.target = episode.target, row['target_before']\n        if row['target_before'] != episode.target:\n            raise ValueError('Replay target history does not match.')")
change("    if episode.target != snapshot['target'] or board_hash(state.board) != snapshot['board_sha256']:", "    if snapshot.get('pending_retarget') is not None:\n        if snapshot['pending_retarget'] != episode.target:\n            raise ValueError('Pending target recovery source does not match.')\n        status = game.target_layer_status(state.board, snapshot['target'])\n        if status.buried or status.filled == game.AREA:\n            raise ValueError('Pending recovery target is inaccessible.')\n        episode.pending_retarget, episode.target = episode.target, snapshot['target']\n    if episode.target != snapshot['target'] or board_hash(state.board) != snapshot['board_sha256']:")
change("    data = json.loads(Path(path).read_text(encoding='utf-8'))", "    path = Path(path)\n    text = gzip.decompress(path.read_bytes()).decode('utf-8') if path.suffix == '.gz' else path.read_text(encoding='utf-8')\n    data = json.loads(text)")
change('    def checkpoint_metadata(self):\n        return {}\n', '    def checkpoint_metadata(self):\n        return {}\n\n    def configure_model(self, resuming):\n        pass\n\n    def create_optimizer(self):\n        return torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)\n\n    def restore_extra(self, saved):\n        pass\n')
change("        self.signature['model_config'] = asdict(self.model.config)\n        self.recorder.context", "        self.configure_model(saved is not None)\n        self.signature['model_config'] = asdict(self.model.config)\n        self.recorder.context")
change('        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)', '        self.optimizer = self.create_optimizer()')
change("            self.scaler.load_state_dict(saved['extra']['scaler'])\n        if args.max_updates:", "            self.scaler.load_state_dict(saved['extra']['scaler'])\n            self.restore_extra(saved)\n        if args.max_updates:")
compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
