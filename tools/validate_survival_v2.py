from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train_survival_v2 as v2

base, neural, game = v2.base, v2.neural, v2.game


def validate(args):
    torch.set_num_threads(1)
    game.configure_backend(args.backend)
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    source = ROOT / 'models/neural-rl.pt'
    original = base.file_hash(source)
    model, checkpoint = neural.load_checkpoint(source, torch.device('cpu'))
    model.config = replace(model.config, policy_mode='teacher-residual', residual_scale=1., recover_target=False)
    model.rank_out.weight.data.zero_()
    model.rank_out.bias.data.zero_()
    model.eval()
    state, target, events = game.Game(7), 1, []
    digest = hashlib.sha256()
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(args.pieces):
            prepared = neural.prepare(state, target, model.config)
            if prepared is None:
                raise AssertionError(f'Unexpected blocked state at piece {step}')
            batch = neural.collate([prepared.arrays], torch.device('cpu'))
            logits = model(batch)['logits']
            selected = int(logits.argmax().item())
            if selected != prepared.teacher:
                raise AssertionError('Zero residual did not select the exact teacher index')
            candidate = prepared.options[selected]
            teacher = game.choose_sustained_move(state.board, state.current, target, model.config.candidates, state.pieces)
            if teacher is None or replace(candidate, score=teacher.move.candidate.score) != teacher.move.candidate:
                raise AssertionError(f'Teacher placement changed at piece {step}')
            move = game.materialize_move(state.board, candidate)
            if move.board != teacher.move.board:
                raise AssertionError('Resulting teacher board differs')
            if step % 128 == 0:
                traced = model(batch, capture=True)
                if not torch.equal(logits, traced['logits']):
                    raise AssertionError('Capturing activations changed the policy logits')
            state.commit(move)
            digest.update(f'{candidate.piece}:{candidate.rotation}:{candidate.x}:{candidate.y}:{candidate.z}:{candidate.w};'.encode())
            if move.layers:
                events.append(state.pieces)
                target = game.next_hyperlayer_target(state.board, target)
    if args.pieces >= 2600 and (events[:2] != [1251, 2466] or state.layers < 2):
        raise AssertionError(f'Expected repeated sustained clears, got {events}')
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / 'run'
        settings = v2.arguments(['--checkpoint', str(source), '--run-dir', str(run), '--device', 'cpu',
                                '--envs', '2', '--rollout', '2', '--batch', '2', '--optimizer-batch', '2',
                                '--max-updates', '1', '--selection-pieces', '4', '--target-kl', '.9',
                                '--calibration-games', '0', '--allow-uncalibrated', '--record-games'])
        with contextlib.redirect_stdout(io.StringIO()), base.run_lock(run):
            trainer = v2.SurvivalTrainer(settings, base.Budget(1))
            report = trainer.run()
        if report['placements'] != 4 or report['optimizer_steps'] != 6:
            raise AssertionError('Full-size optimizer-minibatch smoke failed')
        saved = torch.load(trainer.last, map_location='cpu', weights_only=True)
        if not all(torch.isfinite(t).all() for t in saved['model'].values()):
            raise AssertionError('Non-finite full-model parameters')
        for path in (run / 'games').rglob('*'):
            if path.is_file():
                base.verify_game(path)
        optimizer_steps = report['optimizer_steps']
    if base.file_hash(source) != original:
        raise AssertionError('Original trained checkpoint was modified')
    result = {'passed': True, 'backend': game.BACKEND, 'device': 'cpu', 'torch': str(torch.__version__),
              'parameters': sum(p.numel() for p in model.parameters()), 'initial_checkpoint_sha256': original,
              'source_checkpoint_unchanged': True, 'identical_teacher_choices': state.pieces,
              'clear_events': events, 'final_summary': state.summary(), 'action_digest': digest.hexdigest(),
              'full_model_optimizer_steps': optimizer_steps, 'elapsed_validation_seconds': time.perf_counter() - start,
              'note': 'Correctness and short optimizer smoke only. Zero residual reproduces teacher behavior; no learned playing-strength gain or CUDA speedup is claimed.'}
    if args.output:
        base.durable_json(args.output, result)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pieces', type=int, default=2600)
    parser.add_argument('--backend', choices=('auto', 'native', 'numpy', 'python'), default='auto')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.pieces < 1:
        parser.error('pieces must be positive')
    validate(args)


if __name__ == '__main__':
    main()
