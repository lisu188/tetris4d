import base64
import json
import zlib
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import calibrate_cross_critics as calibration
from tools import verify_cross_calibration as verify


class GateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        self.runtime = self.out / 'runtime'
        for name in (*verify.RUNTIME_FILES, 'models/trained-sigmoid.json'):
            file = self.runtime / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(name)
        self.driver, self.inc, self.donor = (self.out / name for name in ('driver.py', 'inc.pt', 'donor.pt'))
        for path in (self.driver, self.inc, self.donor):
            path.write_text(path.name)
        hashes = dict(INCUMBENT=verify.file_hash(self.inc), DONOR=verify.file_hash(self.donor),
                      TEACHER=verify.file_hash(self.runtime / 'models/trained-sigmoid.json'))
        replace = patch.multiple(verify, **hashes)
        replace.start()
        self.addCleanup(replace.stop)
        self.identity = dict(incumbent=verify.INCUMBENT, donor=verify.DONOR,
                             runtime=verify.runtime_identity(self.runtime, 'cpu'), seeds=list(verify.SEEDS),
                             script=verify.file_hash(self.driver), stride=128, actor_rng_offset=2701,
                             teacher=verify.TEACHER)

    def check(self):
        return verify.verify_gate(self.out, self.runtime, self.driver, 'cpu', incumbent_path=self.inc,
                                  donor_path=self.donor)

    def row(self, seed):
        n = 513
        samples = [[k, 0., float(n-k)] for k in range(0, n, 128)]
        return dict(seed=seed, placements=n, clears=1, terminal_piece=verify.terminal_piece(seed, n), reason='top_out',
                    terminal_board=base64.b64encode(zlib.compress(bytes((255,)) * 8192)).decode(),
                    samples=samples, mae=calibration.errors(samples, n))

    def save(self, base_identity, seed, row):
        path = self.out / 'critic-cache' / verify.digest(base_identity) / f'{seed}.json'
        calibration.ev.write_cache(path, dict(base_identity, seed=seed), row)

    def make(self, complete):
        reports = {}
        for t in verify.TEMPERATURES:
            rows = [self.row(s) for s in (verify.SEEDS if complete else verify.SEEDS[:1])]
            reports[str(t)] = rows
            for row in rows:
                self.save(dict(self.identity, temperature=t), row['seed'], row)
        gate = dict(complete=complete, identity=self.identity, completed={t:len(r) for t,r in reports.items()})
        if complete:
            gate.update(calibration.select(reports))
        else:
            gate.update(selected='incumbent', reason='time_budget')
            self.write('gpu-time.json', [dict(phase='calibration', label='critic', status='closed',
                                             seconds=30., allowance_seconds=120.)])
        self.write('critic-gate.json', gate)
        return gate

    def write(self, name, data):
        (self.out / name).write_text(json.dumps(data))

    def test_actual_hashes_and_independent_mae_selection(self):
        gate = self.make(True)
        self.assertEqual(self.check(), gate)
        self.write('critic-gate.json', dict(gate, selected='incumbent'))
        with self.assertRaisesRegex(ValueError, 'selection'):
            self.check()
        self.write('critic-gate.json', gate)
        self.donor.write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            self.check()
        self.donor.write_text(self.donor.name)
        code = self.runtime / 'tetris_neural.py'
        code.write_text('altered implementation')
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.check()

    def test_rejects_rehashed_wrong_mae_target_block_and_missing_stride(self):
        self.make(True)
        temperature, seed = verify.TEMPERATURES[0], verify.SEEDS[0]
        identity = dict(self.identity, temperature=temperature)
        for mutate, expected in ((lambda r:r['mae']['early'].update(donor=999.), 'per-game MAE'),
                                 (lambda r:r.update(reason='target_or_no_move'), 'naturally terminated'),
                                 (lambda r:r['samples'].pop(), 'Missing calibration samples')):
            row = self.row(seed)
            mutate(row)
            self.save(identity, seed, row)
            with self.assertRaisesRegex(ValueError, expected):
                self.check()
        with self.assertRaisesRegex(ValueError, 'top_out'):
            self.save(identity, seed, dict(self.row(seed), reason='target_or_no_move'))
            calibration.cached_games(self.out / 'critic-cache' / verify.digest(identity), identity)

    def test_rejects_spoofed_or_interrupted_fallback_and_ignores_partial_predictions(self):
        gate = self.make(False)
        self.assertEqual(self.check(), gate)
        for changed in (dict(selected='donor'), dict(reason='signal_15'), dict(completed={'1.0':2,'0.25':1})):
            self.write('critic-gate.json', dict(gate, **changed))
            with self.assertRaises(ValueError):
                self.check()
        self.write('critic-gate.json', gate)
        self.write('gpu-time.json', [dict(phase='calibration', label='critic', status='closed',
                                        seconds=2., allowance_seconds=120.)])
        with self.assertRaisesRegex(ValueError, 'time limit'):
            self.check()

    def test_time_budget_does_not_allocate_either_cuda_model_or_preparer(self):
        stop = calibration.ev.base.Budget(0.)
        with patch.object(calibration.ev, 'load_model', side_effect=AssertionError('model allocated')), patch.object(
                calibration, 'sample', side_effect=AssertionError('sampler initialized')):
            result = calibration.calibrate(self.out, self.inc, self.donor, 'cuda', stop, self.identity)
        self.assertEqual(result['reason'], 'time_budget')
        self.assertEqual(result['selected'], 'incumbent')
        self.assertEqual(result['completed'], {'1.0':0,'0.25':0})

    def test_import_ignores_exported_runtime_and_preserves_sys_path(self):
        before = list(sys.path)
        with patch.dict(os.environ, TETRIS4D_CALIBRATION_RUNTIME=str(self.runtime)):
            loaded = runpy.run_path(calibration.__file__, run_name='tools._imported_calibration')
        self.assertEqual(sys.path, before)
        self.assertEqual(loaded['RUNTIME'], calibration.ROOT)
        self.assertEqual(Path(loaded['ev'].__file__).resolve(), calibration.ROOT / 'tools/evaluate_afterstate_td.py')


if __name__ == '__main__':
    unittest.main()
