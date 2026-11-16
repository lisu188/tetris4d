"""CPU-only harness contracts; CUDA probes require explicit reviewed CLI use."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode=True
SOURCE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gpu_probe_harness',SOURCE/'tools/benchmark_gpu_training.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class ProbeContractTests(unittest.TestCase):
 def test_trainer_progress_same_identity_monotonic(self):
  before={'pid':12,'start_ticks':'4','runner_updates':70,'placements':200000}
  self.assertFalse(h.progress_ok(before,dict(before)))
  self.assertTrue(h.progress_ok(before,dict(before,runner_updates=71,placements=208192)))
  with self.assertRaises(ValueError):h.progress_ok(before,dict(before,pid=13))
  with self.assertRaises(ValueError):h.progress_ok(before,dict(before,placements=0))
 def test_array_parity_rejects_shape_dtype_and_values(self):
  expected=np.asarray([1,2],dtype=np.int32);h.compare_arrays(expected,expected.copy(),'ints')
  for wrong in (expected.astype(np.int64),expected.reshape(1,2),np.asarray([1,3],dtype=np.int32)):
   with self.assertRaises((ValueError,AssertionError)):h.compare_arrays(expected,wrong,'ints')
 def test_float_parity_is_exact(self):
  expected=np.asarray([.1],dtype=np.float32)
  with self.assertRaises(AssertionError):h.compare_arrays(expected,np.nextafter(expected,np.float32(1.)),'score')
 def test_outputs_never_overwrite(self):
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'report.json';h.write_new(path,{'passed':True})
   with self.assertRaises(FileExistsError):h.write_new(path,{'passed':False})
   self.assertTrue(json.loads(path.read_text())['passed'])
 def test_containment_and_symlink_rejection(self):
  with tempfile.TemporaryDirectory() as td,tempfile.TemporaryDirectory() as other:
   root=Path(td);h.contained(root/'report.json',root)
   with self.assertRaises(ValueError):h.contained(Path('relative'),root)
   with self.assertRaises(ValueError):h.contained(Path(other)/'report.json',root)
   (root/'link').symlink_to(other)
   with self.assertRaises(ValueError):h.contained(root/'link/report.json',root)
 def test_fixture_hashes_checked(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);np.savez_compressed(root/'arrays.npz',x=np.arange(4));(root/'cases.json').write_text('[]')
   manifest={'files':[h.info(root/'arrays.npz'),h.info(root/'cases.json')]}
   (root/'manifest.json').write_text(json.dumps(manifest));_,cases,arrays=h.load_fixtures(root)
   self.assertEqual(cases,[]);np.testing.assert_array_equal(arrays['x'],np.arange(4));arrays.close()
   (root/'cases.json').write_text('[{}]')
   with self.assertRaises(ValueError):h.load_fixtures(root)
 def test_default_cli_is_inactive(self):
  with patch.object(h,'gpu_telemetry',side_effect=AssertionError('GPU touched')),patch.object(h,'trainer_snapshot',side_effect=AssertionError('trainer touched')):
   self.assertEqual(h.main([]),0)
 def test_bounds_fail_before_cuda(self):
  with patch.object(h,'gpu_telemetry',side_effect=AssertionError('GPU touched')):
   for args in (['--memory-mib','1025'],['--steps','65'],['--seconds','301']):
    with self.assertRaises(ValueError):h.main(args)
 def test_supervisor_report_prevents_probe_retry(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);output=root/'failed.json';output.with_suffix('.supervisor.json').write_text('{}')
   with patch.object(h,'VALIDATION',root),patch.object(h,'trainer_snapshot',side_effect=AssertionError('trainer touched')):
    with self.assertRaisesRegex(ValueError,'already exists'):
     h.main(['--run-cuda','--output',str(output),'--fixtures',str(root/'fixtures')])
 def test_steady_needs_enough_timed_calls_before_cuda(self):
  with patch.object(h,'gpu_telemetry',side_effect=AssertionError('GPU touched')):
   with self.assertRaisesRegex(ValueError,'eight timed calls'):
    h.main(['--run-cuda','--suite','steady','--steps','4'])
 def test_metric_tail_ignores_partial_lines(self):
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'metrics.jsonl'
   path.write_text('{"event":"update","runner_updates":7}\n{"event":"selection_progress","pieces":8}\n{"event":')
   result=h.metric_tail(path)
   self.assertEqual([row['event'] for row in result['recent_events']],['selection_progress','update'])
   self.assertEqual(result['bytes'],path.stat().st_size)
 def test_existing_benchmark_timing_prevents_probe(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);output=root/'benchmark.json';output.with_suffix('.timing.json').write_text('{}')
   with patch.object(h,'VALIDATION',root),patch.object(h,'trainer_snapshot',side_effect=AssertionError('trainer touched')):
    with self.assertRaisesRegex(ValueError,'timing report already exists'):
     h.main(['--run-cuda','--suite','benchmark','--output',str(output),'--fixtures',str(root/'fixtures')])
 def test_normal_and_artifact_worktree_layouts(self):
  root=Path('/project/tetris4d')
  self.assertEqual(h.artifact_root(root),root/'survival-training.artifacts')
  for name in ('gpu-source','source'):
   self.assertEqual(h.artifact_root(root/'survival-training.artifacts'/name),root/'survival-training.artifacts')
 def test_no_trainer_signal_code(self):
  # The only timeout is subprocess.run's private child. No process control API
  # is used by this source to terminate/change the production process.
  import ast
  tree=ast.parse((SOURCE/'tools/benchmark_gpu_training.py').read_text())
  forbidden={'kill','killpg','send_signal','terminate','reset','device_reset'}
  calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
  self.assertFalse(forbidden.intersection(calls))


if __name__=='__main__':unittest.main(verbosity=2)
