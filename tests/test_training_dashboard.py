import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SPEC = importlib.util.spec_from_file_location('dashboard',Path(__file__).resolve().parents[1]/'tools/training_dashboard.py')
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root/'run'; self.run.mkdir()
        self.data = self.root/'dashboard'
        self.metrics = self.run/'metrics.jsonl'

    def tearDown(self):
        self.temp.cleanup()

    def emit(self,event,**fields):
        row = {'event':event,'utc':'2026-11-15T12:00:00Z','elapsed_seconds':12.5,'session':'test',**fields}
        with self.metrics.open('a') as stream: stream.write(json.dumps(row)+'\n')

    def artifact(self,path,value):
        path.write_text(json.dumps(value))

    def test_empty_and_partial_live_line(self):
        collector = dashboard.Collector(self.run,self.data); collector.poll()
        self.assertEqual(collector.snapshot()['phase'],'waiting_training')
        self.metrics.write_text('{"event":"update"')
        collector.poll(); self.assertEqual(collector.offset,0)
        with self.metrics.open('a') as stream: stream.write(',"utc":"2026-11-15T12:00:00Z","runner_updates":1}\n')
        collector.poll(); self.assertEqual(collector.snapshot()['progress']['runner_updates'],1)
        self.assertFalse(collector.snapshot()['errors'])

    def test_calibration_backfill_and_current_game(self):
        self.emit('calibration_progress',game=0,total_games=6,pieces=512)
        self.emit('calibration_game',seed=3300000,pieces=777,terminated=True,truncated=False,split='train')
        self.emit('calibration_progress',game=1,total_games=6,pieces=1024)
        collector = dashboard.Collector(self.run,self.data); collector.poll(); status = collector.snapshot()
        self.assertEqual(status['phase'],'calibration')
        self.assertEqual(status['calibration']['completed_games'],1)
        self.assertEqual(status['calibration']['current_game'],2)
        self.assertEqual(status['calibration']['current_pieces'],1024)
        self.assertEqual(status['calibration_games'][0]['seed'],3300000)
        self.assertEqual(status['source']['offset_bytes'],self.metrics.stat().st_size)
        self.assertEqual(len(status['series']),3)

    def test_fitting_update_episode_and_best_selection(self):
        self.emit('critic_calibration',fit_updates=25,mse=.25,constant_mse=.9)
        self.emit('calibration_complete',passed=True,fit_updates=300,validation={'mse':.2,'constant_mse':.9})
        self.emit('update',runner_updates=2,placements=4096,optimizer_steps=12,finished_episodes=1,clears=4,
                  entropy=.8,final_exact_kl=.01,actor_diagnostics={'h2_saturated_fraction':.1,'h2_candidate_std':.2,'lower_primary_probability':.03})
        self.emit('episode_end',episode=0,seed=500000,pieces=1500,hyperlayers=3,stop_reason='planner_blocked')
        self.emit('selection_complete',score=[1400,1000,1300],best_updated=True)
        self.emit('selection_complete',score=[900,500,800],best_updated=False)
        collector = dashboard.Collector(self.run,self.data); collector.poll(); status = collector.snapshot()
        self.assertEqual(status['progress']['placements'],4096)
        self.assertEqual(status['latest']['kl'],.01)
        self.assertEqual(status['latest']['lower_primary_probability'],.03)
        self.assertEqual(status['calibration']['validation_mse'],.2)
        self.assertIsNone(status['calibration']['train_mse'])
        self.assertEqual(status['best_selection']['mean_pieces'],1400)
        self.assertEqual(status['series'][3]['episode_pieces'],1500)
        self.assertIsNone(status['series'][3]['calibration_game_pieces'])

    def test_restart_deduplicates_without_overwriting_history(self):
        self.emit('calibration_game',seed=1,pieces=50,terminated=True,truncated=False,split='train')
        first = dashboard.Collector(self.run,self.data); first.poll(); saved = first.history.read_bytes()
        second = dashboard.Collector(self.run,self.data); second.poll()
        self.assertEqual(second.history.read_bytes(),saved)
        self.assertEqual(second.snapshot()['calibration']['completed_games'],1)
        self.emit('update',runner_updates=1,entropy=.5); second.poll()
        self.assertTrue(second.history.read_bytes().startswith(saved))
        self.assertEqual(len(second.history.read_text().splitlines()),2)

    def test_optional_actor_boolean_preserves_legacy_history_and_manifest(self):
        self.emit('update',runner_updates=1,actor_diagnostics={'actor_changed_from_initial':False})
        first = dashboard.Collector(self.run,self.data); first.poll()
        legacy = json.loads(first.history.read_text())
        legacy.pop('actor_changed_from_initial')
        first.history.write_text(json.dumps(legacy)+'\n')
        history_before = first.history.read_bytes(); manifest_before = first.manifest.read_bytes()
        second = dashboard.Collector(self.run,self.data); second.poll()
        self.assertIs(second.snapshot()['latest']['actor_changed_from_initial'],False)
        self.assertEqual(second.history.read_bytes(),history_before)
        self.assertEqual(second.manifest.read_bytes(),manifest_before)
        self.emit('update',runner_updates=2,actor_diagnostics={'actor_changed_from_initial':True})
        second.poll()
        self.assertTrue(second.history.read_bytes().startswith(history_before))
        self.assertIs(json.loads(second.history.read_text().splitlines()[-1])['actor_changed_from_initial'],True)
        rows = list(csv.DictReader(io.StringIO(second.history_csv().decode())))
        self.assertEqual(rows[0]['actor_changed_from_initial'],'')
        self.assertEqual(rows[1]['actor_changed_from_initial'],'True')
        self.emit('update',runner_updates=3,actor_diagnostics={'actor_changed_from_initial':'false'})
        second.poll()
        self.assertIs(second.snapshot()['latest']['actor_changed_from_initial'],True)
        self.assertIsNone(json.loads(second.history.read_text().splitlines()[-1])['actor_changed_from_initial'])
        third = dashboard.Collector(self.run,self.data); third.poll()
        self.assertIs(third.snapshot()['latest']['actor_changed_from_initial'],True)
        self.assertEqual(third.manifest.read_bytes(),manifest_before)
        self.assertEqual(third.offset,self.metrics.stat().st_size)

    def test_budget_games_and_phase_artifacts(self):
        self.artifact(self.root/'pipeline-status.json',{'phases':{'comparison':{'state':'running'}}})
        collector = dashboard.Collector(self.run,self.data)
        self.assertEqual(collector.snapshot()['phase'],'waiting_comparison')
        self.artifact(self.root/'budget-ledger.json',{'start_epoch':100,'deadline_epoch':200,'start_utc':'start','deadline_utc':'deadline'})
        self.artifact(self.run/'summary.json',{'active_games':[{'episode':2,'seed':9,'pieces':10,'hyperlayers':1}],'placements':20})
        self.emit('update',runner_updates=1); collector.poll()
        status = collector.snapshot()
        self.assertEqual(status['budget']['start_utc'],'start')
        self.assertEqual(status['games'][0]['pieces'],10)
        self.assertEqual(status['progress']['placements'],20)
        self.artifact(self.root/'status.json',{'state':'finished'})
        self.assertEqual(collector.snapshot()['phase'],'complete')

    def test_size_freeze_and_source_change_preserve_evidence(self):
        collector = dashboard.Collector(self.run,self.data)
        old = dashboard.MAX_BYTES
        try:
            dashboard.MAX_BYTES = collector.manifest.stat().st_size+20
            self.emit('update',runner_updates=1); collector.poll()
            self.assertTrue(collector.snapshot()['history']['frozen'])
            self.assertTrue(collector.snapshot()['errors'])
        finally: dashboard.MAX_BYTES = old
        self.assertEqual(self.metrics.read_text().count('\n'),1)
        self.metrics.write_text('')
        collector.poll(); self.assertIn('replaced or truncated',' '.join(collector.errors))

    def test_restart_detects_changed_source_and_partial_history(self):
        self.emit('update',runner_updates=1)
        first = dashboard.Collector(self.run,self.data); first.poll(); saved = first.history.read_bytes()
        self.metrics.write_text('{"event":"other"}\n')
        second = dashboard.Collector(self.run,self.data); second.poll()
        self.assertTrue(second.snapshot()['history']['frozen'])
        self.assertEqual(second.history.read_bytes(),saved)
        with first.history.open('ab') as stream: stream.write(b'{"partial":')
        third = dashboard.Collector(self.run,self.data)
        self.assertTrue(third.frozen_reason)

    def test_invalid_input_and_data_directory_are_explicit(self):
        with self.assertRaises(ValueError): dashboard.Collector(self.run,self.run/'dashboard')
        self.metrics.write_text('bad json\n')
        collector = dashboard.Collector(self.run,self.data); collector.poll()
        self.assertTrue(collector.snapshot()['errors'])
        with self.assertRaises(ValueError): dashboard.Collector(self.root/'other-run',self.data)

    def test_http_allowlist_and_downloads(self):
        self.emit('update',runner_updates=1,entropy=.75)
        collector = dashboard.Collector(self.run,self.data); collector.poll()
        static = self.root/'static'; static.mkdir(); (static/'index.html').write_text('<h1>Local dashboard</h1>')
        with self.assertRaises(ValueError): dashboard.make_server(collector,'0.0.0.0',0,static)
        server = dashboard.make_server(collector,port=0,static_dir=static)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        base = 'http://127.0.0.1:'+str(server.server_port)
        try:
            with urlopen(base+'/api/status') as response:
                self.assertEqual(json.load(response)['latest']['entropy'],.75)
            with urlopen(base+'/api/history.csv') as response:
                self.assertIn('attachment',response.headers['Content-Disposition'])
                rows = list(csv.DictReader(io.StringIO(response.read().decode())))
                self.assertEqual(rows[0]['runner_updates'],'1')
            with urlopen(base+'/api/history.jsonl') as response:
                self.assertEqual(response.read(),collector.history.read_bytes())
            with urlopen(base+'/') as response: self.assertIn(b'Local dashboard',response.read())
            for path in ('/../metrics.jsonl','/api/../metrics.jsonl','/metrics.jsonl','/%2e%2e/secret'):
                with self.assertRaises(HTTPError) as caught: urlopen(base+path)
                self.assertEqual(caught.exception.code,404)
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base+'/api/status',headers={'Host':'example.com'}))
            self.assertEqual(caught.exception.code,403)
        finally: server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
