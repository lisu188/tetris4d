import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
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

    def test_reference_and_bootstrapped_value_metrics_preserve_existing_provenance(self):
        self.emit('update',runner_updates=1,reference_kl=.003,final_reference_kl=.004,
                  reference_kl_weight=.05,final_exact_kl=.002,
                  critic_bootstrapped_diagnostics={'explained_variance':.98})
        first = dashboard.Collector(self.run,self.data); first.poll()
        latest = first.snapshot()['latest']
        self.assertEqual(latest['kl'],.002)
        self.assertEqual(latest['reference_kl'],.003)
        self.assertEqual(latest['final_reference_kl'],.004)
        self.assertEqual(latest['reference_kl_weight'],.05)
        self.assertEqual(latest['critic_bootstrapped_explained_variance'],.98)
        # Emulate an older collector's history without optional measurements.
        legacy = json.loads(first.history.read_text())
        for key in dashboard.OPTIONAL_NUMBERS:
            legacy.pop(key)
        first.history.write_text(json.dumps(legacy)+'\n')
        saved_history = first.history.read_bytes(); saved_manifest = first.manifest.read_bytes()
        second = dashboard.Collector(self.run,self.data); second.poll()
        self.assertEqual(second.snapshot()['latest'],latest)
        self.assertEqual(second.history.read_bytes(),saved_history)
        self.assertEqual(second.manifest.read_bytes(),saved_manifest)
        # Zero is a real coefficient and negative EV is a real measurement.
        self.emit('update',runner_updates=2,reference_kl=0.,final_reference_kl=0.,reference_kl_weight=0.,
                  critic_bootstrapped_diagnostics={'explained_variance':-.5})
        second.poll()
        rows = list(csv.DictReader(io.StringIO(second.history_csv().decode())))
        self.assertEqual(rows[0]['final_reference_kl'],'')
        self.assertEqual(rows[1]['reference_kl_weight'],'0.0')
        self.assertEqual(rows[1]['critic_bootstrapped_explained_variance'],'-0.5')
        self.assertEqual(second.snapshot()['latest']['final_reference_kl'],0.)
        third = dashboard.Collector(self.run,self.data); third.poll()
        self.assertEqual(third.snapshot()['latest'],second.snapshot()['latest'])
        self.assertEqual(third.manifest.read_bytes(),saved_manifest)
        self.assertEqual(third.snapshot()['progress']['runner_updates'],2)
        # A bounded raw backfill may visit an older event after loading the newest history.
        third.apply_optional({**legacy, 'final_reference_kl':.004, 'reference_kl_weight':.05})
        self.assertEqual(third.snapshot()['latest']['final_reference_kl'],0.)
        self.assertEqual(third.snapshot()['latest']['reference_kl_weight'],0.)

    def test_prior_reference_measurements_do_not_leak_into_unregularized_run(self):
        self.emit('update',runner_updates=1,final_reference_kl=.02,reference_kl_weight=.1,
                  critic_bootstrapped_diagnostics={'explained_variance':.9})
        first = dashboard.Collector(self.run,self.data); first.poll()
        new_run = self.root/'unregularized'; new_run.mkdir()
        (new_run/'metrics.jsonl').write_text(json.dumps({'event':'update','utc':'2026-11-15T13:00:00Z',
                                                       'runner_updates':2,'entropy':.8})+'\n')
        second = dashboard.Collector(new_run,self.root/'new-dashboard',self.data); second.poll()
        self.assertEqual(second.snapshot()['latest']['entropy'],.8)
        for key in dashboard.OPTIONAL_NUMBERS:
            self.assertIsNone(second.snapshot()['latest'][key])
        self.assertEqual(second.snapshot()['series'][0]['reference_kl_weight'],.1)
        self.assertEqual(second.snapshot()['series'][0]['final_reference_kl'],.02)

    def test_optional_measurements_reject_invalid_values_and_keep_missing_empty(self):
        self.emit('update',runner_updates=1,reference_kl=True,final_reference_kl=float('nan'),
                  reference_kl_weight='0.2',critic_bootstrapped_diagnostics={'explained_variance':float('inf')})
        collector = dashboard.Collector(self.run,self.data); collector.poll()
        for key in dashboard.OPTIONAL_NUMBERS:
            self.assertIsNone(collector.snapshot()['latest'][key])
            self.assertIsNone(json.loads(collector.history.read_text())[key])
        self.emit('update',runner_updates=2,critic_bootstrapped_diagnostics=[])
        collector.poll()
        self.assertFalse(collector.snapshot()['errors'])

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

    def test_selection_start_phase_danger_starts_and_selection_cap(self):
        collector = dashboard.Collector(self.run,self.data)
        self.artifact(self.run/'manifest.json',{'source_sha256':{'selection_pieces':32000},'calibration_stage':'passed'})
        self.emit('update',runner_updates=1)
        self.emit('episode_end',pieces=900,episode=1,seed=1,start_kind='library',start_pieces=4000)
        self.emit('episode_end',pieces=20000,episode=2,seed=2,start_kind='fresh')
        self.emit('selection_started',runner_updates=1,selection_seeds=8)
        collector.poll()
        status = collector.snapshot()
        self.assertEqual(status['phase'],'selection')
        self.assertEqual(status['selection_cap'],32000)
        self.assertEqual(status['maximum_survival']['danger_start_episodes'],1)
        self.assertEqual(status['maximum_survival']['completed_episodes'],1)
        self.assertEqual(status['maximum_survival']['completed_record_pieces'],20000)
        self.emit('update',runner_updates=2); collector.poll()
        self.assertEqual(collector.snapshot()['phase'],'training')

    def test_running_record_survives_both_plot_sampling_and_deque_eviction(self):
        collector = dashboard.Collector(self.run,self.data)
        for index in range(10025):
            raw={'event':'episode_end' if index in (0,10024) else 'update',
                 'utc':'2026-11-15T12:00:00Z','pieces':9000 if index==0 else 100,
                 'seed':index,'episode':index}
            row=collector.normalize(raw,index,index+1,b'fixture')
            collector.apply(row);collector.rows.append(row)
        status=collector.snapshot()
        self.assertEqual(len(collector.rows),10000)
        self.assertEqual(len(status['series']),3000)
        self.assertEqual(status['series_total_count'],10025)
        self.assertEqual(status['maximum_survival']['completed_record_pieces'],9000)
        self.assertEqual(status['maximum_survival']['completed_episodes'],2)
        self.assertEqual(status['maximum_survival']['records'][0]['seed'],0)

    def test_record_excludes_calibration_selection_baseline_and_teacher_games(self):
        self.emit('episode_end',pieces=100,episode=1,seed=1,teacher_placements=0)
        self.emit('calibration_game',pieces=90000,seed=2,terminated=True,truncated=False)
        self.emit('selection_game',pieces=80000,policy='neural')
        self.emit('episode_end',pieces=70000,policy='teacher')
        self.emit('episode_end',pieces=60000,teacher_placements=2)
        self.emit('episode_end',pieces=50000,policy='sustained')
        self.emit('episode_end',pieces=80,episode=2,seed=3)
        self.artifact(self.run/'summary.json',{'active_games':[{'pieces':150},{'pieces':120}]})
        first=dashboard.Collector(self.run,self.data);first.poll()
        maximum=first.snapshot()['maximum_survival']
        self.assertEqual(maximum['completed_record_pieces'],100)
        self.assertEqual(maximum['completed_episodes'],2)
        self.assertEqual(maximum['active_max_pieces'],150)
        self.assertEqual(maximum['current_max_pieces'],150)
        self.assertEqual(len(maximum['records']),1)
        self.assertEqual(dashboard.Collector(self.run,self.data).snapshot()['maximum_survival'],maximum)

    def test_migrated_run_combines_prior_history_read_only_and_deduplicates(self):
        self.emit('episode_end',pieces=100,episode=1,seed=1)
        first=dashboard.Collector(self.run,self.data);first.poll()
        saved={path:path.read_bytes() for path in (first.history,first.manifest,self.metrics)}
        new_run=self.root/'run-16env';new_run.mkdir();new_data=self.root/'dashboard-16env'
        (new_run/'metrics.jsonl').write_text(json.dumps({'event':'episode_end','utc':'2026-11-15T13:00:00Z',
                                                      'pieces':200,'episode':2,'seed':2})+'\n')
        self.artifact(new_run/'summary.json',{'active_games':[{'pieces':index} for index in range(16)]})
        second=dashboard.Collector(new_run,new_data,self.data);second.poll()
        maximum=second.snapshot()['maximum_survival']
        self.assertEqual(maximum['completed_record_pieces'],200)
        self.assertEqual(maximum['completed_episodes'],2)
        self.assertEqual(len(second.snapshot()['games']),16)
        self.assertEqual(len(second.history.read_text().splitlines()),1)
        combined=list(second.history_lines())
        self.assertEqual(len(combined),2)
        # A repeated identity across prior/current files is omitted from downloads and counters.
        with second.history.open('ab') as stream:stream.write(first.history.read_bytes())
        third=dashboard.Collector(new_run,new_data,self.data);third.poll()
        self.assertEqual(third.snapshot()['maximum_survival']['completed_episodes'],2)
        self.assertEqual(len(list(third.history_lines())),2)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(third.history_csv().decode())))),2)
        self.assertEqual(saved,{path:path.read_bytes() for path in saved})

    def test_requested_optimization_and_valid_amended_budget(self):
        old={'start_epoch':100,'deadline_epoch':28900,'start_utc':'start','deadline_utc':'old'}
        self.artifact(self.root/'budget-ledger.json',old)
        original=self.root/'budget-ledger.json'
        amended={'schema':'tetris4d-survival-budget-v1','start_epoch':100,'deadline_epoch':86500,
                 'start_utc':'start','deadline_utc':'new','fixed':{'hours':24},
                 'amendment':{'authorized_total_hours':24},'original_8h_ledger':{
                     'path':str(original),'bytes':original.stat().st_size,
                     'sha256':dashboard.hashlib.sha256(original.read_bytes()).hexdigest()}}
        self.artifact(self.root/'budget-24h-ledger.json',amended)
        self.artifact(self.root/'optimization-stop-request.json',{'reason':'user requested optimization'})
        self.artifact(self.run/'summary.json',{'reason':'signal_2','active_games':[{'pieces':50}]})
        self.artifact(self.root/'status.json',{'state':'failed'})
        self.artifact(self.root/'pipeline-status.json',{'state':'failed'})
        collector=dashboard.Collector(self.run,self.data)
        self.assertEqual(collector.snapshot()['phase'],'optimizing')
        self.assertEqual(collector.snapshot()['budget']['deadline_utc'],'new')
        amended['deadline_epoch']=999
        self.artifact(self.root/'budget-24h-ledger.json',amended)
        self.assertEqual(collector.snapshot()['budget']['deadline_utc'],'old')
        self.assertTrue(collector.snapshot()['errors'])

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


class NeuronAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audit_path = self.root/'neurons.json'

    @staticmethod
    def report():
        effects = {key: 0.0 for key in (
            'mean_absolute_score_change', 'maximum_absolute_score_change', 'mean_absolute_raw_score_change',
            'top1_changed_fraction', 'mean_absolute_rank_change', 'original_winner_rank_after',
            'original_winner_probability_change', 'probability_total_variation', 'signed_value_change',
            'absolute_value_change', 'mean_absolute_completion_probability_change', 'top1_changes',
            'absolute_value_change_in_placements')}
        groups = [{'id': 'stem', 'units': 1, 'role': 'encoder channel', 'spatial': True, 'sigmoid': True},
                  {'id': 'value', 'units': 1, 'role': 'critic output', 'spatial': False, 'sigmoid': False}]
        units = [{'id': group['id']+'[0]', 'group': group['id'], 'index': 0,
                  'computation': {'formula': 'sigmoid(x)', 'structural_role': group['role']},
                  'activation': {'observations': 16, 'mean': .5, 'std': .1, 'minimum': .2,
                                 'maximum': .8, **({'saturated_fraction': 0.0} if group['sigmoid'] else {})},
                  'intervention': dict(effects), 'explanation': 'Measured midpoint intervention.'} for group in groups]
        return {'schema': 'tetris4d-neuron-audit-v1',
                'checkpoint': {'path': '/protected-best.pt', 'sha256': 'a'*64, 'parameters': 2, 'config': {}},
                'sources': [{'path': 'tools/analyze_neurons.py', 'sha256': 'b'*64}],
                'recordings': [{'path': '/selection.json.gz', 'sha256': 'c'*64, 'verified': True, 'placements': 100}],
                'baseline_forward_parity_passed': True,
                'samples': [{'id': 0, 'recording': 0, 'board_sha256': 'd'*64}],
                'groups': groups, 'units': units,
                'semantics': {'value_code': 'expected_remaining_placements_times_0.001'},
                **{key: {} for key in ('environment', 'input_schema', 'policy', 'sampling', 'intervention')}}

    def write_audit(self, report=None):
        self.audit_path.write_text(json.dumps(self.report() if report is None else report, indent=2)+'\n')
        return self.audit_path.read_bytes()

    def serve(self, audit=None):
        run = self.root/'run'; run.mkdir(exist_ok=True)
        collector = dashboard.Collector(run, self.root/'history')
        static = self.root/'static'; static.mkdir(exist_ok=True)
        for name in ('neurons.html', 'neurons.js', 'neurons.css'):
            (static/name).write_text('test '+name)
        server = dashboard.make_server(collector, port=0, static_dir=static, neuron_audit=audit)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
        thread.start()
        def stop():
            server.shutdown(); server.server_close(); thread.join(timeout=2)
        self.addCleanup(stop)
        return 'http://127.0.0.1:'+str(server.server_port), collector

    def test_http_serves_exact_immutable_audit_and_metadata_without_history_changes(self):
        original = self.write_audit()
        base, collector = self.serve(self.audit_path)
        before = collector.manifest.read_bytes()
        self.audit_path.write_text('{"replaced":true}')
        for route in ('/api/neuron-audit', '/api/neuron-audit?path=/etc/passwd'):
            with urlopen(base+route) as response:
                self.assertEqual(response.headers['Content-Type'], 'application/json')
                self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
                self.assertEqual(response.read(), original)
        with urlopen(base+'/api/status') as response:
            metadata = json.load(response)['neuron_audit']
        self.assertEqual(metadata['units'], 2)
        self.assertEqual(metadata['sampled_states'], 1)
        self.assertEqual(metadata['sha256'], dashboard.hashlib.sha256(original).hexdigest())
        self.assertEqual(metadata['checkpoint_sha256'], 'a'*64)
        self.assertEqual(metadata['bytes'], len(original))
        self.assertTrue(metadata['available'])
        self.assertFalse(collector.history.exists())
        self.assertEqual(collector.manifest.read_bytes(), before)

    def test_unconfigured_audit_is_explicit_and_static_page_is_available(self):
        base, _ = self.serve()
        with self.assertRaises(HTTPError) as caught:
            urlopen(base+'/api/neuron-audit')
        self.assertEqual(caught.exception.code, 404)
        self.assertEqual(json.load(caught.exception), {'error': 'Neuron audit is not configured'})
        with urlopen(base+'/api/status') as response:
            self.assertEqual(json.load(response)['neuron_audit'], {'available': False})
        for route, filename in (('/neurons', 'neurons.html'), ('/neurons.js', 'neurons.js'), ('/neurons.css', 'neurons.css')):
            with urlopen(base+route) as response:
                self.assertEqual(response.read(), ('test '+filename).encode())

    def test_legacy_critic_report_is_served_without_placement_measurements(self):
        report = self.report()
        report['semantics'] = {'value_code': 'discounted_hyperlayers', 'value': 'Discounted future hyperlayer clears.'}
        for unit in report['units']:
            del unit['intervention']['absolute_value_change_in_placements']
        original = self.write_audit(report)
        base, _ = self.serve(self.audit_path)
        with urlopen(base+'/api/neuron-audit') as response:
            self.assertEqual(response.read(), original)

    def test_survival_critic_report_requires_placement_measurements(self):
        report = self.report()
        del report['units'][0]['intervention']['absolute_value_change_in_placements']
        self.write_audit(report)
        with self.assertRaisesRegex(ValueError, 'unit intervention'):
            dashboard.NeuronAudit(self.audit_path)

    @unittest.skipUnless(shutil.which('node'), 'Node is required for the explorer value-unit contract')
    def test_explorer_critic_scaling_follows_report_semantics(self):
        script = r'''
const assert = require('node:assert/strict');
const explorer = require(process.argv[1]);
const survival = {value_code: 'expected_remaining_placements_times_0.001'};
const legacy = {value_code: 'discounted_hyperlayers'};
const unit = {group: 'value', intervention: {absolute_value_change: 0.25, absolute_value_change_in_placements: 250}};
const groups = new Map([['value', {sigmoid: false}]]);
assert.equal(explorer.valueUnits(survival).label, 'placements');
assert.equal(explorer.valueUnits(legacy).label, 'discounted hyperlayers');
assert.equal(explorer.valueUnits({}).label, 'critic units');
assert.equal(explorer.valueUnits(survival).multiplier, 1000);
assert.equal(explorer.valueUnits(legacy).multiplier, 1);
assert.equal(explorer.criticChange(unit, survival), 250);
assert.equal(explorer.criticChange(unit, legacy), 0.25);
assert.equal(explorer.criticChange(unit, {}), 0.25);
assert.equal(explorer.metricValue(unit, groups.get('value'), 'critic', legacy), 0.25);
assert.equal(explorer.metricScale([unit], groups, 'critic', legacy).maximum, 0.25);
delete unit.intervention.absolute_value_change_in_placements;
assert.equal(explorer.criticChange(unit, legacy), 0.25);
assert.equal(explorer.criticChange(unit, survival), 250);
const range = {minimum: 0.1, maximum: 0.5, mean: 0.25, std: 0.1};
assert.equal(explorer.activationRange(range, false, explorer.valueUnits(legacy).multiplier).mean, 0.25);
assert.equal(explorer.activationRange(range, false, explorer.valueUnits(survival).multiplier).mean, 250);
'''
        subprocess.run(['node', '-e', script, str(dashboard.STATIC/'neurons.js')], check=True, capture_output=True, text=True)

    def test_audit_routes_keep_host_guard_and_file_allowlist(self):
        self.write_audit()
        base, _ = self.serve(self.audit_path)
        for route in ('/api/neuron-audit', '/neurons', '/neurons.js'):
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base+route, headers={'Host': 'example.com'}))
            self.assertEqual(caught.exception.code, 403)
        for route in ('/neurons.json', '/api/neuron-audit/../../etc/passwd', '/neurons/../neurons.json',
                      '/%2e%2e/neurons.json', '/api/neuron-audit%2f..%2fsecret'):
            with self.assertRaises(HTTPError) as caught:
                urlopen(base+route)
            self.assertEqual(caught.exception.code, 404)

    def test_malformed_audit_is_rejected_before_server_start(self):
        cases = (
            ('schema', lambda r: r.update(schema='other')),
            ('duplicate unit', lambda r: r['units'].__setitem__(1, r['units'][0])),
            ('group count', lambda r: r['groups'][0].update(units=2)),
            ('nonfinite activation', lambda r: r['units'][0]['activation'].update(mean=float('nan'))),
            ('overflow effect', lambda r: r['units'][0]['intervention'].update(signed_value_change=float('inf'))),
            ('overflow integer', lambda r: r['units'][0]['activation'].update(mean=2**2000)),
            ('invalid sigmoid range', lambda r: r['units'][0]['activation'].update(maximum=2)),
            ('invalid fraction', lambda r: r['units'][0]['intervention'].update(top1_changed_fraction=2)),
            ('inconsistent count', lambda r: r['units'][0]['intervention'].update(top1_changes=1)),
            ('fractional count', lambda r: r['units'][0]['intervention'].update(top1_changes=.5, top1_changed_fraction=.5)),
            ('missing formula', lambda r: r['units'][0]['computation'].pop('formula')),
            ('missing effect', lambda r: r['units'][0]['intervention'].pop('absolute_value_change')),
            ('invalid group type', lambda r: r['units'][0].update(group=[])),
            ('checkpoint checksum', lambda r: r['checkpoint'].update(sha256='bad')),
            ('unverified recording', lambda r: r['recordings'][0].update(verified=False)),
            ('parity failure', lambda r: r.update(baseline_forward_parity_passed=False)),
            ('invalid sample reference', lambda r: r['samples'][0].update(recording=9)),
        )
        for name, change in cases:
            with self.subTest(name=name):
                report = self.report(); change(report); self.write_audit(report)
                with self.assertRaises(ValueError): dashboard.NeuronAudit(self.audit_path)
        self.audit_path.write_text('{"schema":"tetris4d-neuron-audit-v1","overflow":1e999}')
        with self.assertRaisesRegex(ValueError, 'non-finite'): dashboard.NeuronAudit(self.audit_path)
        self.audit_path.write_bytes(b'not JSON')
        with self.assertRaises(ValueError): dashboard.NeuronAudit(self.audit_path)

    def test_missing_oversize_and_nonregular_audit_inputs_are_rejected(self):
        with self.assertRaises(FileNotFoundError): dashboard.NeuronAudit(self.audit_path)
        self.write_audit()
        link = self.root/'linked.json'; link.symlink_to(self.audit_path)
        with self.assertRaises(ValueError): dashboard.NeuronAudit(link)
        fifo = self.root/'audit.pipe'; os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, 'regular file'): dashboard.NeuronAudit(fifo)
        with self.audit_path.open('wb') as stream: stream.truncate(dashboard.MAX_AUDIT_BYTES+1)
        with self.assertRaisesRegex(ValueError, '8 MiB'): dashboard.NeuronAudit(self.audit_path)


if __name__ == '__main__':
    unittest.main()
