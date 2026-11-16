"""Deterministic input/update parity; CUDA cases require an explicit idle window."""
import copy
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_survival_v2 as v2


class RolloutCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, device='cpu', cached=False, trainer_class=None):
        torch.manual_seed(321)
        neural=v2.neural
        model=neural.ColumnPolicy(neural.Config(width=8,blocks=1,detach_critic=True,centered_critic=True)).to(device)
        rng=np.random.default_rng(82)
        records=[]
        for count in (2,5,3,4):
            valid=np.ones(count,dtype=np.bool_)
            if count>2:valid[1]=False
            mask=np.zeros((count,neural.COLUMNS),dtype=np.bool_);mask[:,:4]=True
            records.append({'boards':np.zeros(4096,dtype=np.int32),'targets':np.asarray(0,dtype=np.int64),
                            'context':rng.normal(0,.1,neural.CONTEXT).astype(np.float32),
                            'features':rng.normal(0,.1,(count,neural.FEATURES)).astype(np.float32),
                            'indices':rng.integers(0,4096,(count,neural.COLUMNS),dtype=np.int64),
                            'column_mask':mask,'valid':valid,
                            'teacher_prior':np.linspace(0,-1,count,dtype=np.float32)})
        with torch.no_grad():
            batch=neural.collate(records,torch.device(device));prediction=model(batch)
            logp=v2.legacy.categorical_logp(prediction['logits'],batch['valid']).cpu()
        rollout=v2.Rollout(records,[0]*4,[0]*4,np.asarray([-1,.5,1,-.5],dtype=np.float32),
                           np.asarray([.1,.2,.3,.4],dtype=np.float32),
                           [logp[i,:len(row['features'])].clone() for i,row in enumerate(records)],
                           prediction['value'].cpu().numpy(),np.asarray([-1,.5,1,-.5],dtype=np.float32),
                           np.asarray([False,False,False,True]),np.zeros(4,dtype=np.bool_),{})
        cls=trainer_class or v2.SurvivalTrainer
        trainer=cls.__new__(cls);trainer.model=model;trainer.device=torch.device(device)
        trainer.args=SimpleNamespace(batch=2,optimizer_batch=4,ppo_epochs=2,target_kl=.9,precision='fp32',
                                     gradient_every=10,value_loss='mse',value_beta=1.,clip_ratio=.2,
                                     imitation=0.,secondary_weight=0.,regret_weight=0.,entropy=.001,
                                     gradient_clip_scope='separate',cache_rollout=cached,rollout_cache_mib=1)
        actor=[p for n,p in model.named_parameters() if not n.startswith(('value.','completion.'))]
        trainer.optimizer=torch.optim.AdamW([{'params':actor,'lr':1e-4,'name':'actor'},
                                             {'params':list(model.value.parameters()),'lr':3e-4,'name':'critic'}],weight_decay=1e-4)
        trainer.scaler=torch.amp.GradScaler(torch.device(device).type,enabled=False)
        trainer.progress={'updates':10,'runner_updates':1,'optimizer_steps':0,'survival_updates':1,
                          'warmup_updates':0,'kl_stopped_updates':0}
        trainer.events=[];trainer.emit=lambda event,**fields:trainer.events.append((event,fields))
        return trainer,rollout

    def assert_state_equal(self,left,right,tolerance=0.):
        self.assertEqual(left.progress,right.progress)
        for name,value in left.model.state_dict().items():
            torch.testing.assert_close(value,right.model.state_dict()[name],rtol=tolerance,atol=tolerance,msg=name)
        a,b=left.optimizer.state_dict(),right.optimizer.state_dict()
        self.assertEqual(a['param_groups'],b['param_groups'])
        self.assertEqual(a['state'].keys(),b['state'].keys())
        for index,state in a['state'].items():
            for key,value in state.items():
                torch.testing.assert_close(value,b['state'][index][key],rtol=tolerance,atol=tolerance)

    def parity(self,device='cpu',tolerance=0.):
        plain,rollout=self.fixture(device)
        cached,_=self.fixture(device,True)
        torch.manual_seed(777);plain.update(rollout,False)
        torch.manual_seed(777);cached.update(rollout,False)
        self.assert_state_equal(plain,cached,tolerance)
        self.assertIsNone(cached._rollout_cache)
        self.assertTrue(cached.events[-1][1]['rollout_cache']['enabled'])
        self.assertFalse(plain.events[-1][1]['rollout_cache']['enabled'])
        self.assertAlmostEqual(plain.events[-1][1]['final_exact_kl'],cached.events[-1][1]['final_exact_kl'],places=6)

    def test_ragged_padding_invalid_holes_old_inf_and_targets_exact(self):
        trainer,rollout=self.fixture()
        cache=v2.RolloutCache(rollout,trainer.device)
        self.assertEqual(cache.bytes,cache.required_bytes(rollout))
        trainer._rollout_cache=cache
        for indices in ([0],[2,0],[3,1],[1,0,3]):
            batch,values=cache.select(indices)
            expected=v2.neural.collate([rollout.records[i] for i in indices],trainer.device)
            for key in expected:
                torch.testing.assert_close(batch[key],expected[key],rtol=0,atol=0)
                self.assertTrue(batch[key].is_contiguous())
            old=v2.legacy.SurvivalTrainer.old_batch(trainer,rollout,indices,batch['valid'].shape[1])
            torch.testing.assert_close(values['old'],old,rtol=0,atol=0)
            self.assertTrue(torch.isneginf(values['old'][~batch['valid']]).all())
            torch.testing.assert_close(values['advantage'],torch.as_tensor(rollout.advantages[indices]),rtol=0,atol=0)
            torch.testing.assert_close(values['returns'],torch.as_tensor(rollout.returns[indices]),rtol=0,atol=0)
            torch.testing.assert_close(values['labels'],torch.tensor([rollout.labels[i] for i in indices]),rtol=0,atol=0)

    def test_cache_construction_consumes_no_rng(self):
        trainer,rollout=self.fixture()
        before=(torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate())
        v2.RolloutCache(rollout,trainer.device)
        self.assertTrue(torch.equal(before[0],torch.get_rng_state()))
        after=np.random.get_state();self.assertEqual(before[1][0],after[0]);np.testing.assert_array_equal(before[1][1],after[1]);self.assertEqual(before[1][2:],after[2:])
        self.assertEqual(before[2],random.getstate())

    def test_rollout_cli_is_not_consumed_as_cache_option_abbreviation(self):
        args=v2.arguments(['--rollout','3','--rollout-cache-mib','2','--cache-rollout'])
        self.assertEqual(args.rollout,3)
        self.assertEqual(args.rollout_cache_mib,2)
        self.assertTrue(args.cache_rollout)
        self.assertFalse(v2.arguments(['--rollout','3']).cache_rollout)

    def test_full_kl_and_two_epoch_model_adam_trajectory_exact(self):
        self.parity()

    def test_byte_bound_and_oom_fallback_before_optimization(self):
        trainer,rollout=self.fixture(cached=True)
        with patch.object(v2.RolloutCache,'required_bytes',return_value=2*2**20),patch.object(v2.neural,'collate') as collate:
            result=trainer.create_rollout_cache(rollout)
            self.assertTrue(result['requested']);self.assertFalse(result['enabled']);self.assertEqual(result['reason'],'memory_bound');collate.assert_not_called()
        with patch.object(v2,'RolloutCache') as cache:
            cache.required_bytes.return_value=100;cache.side_effect=torch.OutOfMemoryError('synthetic cache allocation')
            result=trainer.create_rollout_cache(rollout)
            self.assertTrue(result['requested']);self.assertFalse(result['enabled']);self.assertEqual(result['reason'],'allocation_failed')
        self.assertIsNone(trainer._rollout_cache);self.assertEqual(trainer.progress['optimizer_steps'],0)

    def test_update_error_releases_cache_and_does_not_retry_optimizer(self):
        trainer,rollout=self.fixture(cached=True)
        def fail(*args):
            self.assertIsNotNone(trainer._rollout_cache)
            raise ValueError('synthetic update failure')
        with patch.object(trainer,'_update',side_effect=fail):
            with self.assertRaisesRegex(ValueError,'synthetic update failure'):trainer.update(rollout,False)
        self.assertIsNone(trainer._rollout_cache);self.assertEqual(trainer.progress['optimizer_steps'],0)

    @unittest.skipUnless(os.environ.get('RUN_TETRIS_CUDA_CACHE_TESTS')=='1' and torch.cuda.is_available(),
                         'CUDA parity requires an explicitly isolated GPU window')
    def test_cuda_ragged_model_and_adam_parity(self):
        self.parity('cuda',1e-6)


if __name__=='__main__':unittest.main()
