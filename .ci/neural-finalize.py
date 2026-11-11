from pathlib import Path
import ast
p = Path('tetris_neural.py')
s = p.read_text(encoding='utf-8')
a = '    def __len__(self):\n        return len(self.samples)\n\n    def __getitem__(self, index):'
b = '    def __len__(self):\n        return len(self.samples)\n\n    def sample_batch(self, size):\n        anchor = random.randrange(len(self.samples))\n        shard, row = self.samples[anchor]\n        offset = anchor - row\n        count = self.shards[shard][\'samples\']\n        return [self[offset + random.randrange(count)] for _ in range(size)]\n\n    def __getitem__(self, index):'
assert s.count(a) == 1
s = s.replace(a, b)
a = '            rows = [data[random.randrange(len(data))] for _ in range(args.batch)]'
assert s.count(a) == 1
s = s.replace(a, '            rows = data.sample_batch(args.batch)')
ast.parse(s)
p.write_text(s, encoding='utf-8')
p = Path('docs/neural-policy.md')
s = p.read_text(encoding='utf-8')
a = 'Training defaults to FP32, 16 boards per minibatch and four accumulation steps (effective batch 64).'
assert s.count(a) == 1
s = s.replace(a, a + ' Each minibatch randomly selects one shard proportional to its sample count and samples within it, avoiding repeated decompression of many shards per minibatch. Every sample retains equal marginal probability; samples within a minibatch are correlated, and the four accumulation steps independently select shards. The CPU shard cache is bounded to two shards.')
p.write_text(s, encoding='utf-8')
