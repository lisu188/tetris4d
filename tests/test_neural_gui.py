from pathlib import Path
import json
import sys
import tempfile
import time
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import tetris_neural as neural

game = neural.game


def run():
    torch.set_num_threads(2)
    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    checks = 0
    def check(value, message):
        nonlocal checks
        if not value:
            raise AssertionError(message)
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / 'smoke-trained.pt'
        config = neural.Config(width=8, blocks=1, shortlist=8, candidates=32)
        model = neural.ColumnPolicy(config)
        state = game.Game(7)
        prepared = neural.prepare(state, 1, config)
        batch = neural.collate([prepared.arrays], torch.device('cpu'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loss = torch.nn.functional.cross_entropy(model(batch)['logits'], torch.tensor([prepared.teacher]))
        loss.backward()
        optimizer.step()
        neural.save_checkpoint(checkpoint, model, optimizer, {'updates': 1})
        original_agent = game.NEURAL_AGENT
        try:
            for width, height in ((1560, 960), (1120, 760)):
                game.NEURAL_AGENT = neural.NeuralAgent(checkpoint, 'cpu', engine=game)
                root = tk.Tk()
                app = game.TetrisApp(root, policy='neural', candidates=config.candidates)
                root.geometry(f'{width}x{height}')
                root.after_cancel(app.tick_id)
                app.tick_id = None
                def ready():
                    deadline = time.monotonic() + 20
                    while app.search is not None and time.monotonic() < deadline:
                        app._advance_search(time.perf_counter() + .008)
                        root.update()
                    app.draw(force=True)
                    root.update()
                    check(app.search is None, 'Neural search did not complete')
                    check(not app.errors, 'Tk callback failed')
                try:
                    ready()
                    check(app.speed.get() == 0, 'Unlimited default changed')
                    check(len(app.well.views) == 16, 'A section is hidden')
                    check(len(app.network.positions) == 464, 'Actual activation maps missing')
                    check(app.decision.policy == 'neural', 'Wrong policy selected')
                    for i in range(8):
                        before = app.game.pieces
                        app.step()
                        ready()
                        check(app.game.pieces == before + 1, 'Step did not commit one rigid move')
                        check(not app.running, 'Step started autoplay')
                    check(app.render_metrics()['neural']['active'], 'Neural provenance absent from telemetry')
                    app.network.tabs.select(1)
                    app.draw(force=True)
                    root.update()
                    check(bool(app.network.legacy.positions), 'Legacy trace tab missing')
                    app.network.tabs.select(0)
                    app.policy.set('sustained')
                    app.change_policy()
                    ready()
                    check('not active' in app.network.detail.get(), 'Stored neural trace labelled as current policy')
                    app.policy.set('neural')
                    app.change_policy()
                    ready()
                    app.human.set(True)
                    app.change_mode()
                    before = app.game.pieces
                    app.tick()
                    root.after_cancel(app.tick_id)
                    app.tick_id = None
                    check(app.game.pieces == before, 'Human mode auto-committed')
                    app.human.set(False)
                    app.change_mode()
                    app.reset()
                    app.game.board = (0, 10) + (2,) * (game.AREA - 2)
                    app.game.current = 'I'
                    app.target.set('1')
                    app.refresh_plan()
                    ready()
                    app.planned = game.choose_sustained_move(app.game.board, 'I', target=1, candidates=64).move
                    app.running = True
                    app.place()
                    check(app.running and app.game.layers == 1 and app.target.get() == '2', 'Neural mode failed to continue/retarget after a clear')
                    app.running = False
                    ready()
                finally:
                    app.close()
        finally:
            game.NEURAL_AGENT = original_agent
    return {'passed': True, 'checks': checks, 'sizes': [[1560, 960], [1120, 760]],
            'environment': 'CPU PyTorch on Linux virtual display; not a CUDA or WSLg benchmark'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
