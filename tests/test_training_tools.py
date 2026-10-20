import sys
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'notebooks'))
from training_tools import Standardizer, ProfileMLP, evaluate, fit, make_profile_problem


class TrainingChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_unequal_batch_average(self):
        x = torch.arange(5.).reshape(-1, 1)
        data = DataLoader(TensorDataset(x, torch.zeros_like(x)), batch_size=2)
        self.assertAlmostEqual(evaluate(nn.Identity(), data, nn.MSELoss())['loss'], 6.)

    def test_scaler_constant_and_inverse(self):
        a = torch.tensor([[1., 4.], [3., 4.]])
        s = Standardizer.fit(a)
        torch.testing.assert_close(s.transform(a), torch.tensor([[-1., 0.], [1., 0.]]))
        torch.testing.assert_close(s.inverse(s.transform(a)), a)

    def test_regression_and_update_budget(self):
        x = torch.linspace(-1, 1, 40).reshape(-1, 1)
        data = DataLoader(TensorDataset(x, 2*x+1), batch_size=10)
        m = nn.Linear(1, 1)
        result = fit(m, data, data, nn.MSELoss(), torch.optim.SGD(m.parameters(), lr=.1), epochs=100, max_updates=101)
        self.assertEqual(result['history'][-1]['updates'], 101)
        self.assertEqual(result['history'][-1]['samples_seen'], 1010)
        self.assertLess(evaluate(m, data, nn.MSELoss())['loss'], 1e-5)
        for name, p in m.state_dict().items():
            torch.testing.assert_close(p, result['best']['model'][name])

    def test_classification_contract(self):
        x = torch.tensor([[-2.], [-1.], [1.], [2.]])
        data = DataLoader(TensorDataset(x, torch.tensor([0, 0, 1, 1])), batch_size=4)
        m = nn.Linear(1, 2)
        fit(m, data, data, nn.CrossEntropyLoss(), torch.optim.SGD(m.parameters(), lr=.2), epochs=40)
        self.assertTrue(torch.equal(m(x).argmax(-1), torch.tensor([0, 0, 1, 1])))

    def test_modes_buffers_and_singleton(self):
        data = make_profile_problem()['datasets']
        m = ProfileMLP(normalization='batch', dropout=.2)
        result = fit(m, DataLoader(data['train'], batch_size=64), DataLoader(data['validation'], batch_size=64), nn.MSELoss(), torch.optim.Adam(m.parameters()), epochs=2)
        self.assertFalse(m.training)
        self.assertTrue(any('running_mean' in key for key in result['best']['model']))
        x = data['validation'].tensors[0]
        with torch.no_grad():
            torch.testing.assert_close(m(x), torch.cat([m(v) for v in x.split(17)]), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(m(x), m(x))
        m = ProfileMLP(normalization='batch')
        singleton = DataLoader(TensorDataset(x[:5], data['validation'].tensors[1][:5]), batch_size=2)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            fit(m, singleton, singleton, nn.MSELoss(), torch.optim.Adam(m.parameters()), epochs=1)

    def test_scheduler_order_and_early_stop(self):
        x = torch.ones(4, 1)
        data = DataLoader(TensorDataset(x, x), batch_size=2)
        m = nn.Linear(1, 1); opt = torch.optim.SGD(m.parameters(), lr=.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.5)
        r = fit(m, data, data, nn.MSELoss(), opt, epochs=5, scheduler=sched,
                metric_fn=lambda p,y: torch.tensor(1.), patience=2)
        self.assertEqual(len(r['history']), 3)
        self.assertEqual(r['best']['epoch'], 1)
        self.assertAlmostEqual(r['best']['optimizer']['param_groups'][0]['lr'], .05)
        self.assertAlmostEqual(r['history'][1]['lr_first'], .05)

    def test_default_patience_and_disable(self):
        data = DataLoader(TensorDataset(torch.ones(4, 1), torch.ones(4, 1)), batch_size=2)
        for options, expected in [({}, 21), ({'patience': None}, 30)]:
            model = nn.Linear(1, 1)
            result = fit(model, data, data, nn.MSELoss(),
                         torch.optim.SGD(model.parameters(), lr=.01), epochs=30,
                         metric_fn=lambda p, y: torch.tensor(1.), **options)
            self.assertEqual(len(result['history']), expected)
            self.assertEqual(result['best']['epoch'], 1)

    def test_synthetic_conservation_and_corruption(self):
        a, b = make_profile_problem(), make_profile_problem(outliers=True)
        torch.testing.assert_close((a['profiles']*a['weights']).sum(1), a['ub'])
        self.assertTrue(torch.isfinite(a['profiles']).all())
        torch.testing.assert_close(a['datasets']['validation'].tensors[1], b['datasets']['validation'].tensors[1])
        self.assertFalse(torch.equal(a['datasets']['train'].tensors[1], b['datasets']['train'].tensors[1]))


if __name__ == '__main__':
    unittest.main()
