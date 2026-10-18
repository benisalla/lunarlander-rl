import torch
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from lunarlander_rl import networks as N


def test_shapes():
    x = torch.randn(5, 8)
    assert N.QNet(8, 4, (16,)).forward(x).shape == (5, 4)
    assert N.DuelingQNet(8, 4, (16, 16))(x).shape == (5, 4)
    assert N.QCritic(8, 2, (16,))(x, torch.randn(5, 2)).shape == (5,)
    assert N.ValueNet(8, (16,))(x).shape == (5,)
    assert N.CategoricalActor(8, 4, (16,)).dist(x).sample().shape == (5,)
    d = N.GaussianActor(8, 2, (16,)).dist(x)
    assert d.sample().shape == (5, 2) and d.log_prob(d.sample()).shape == (5,)
    a = N.DeterministicActor(8, 2, (16,))(x)
    assert a.shape == (5, 2) and a.abs().max() <= 1


def test_dueling_advantages_have_zero_mean_offset():
    net = N.DuelingQNet(8, 4, (16, 16))
    x = torch.randn(3, 8)
    q = net(x)
    v = net.value(net.trunk(x))
    assert torch.allclose(q.mean(1, keepdim=True), v, atol=1e-5)


def test_tanh_gaussian_log_prob_matches_torch_transform():
    torch.manual_seed(0)
    actor = N.TanhGaussianActor(8, 2, (16,))
    x = torch.randn(6, 8)
    torch.manual_seed(1)
    a, logp = actor.sample(x)
    mu, log_std = actor(x)
    ref = TransformedDistribution(Normal(mu, log_std.exp()), [TanhTransform(cache_size=1)])
    assert a.abs().max() < 1
    assert torch.allclose(logp, ref.log_prob(a.clamp(-0.999999, 0.999999)).sum(-1), atol=1e-3)


def test_soft_update():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    w_a, w_b = a.weight.detach().clone(), b.weight.detach().clone()
    N.soft_update(a, b, 0.25)
    assert torch.allclose(a.weight, 0.75 * w_a + 0.25 * w_b)
