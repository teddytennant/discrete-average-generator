"""One-step sampler against an exact average generator."""

import jax
import jax.numpy as jnp
import pytest

from discrete_average_generator import (
    average_generator,
    clip_and_renormalize,
    mixture_transition,
    sample_next,
    sampling_distribution,
)
from discrete_average_generator.schedules import (
    kappa_dot_linear,
    kappa_dot_quadratic,
    kappa_linear,
    kappa_quadratic,
)


def _tv(p, q):
    return 0.5 * jnp.sum(jnp.abs(p - q), axis=-1)


@pytest.mark.parametrize(
    "kappa,kappa_dot",
    [(kappa_linear, kappa_dot_linear), (kappa_quadratic, kappa_dot_quadratic)],
)
def test_exact_sampler_matches_transition(kappa, kappa_dot):
    posterior = jax.nn.softmax(jnp.array([0.1, -0.3, 0.8, 0.0]))
    t = jnp.asarray(0.2)
    r = jnp.asarray(0.7)
    transition = mixture_transition(t, r, posterior, kappa, kappa_dot)
    u_bar = average_generator(transition, r, t)
    vocab = posterior.shape[0]

    distances = []
    for x in range(vocab):
        probs = sampling_distribution(u_bar[x], x, r, t)
        # Exact rows are nonnegative, so the clip-and-renormalize guard is a no-op.
        raw = transition[x]
        assert jnp.allclose(probs, raw, atol=1e-6)
        distances.append(_tv(probs, raw))
        token = sample_next(u_bar[x], x, r, t, jax.random.PRNGKey(x))
        assert int(token) in range(vocab)
    assert float(jnp.max(jnp.stack(distances))) < 1e-6


def test_negative_row_is_clipped_and_renormalized():
    raw = jnp.array([0.5, -0.2, 0.7])
    cleaned = clip_and_renormalize(raw)
    expected = jnp.array([0.5, 0.0, 0.7])
    expected = expected / jnp.sum(expected)
    assert jnp.allclose(cleaned, expected, atol=1e-6)
    assert jnp.allclose(jnp.sum(cleaned), 1.0)
