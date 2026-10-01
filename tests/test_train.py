"""Loss formula and a short Equinox training run."""

import jax
import jax.numpy as jnp

from discrete_average_generator import (
    AverageGeneratorMLP,
    generator_matching_loss,
    mixture_rate,
    train_average_generator,
)
from discrete_average_generator.schedules import kappa_dot_linear, kappa_linear


def test_loss_matches_stopped_target_formula():
    key = jax.random.PRNGKey(0)
    vocab = 4
    model = AverageGeneratorMLP(vocab, key)
    posterior = jax.nn.softmax(jnp.array([0.2, -0.1, 0.4, 0.0]))
    x = jnp.asarray(1)
    t = jnp.asarray(0.2)
    r = jnp.asarray(0.5)

    value = generator_matching_loss(
        model, x, t, r, posterior, kappa_linear, kappa_dot_linear
    )

    def q_of(tt):
        return model(x, tt, r)

    q_hat, dq_hat = jax.jvp(q_of, (t,), (jnp.ones_like(t),))
    alpha = r - t
    u_theta = (q_hat - jax.nn.one_hot(x, vocab)) / alpha
    jvp_t = dq_hat / alpha
    rate = 1.0 / (1.0 - t)
    u_tgt = mixture_rate(posterior, rate)[x] + (alpha / 1.0) * jvp_t
    expected = jnp.mean((u_theta - jax.lax.stop_gradient(u_tgt)) ** 2)
    assert jnp.allclose(value, expected, atol=1e-6)
    assert jnp.isfinite(value)

    # Quotient-rule tangent would cancel u and must not be what the loss uses.
    quotient_jvp = dq_hat / alpha + u_theta / alpha
    quotient_tgt = mixture_rate(posterior, rate)[x] + alpha * quotient_jvp
    quotient_loss = jnp.mean((u_theta - quotient_tgt) ** 2)
    assert float(jnp.abs(value - quotient_loss)) > 1e-4


def test_endpoint_fallback_is_posterior_mse():
    model = AverageGeneratorMLP(4, jax.random.PRNGKey(1))
    posterior = jax.nn.softmax(jnp.arange(4.0))
    x = jnp.asarray(3)
    t = jnp.asarray(0.4)
    value = generator_matching_loss(
        model, x, t, t, posterior, kappa_linear, kappa_dot_linear
    )
    q_hat = model(x, t, t)
    expected = jnp.mean((q_hat - posterior) ** 2)
    assert jnp.allclose(value, expected, atol=1e-6)


def test_training_loss_decreases():
    model, losses, _posterior = train_average_generator(
        jax.random.PRNGKey(0),
        n_steps=30,
        vocab=4,
    )
    assert model.mlp.width_size == 32
    assert model.mlp.depth == 2
    assert model.mlp.layers[0].out_features == 32
    assert model.mlp.layers[1].out_features == 32
    assert model.mlp.layers[2].out_features == 4
    assert losses.shape == (30,)
    assert bool(jnp.all(jnp.isfinite(losses)))
    assert float(losses[-1]) < float(losses[0])
