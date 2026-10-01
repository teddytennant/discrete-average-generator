"""Loss formula and a short Equinox training run."""

import jax
import jax.numpy as jnp

from discrete_average_generator import (
    AverageGeneratorMLP,
    equation8_target,
    generator_matching_loss,
    mixture_rate,
    mixture_transition,
    train_average_generator,
)
from discrete_average_generator.schedules import (
    kappa_dot_linear,
    kappa_dot_quadratic,
    kappa_linear,
    kappa_quadratic,
    lam,
)


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

    def u_matrix(tt):
        def row(y):
            q_hat = model(y, tt, r)
            eye = jax.nn.one_hot(y, vocab, dtype=q_hat.dtype)
            return (q_hat - eye) / (r - tt)

        return jax.vmap(row)(jnp.arange(vocab))

    u_mat, du_mat = jax.jvp(u_matrix, (t,), (jnp.ones_like(t),))
    q_mat = mixture_rate(posterior, lam(t, kappa_linear, kappa_dot_linear))
    target = equation8_target(u_mat, du_mat, q_mat, r, t)
    expected = jnp.mean((u_mat[x] - jax.lax.stop_gradient(target[x])) ** 2)
    assert jnp.allclose(value, expected, atol=1e-6)
    assert jnp.isfinite(value)

    # The formula without the transport term is not equation 8.
    dropped = q_mat[x] + (r - t) * du_mat[x]
    dropped_loss = jnp.mean((u_mat[x] - jax.lax.stop_gradient(dropped)) ** 2)
    assert float(jnp.abs(value - dropped_loss)) > 1e-6


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


def test_exact_model_loss_is_equation8():
    """A network that emits the true transition has a zero equation-8 residual.

    This is the pair the published loss previously failed: t=0.11, r=0.83,
    quadratic schedule. The identity holds, and the training target must too.
    """
    jax.config.update("jax_enable_x64", True)
    posterior = jnp.array([0.2, 0.3, 0.5])
    t = jnp.asarray(0.11)
    r = jnp.asarray(0.83)

    def model(x, tt, rr):
        return mixture_transition(tt, rr, posterior, kappa_quadratic, kappa_dot_quadratic)[x]

    value = generator_matching_loss(
        model,
        jnp.asarray(0),
        t,
        r,
        posterior,
        kappa_quadratic,
        kappa_dot_quadratic,
    )
    assert float(value) < 1e-10


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
