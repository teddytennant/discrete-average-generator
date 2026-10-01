"""Equinox network and a short training loop for the matching loss."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from discrete_average_generator.loss import generator_matching_loss
from discrete_average_generator.schedules import kappa_dot_linear, kappa_linear


class AverageGeneratorMLP(eqx.Module):
    """Two hidden layers of width 32, then a vocabulary-sized logit readout.

    equinox.nn.MLP depth=2 is Linear(in, 32), Linear(32, 32), Linear(32, out):
    two hidden layers of width 32. The module returns a probability q_hat.
    """

    mlp: eqx.nn.MLP
    vocab: int = eqx.field(static=True)

    def __init__(self, vocab, key):
        self.vocab = vocab
        self.mlp = eqx.nn.MLP(
            in_size=vocab + 2,
            out_size=vocab,
            width_size=32,
            depth=2,
            key=key,
        )

    def __call__(self, x, t, r):
        one_hot = jax.nn.one_hot(x, self.vocab)
        times = jnp.stack([jnp.asarray(t), jnp.asarray(r)]).astype(one_hot.dtype)
        logits = self.mlp(jnp.concatenate([one_hot, times]))
        return jax.nn.softmax(logits)


def train_average_generator(key, n_steps=30, vocab=4, lr=1e-4, batch=64):
    """Fit the matching loss for n_steps on one frozen batch.

    Returns the trained model, the per-step loss vector, and the posterior
    that defines Q_denoise. The PRNG key fixes initialization and the batch.
    """
    key_model, key_post, key_batch = jax.random.split(key, 3)
    posterior = jax.nn.softmax(jax.random.normal(key_post, (vocab,)))
    model = AverageGeneratorMLP(vocab, key_model)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    key_x, key_t, key_gap = jax.random.split(key_batch, 3)
    xs = jax.random.randint(key_x, (batch,), 0, vocab)
    ts = jax.random.uniform(key_t, (batch,), minval=0.1, maxval=0.35)
    rs = jnp.minimum(
        ts + jax.random.uniform(key_gap, (batch,), minval=0.05, maxval=0.25),
        jnp.asarray(0.9),
    )

    def batch_loss(net):
        losses = jax.vmap(
            lambda x, t, r: generator_matching_loss(
                net, x, t, r, posterior, kappa_linear, kappa_dot_linear
            )
        )(xs, ts, rs)
        return jnp.mean(losses)

    @eqx.filter_jit
    def step(net, state):
        value, grads = eqx.filter_value_and_grad(batch_loss)(net)
        updates, state = optimizer.update(grads, state, eqx.filter(net, eqx.is_array))
        net = eqx.apply_updates(net, updates)
        return net, state, value

    losses = []
    for _ in range(n_steps):
        model, opt_state, value = step(model, opt_state)
        losses.append(value)
    return model, jnp.stack(losses), posterior
