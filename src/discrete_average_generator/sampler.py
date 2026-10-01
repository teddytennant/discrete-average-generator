"""One-step sampler from an average generator row."""

import jax
import jax.numpy as jnp

from discrete_average_generator.reparam import probability_from_average


def transition_from_average(u_bar, r, t):
    """P = I + (r - t) * U. Exact for the clock-time average generator."""
    vocab = u_bar.shape[-1]
    eye = jnp.eye(vocab, dtype=u_bar.dtype)
    return eye + (r - t) * u_bar


def clip_and_renormalize(probs):
    """Clip negative entries to 0 and renormalize.

    The exact-generator case does not need this. A row built from a true
    transition matrix is already nonnegative and sums to 1, so clipping is a
    no-op. Approximate network rows can dip slightly negative.
    """
    clipped = jnp.clip(probs, 0.0)
    total = jnp.sum(clipped, axis=-1, keepdims=True)
    floor = jnp.asarray(1e-12, dtype=probs.dtype)
    return clipped / jnp.maximum(total, floor)


def sampling_distribution(u_row, x, r, t):
    """Categorical distribution used by the one-step sampler."""
    return clip_and_renormalize(probability_from_average(u_row, x, r, t))


def sample_next(u_row, x, r, t, key):
    """Sample the next token from P = I + (r - t) * U."""
    probs = sampling_distribution(u_row, x, r, t)
    safe = jnp.clip(probs, jnp.asarray(1e-30, dtype=probs.dtype))
    return jax.random.categorical(key, jnp.log(safe))
