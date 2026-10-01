"""Exactly enumerated D=2, alphabet-3 Potts distribution.

q(x) is proportional to exp(1.5 * sum_{i<j} 1_{x_i=x_j} + eps[x]),
with eps = 0.3 * normal frozen at PRNGKey(0).
"""

import jax
import jax.numpy as jnp

ALPHABET = 3
DIMENSION = 2
COUPLING = 1.5
EPS_SCALE = 0.3

# Frozen external field. Not trained. float32 so the draw does not depend on
# whether 64-bit mode was enabled before import.
FROZEN_EPS = EPS_SCALE * jax.random.normal(
    jax.random.PRNGKey(0),
    (ALPHABET ** DIMENSION,),
    dtype=jnp.float32,
)


def enumerate_configs(alphabet=ALPHABET, dimension=DIMENSION):
    """Lexicographic list of configurations, shape (alphabet**dimension, dimension)."""
    axes = [jnp.arange(alphabet) for _ in range(dimension)]
    grids = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([grid.reshape(-1) for grid in grids], axis=-1)


def potts_log_weights(eps=FROZEN_EPS, coupling=COUPLING):
    """Unnormalized log weights. For D=2 the pair sum has a single term."""
    configs = enumerate_configs()
    agree = (configs[:, 0] == configs[:, 1]).astype(eps.dtype)
    return coupling * agree + eps


def potts_distribution(eps=FROZEN_EPS, coupling=COUPLING):
    """Normalized Potts distribution on the 9 configurations. Sums to 1."""
    return jax.nn.softmax(potts_log_weights(eps, coupling))
