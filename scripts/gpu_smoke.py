"""One short train on whatever device JAX sees. Exit 1 if the loss does not fall."""

import jax
from discrete_average_generator import train_average_generator

print("jax", jax.__version__, jax.devices())
_model, losses, _posterior = train_average_generator(
    jax.random.PRNGKey(0),
    n_steps=40,
    vocab=4,
    lr=1e-4,
    batch=64,
)
start = float(losses[0])
end = float(losses[-1])
print("loss_start", start)
print("loss_end", end)
if not (end < start):
    raise SystemExit(1)
print("dag gpu smoke ok")
