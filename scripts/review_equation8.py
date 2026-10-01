"""Re-run the review pair for equation 8.

Prints eq8_gap and loss_formula_gap at t=0.11, r=0.83 on the quadratic
scaled generator. loss_formula is equation8_target, the function the training
loss calls. A dropped transport term shows up as loss_formula_gap around 0.3.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from discrete_average_generator import (
    average_generator,
    equation8_target,
    integrate_transition,
    scaled_generator,
    self_consistency_rhs,
)
from discrete_average_generator.schedules import kappa_dot_quadratic, kappa_quadratic


def main():
    t = jnp.asarray(0.11, dtype=jnp.float64)
    r = jnp.asarray(0.83, dtype=jnp.float64)

    def rate_fn(s):
        return scaled_generator(s, kappa_quadratic, kappa_dot_quadratic)

    transition = integrate_transition(t, r, rate_fn, n_steps=256)
    u_bar = average_generator(transition, r, t)
    q_t = rate_fn(t)
    du = self_consistency_rhs(u_bar, q_t, r, t)
    eq8 = equation8_target(u_bar, du, q_t, r, t)
    dropped = q_t - (t - r) * du
    eq8_gap = float(jnp.max(jnp.abs(u_bar - eq8)))
    loss_formula_gap = float(jnp.max(jnp.abs(u_bar - eq8)))
    dropped_gap = float(jnp.max(jnp.abs(eq8 - dropped)))
    print(f"eq8_gap {eq8_gap}")
    print(f"loss_formula_gap {loss_formula_gap}")
    print(f"dropped_transport_gap {dropped_gap}")
    if eq8_gap > 1e-12 or loss_formula_gap > 1e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
