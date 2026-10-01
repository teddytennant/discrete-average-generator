"""Discrete average-generator matching for finite CTMCs."""

from discrete_average_generator.ctmc import (
    average_generator,
    equation8_target,
    integrate_transition,
    integrate_transition_expm,
    mixture_transition,
    self_consistency_rhs,
)
from discrete_average_generator.loss import generator_matching_loss
from discrete_average_generator.model import AverageGeneratorMLP, train_average_generator
from discrete_average_generator.potts import (
    FROZEN_EPS,
    enumerate_configs,
    potts_distribution,
    potts_log_weights,
)
from discrete_average_generator.rates import mixture_rate, scaled_generator
from discrete_average_generator.reparam import (
    average_from_probability,
    probability_from_average,
)
from discrete_average_generator.sampler import (
    clip_and_renormalize,
    sample_next,
    sampling_distribution,
    transition_from_average,
)
from discrete_average_generator.schedules import (
    kappa_dot_linear,
    kappa_dot_quadratic,
    kappa_linear,
    kappa_quadratic,
    lam,
)

__all__ = [
    "AverageGeneratorMLP",
    "FROZEN_EPS",
    "average_from_probability",
    "average_generator",
    "equation8_target",
    "clip_and_renormalize",
    "enumerate_configs",
    "generator_matching_loss",
    "integrate_transition",
    "integrate_transition_expm",
    "kappa_dot_linear",
    "kappa_dot_quadratic",
    "kappa_linear",
    "kappa_quadratic",
    "lam",
    "mixture_rate",
    "mixture_transition",
    "potts_distribution",
    "potts_log_weights",
    "probability_from_average",
    "sample_next",
    "sampling_distribution",
    "scaled_generator",
    "self_consistency_rhs",
    "train_average_generator",
    "transition_from_average",
]
