from .bayesian_net import Factor, factor_marginalization, factor_product, observe_evidence, compute_joint_distribution
from pandas.util.testing import assert_frame_equal

factor1 = Factor.from_scratch(variables=['v1'], variable_cardinalities=[2], values=[.11, .89])
factor2 = Factor.from_scratch(variables=['v1', 'v2'], variable_cardinalities=[2, 2], values=[.59, .41, .22, .78])
factor3 = Factor.from_scratch(variables=['v2', 'v3'], variable_cardinalities=[2, 2], values=[.39, .61, .06, .94])
marginal = Factor.from_scratch(variables=['v2', 'v3'], variable_cardinalities=[2, 2],
                               values=[.0858, .0468, .1342, .7332])


def test_factor_product():
    factor = Factor.from_scratch(variables=['v2', 'v1'], variable_cardinalities=[2, 2],
                                 values=[.0649, .1958, .0451, .6942])
    assert_frame_equal(factor_product(factor1, factor2).values, factor.values)


def test_factor_marginalisation():
    factor = Factor.from_scratch(variables=['v1'], variable_cardinalities=[2], values=[1., 1.])
    assert_frame_equal(factor_marginalization(factor2, ['v2']).values, factor.values)


def test_observe_evidence():
    evidences = {'v2': 0, 'v3': 1}
    assert_frame_equal(factor1.values, observe_evidence(factor1, evidences).values)
    factor2bis = Factor.from_scratch(variables=['v1', 'v2'], variable_cardinalities=[2, 2], values=[.59, 0, .22, 0])
    assert_frame_equal(factor2bis.values, observe_evidence(factor2, evidences).values)
    factor3bis = Factor.from_scratch(variables=['v2', 'v3'], variable_cardinalities=[2, 2], values=[0, .61, 0, 0])
    assert_frame_equal(factor3bis.values, observe_evidence(factor3, evidences).values)


def test_compute_joint_distribution():
    joint_factor = Factor.from_scratch(variables=['v1', 'v2', 'v3'],
                                       variable_cardinalities=[2, 2, 2],
                                       values=[.025311, .076362, .002706, .041652, .039589, .119438, .042394, .652548])
    assert_frame_equal(joint_factor.values, compute_joint_distribution([factor1, factor2, factor3]).values)
