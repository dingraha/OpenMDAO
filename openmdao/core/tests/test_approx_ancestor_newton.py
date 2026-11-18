"""Residual solves above approximating groups must not use explicit semi-total blocks."""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class _State(om.ImplicitComponent):
    """R(w, x) = 4w - 2x; the component can also solve itself."""

    def setup(self):
        self.add_input('x', 1.5)
        self.add_output('w', 1.0)
        self.declare_partials('w', 'w', val=4.0)
        self.declare_partials('w', 'x', val=-2.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['w'] = 4.0 * outputs['w'] - 2.0 * inputs['x']

    def solve_nonlinear(self, inputs, outputs):
        outputs['w'] = inputs['x'] / 2.0

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['w'] = d_residuals['w'] / 4.0
        else:
            d_residuals['w'] = d_outputs['w'] / 4.0


class _Scale(om.ExplicitComponent):
    """y = 3w with constant analytic partials."""

    def setup(self):
        self.add_input('w', 1.0)
        self.add_output('y', 1.0)
        self.declare_partials('y', 'w', val=3.0)

    def compute(self, inputs, outputs):
        outputs['y'] = 3.0 * inputs['w']


class _FailingState(_State):
    """Raise after suspension has begun, while component partials are linearized."""

    def linearize(self, inputs, outputs, partials):
        raise om.AnalysisError('deliberate linearization failure')


def _build(linear, method, nonlinear='newton', nested=False, failing=False):
    p = om.Problem(reports=False)
    p.model.add_subsystem('ivc', om.IndepVarComp('x', 1.5))
    outer = p.model.add_subsystem('outer', om.Group())
    child = outer.add_subsystem('child', om.Group())
    cls = _FailingState if failing else _State
    child.add_subsystem('state', cls())
    child.approx_totals(method=method)
    if nested:
        outer.approx_totals(method=method)
    p.model.connect('ivc.x', 'outer.child.state.x')
    p.model.add_subsystem('downstream', _Scale())
    p.model.connect('outer.child.state.w', 'downstream.w')
    if linear == 'direct':
        p.model.linear_solver = om.DirectSolver(assemble_jac=False)
    elif linear == 'assembled':
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
    elif linear == 'block':
        p.model.linear_solver = om.LinearBlockGS(atol=1e-12, rtol=1e-12, maxiter=20,
                                               iprint=-1, err_on_non_converge=True)
    else:
        p.model.linear_solver = om.ScipyKrylov(assemble_jac=linear == 'krylov_assembled',
                                             atol=1e-12, rtol=1e-12, iprint=-1,
                                             err_on_non_converge=True)
    if nonlinear == 'newton':
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=1,
                                                 atol=1e-12, rtol=1e-12, iprint=-1,
                                                 err_on_non_converge=True)
    else:
        p.model.nonlinear_solver = om.BroydenSolver(compute_jacobian=True, maxiter=10,
                                                  atol=1e-12, rtol=1e-12, iprint=-1,
                                                  err_on_non_converge=True)
        if nonlinear == 'broyden_states':
            p.model.nonlinear_solver.options['state_vars'] = [
                'outer.child.state.w', 'downstream.y']
    return p


class TestApproxAncestorNewton(unittest.TestCase):

    def test_newton_residuals_and_repeated_totals(self):
        for linear in ('direct', 'assembled', 'krylov', 'krylov_assembled', 'block'):
            for method in ('fd', 'cs'):
                for mode in ('fwd', 'rev'):
                    for nested in (False, True):
                        with self.subTest(linear=linear, method=method, mode=mode, nested=nested):
                            p = _build(linear, method, nested=nested)
                            p.setup(mode=mode, force_alloc_complex=True)
                            for x in (1.5, 3.0):
                                p.set_val('ivc.x', x)
                                p.run_model()
                                self.assertEqual(p.model.nonlinear_solver._iter_count, 1)
                                assert_near_equal(p.get_val('outer.child.state.w'), [x / 2.0],
                                                  1e-12)
                                assert_near_equal(p.get_val('downstream.y'), [1.5 * x], 1e-12)
                                self.assertTrue(p.model.outer.child._owns_approx_jac)
                                self.assertEqual(p.model.outer._owns_approx_jac, nested)
                                for _ in range(2):
                                    J = p.compute_totals(of=['downstream.y'], wrt=['ivc.x'])
                                    assert_near_equal(J['downstream.y', 'ivc.x'], [[1.5]], 2e-6)

    def test_broyden_inverse(self):
        for nonlinear in ('broyden', 'broyden_states'):
            for linear in ('direct', 'assembled'):
                for method in ('fd', 'cs'):
                    with self.subTest(nonlinear=nonlinear, linear=linear, method=method):
                        p = _build(linear, method, nonlinear=nonlinear)
                        p.setup(force_alloc_complex=True)
                        for x in (1.5, 3.0):
                            p.set_val('ivc.x', x)
                            p.run_model()
                            assert_near_equal(p.get_val('downstream.y'), [1.5 * x], 1e-12)
                            J = p.compute_totals(of=['downstream.y'], wrt=['ivc.x'])
                            assert_near_equal(J['downstream.y', 'ivc.x'], [[1.5]], 2e-6)

    def test_restoration_after_linearization_error(self):
        for linear in ('direct', 'assembled'):
            with self.subTest(linear=linear):
                p = _build(linear, 'fd', nested=True, failing=True)
                p.setup()
                p.final_setup()
                groups = [p.model, p.model.outer, p.model.outer.child]
                info = [g._subjacs_info for g in groups]
                with self.assertRaisesRegex(om.AnalysisError, 'deliberate linearization failure'):
                    p.run_model()
                self.assertFalse(p.model._owns_approx_jac)
                self.assertTrue(p.model.outer._owns_approx_jac)
                self.assertTrue(p.model.outer.child._owns_approx_jac)
                for group, original in zip(groups, info):
                    self.assertIs(group._subjacs_info, original)
                self.assertIsNone(p.model._assembled_jac)
                self.assertIsNone(p.model.linear_solver._assembled_jac)

    def test_nested_context_restores_assembled_views(self):
        p = _build('assembled', 'fd')
        p.setup()
        p.run_model()
        p.compute_totals(of=['downstream.y'], wrt=['ivc.x'])
        model = p.model
        totals_jac = model._assembled_jac
        residual_jac = model._residual_assembled_jac
        info = model._subjacs_info
        with model._residual_jacobian_context():
            self.assertIs(model._get_assembled_jac(), residual_jac)
            with model._residual_jacobian_context():
                self.assertIs(model._get_assembled_jac(), residual_jac)
            self.assertIs(model._get_assembled_jac(), residual_jac)
            self.assertFalse(model.outer.child._owns_approx_jac)
        self.assertIs(model._assembled_jac, totals_jac)
        self.assertIs(model.linear_solver._assembled_jac, totals_jac)
        self.assertIs(model._subjacs_info, info)
        self.assertTrue(model.outer.child._owns_approx_jac)

    def test_resetup(self):
        p = _build('assembled', 'cs', nested=True)
        for _ in range(2):
            p.setup(force_alloc_complex=True)
            p.set_val('outer.child.state.w', 1.0)
            p.run_model()
            assert_near_equal(p.get_val('outer.child.state.w'), [0.75], 1e-12)
            J = p.compute_totals(of=['downstream.y'], wrt=['ivc.x'])
            assert_near_equal(J['downstream.y', 'ivc.x'], np.array([[1.5]]), 2e-6)


if __name__ == '__main__':
    unittest.main()
