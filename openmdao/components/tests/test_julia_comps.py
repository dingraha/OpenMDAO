"""
Unit test for JuliaExplicitComp and JuliaImplicitComp
"""
import os
import time
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

# Should change this to creating a new Julia module once this bug is fixed: https://github.com/cjdoris/PythonCall.jl/issues/170
from juliacall import Main as jl

d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "test_ecomp.jl"))

class TestSimpleJuliaExplicitComp(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        ecomp = jl.ECompTest.EComp1()
        comp = om.JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.run_model()

    def test_results(self):
        p = self.p
        expected = 2*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestJuliaExplicitCompWithOption(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        a = self.a = 0.5
        ecomp = jl.ECompTest.EComp2(a)
        comp = om.JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=True)
        p.set_val("x", 3.0)
        p.run_model()

    def test_results(self):
        p = self.p
        expected = 2*self.a*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=True, out_stream=None, method='cs')

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=12)


class TestJuliaExplicitCompWithLargeOption(unittest.TestCase):

    def setUp(self):
        n_small = self.n_small = 10
        p_small = self.p_small = om.Problem()
        ecomp_small = jl.ECompTest.EComp3(n_small)
        comp_small = om.JuliaExplicitComp(jlcomp=ecomp_small)
        p_small.model.add_subsystem("ecomp", comp_small, promotes_inputs=["x"], promotes_outputs=["y"])
        p_small.setup(force_alloc_complex=True)
        p_small.set_val("x", 3.0)
        p_small.run_model()

        n_big = self.n_big = 1_000_000_000
        p_big = self.p_big = om.Problem()
        ecomp_big = jl.ECompTest.EComp3(n_big)
        comp_big = om.JuliaExplicitComp(jlcomp=ecomp_big)
        p_big.model.add_subsystem("ecomp", comp_big, promotes_inputs=["x"], promotes_outputs=["y"])
        p_big.setup(force_alloc_complex=True)
        p_big.set_val("x", 3.0)
        p_big.run_model()

    def test_results(self):
        for p in [self.p_small, self.p_big]:
            expected = 2*3*p.get_val("x")[0]**2 + 1
            actual = p.get_val("y")[0]
            np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        for p in [self.p_small, self.p_big]:
            np.set_printoptions(linewidth=1024)
            cpd = p.check_partials(compact_print=True, out_stream=None, method='cs')

            for comp in cpd:
                for (var, wrt) in cpd[comp]:
                    np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                                   desired=cpd[comp][var, wrt]['J_fd'],
                                                   decimal=12)

    def test_timings(self):
        time_avg = []
        n_samples = 1000
        for p in [self.p_small, self.p_big]:
            # Just to make sure the Julia JIT is all warmed up.
            p.run_model()
            tavg = 0.0
            for sample in range(n_samples):
                tstart = time.time()
                p.run_model()
                telapsed = time.time() - tstart
                tavg += telapsed/n_samples

            time_avg.append(tavg)

        # Compare the average timings.
        np.testing.assert_almost_equal(time_avg[1]/time_avg[0], 1.0, decimal=1)


if __name__ == '__main__':
    unittest.main()
