"""
Unit test for JuliaExplicitComp and JuliaImplicitComp
"""
import os
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

# Should change this to creating a new Julia module once that bug is fixed.
from juliacall import Main as jl

d = os.path.dirname(os.path.abspath(__file__))
jl.include(os.path.join(d, "test_ecomp.jl"))

class TestJuliaExplicitComp(unittest.TestCase):

    def setUp(self):
        p = self.p = om.Problem()
        ecomp = jl.EComp1Test.EComp1()
        comp = om.JuliaExplicitComp(jlcomp=ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup()
        p.set_val("x", 3.0)
        p.run_model()

    def test_results(self):
        p = self.p
        expected = 2*p.get_val("x")[0]**2 + 1
        actual = p.get_val("y")[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_partials(self):
        p = self.p
        expected = 4*p.get_val("x")[0]
        actual = p.compute_totals(of="y", wrt="x")["y", "x"][0,0]
        np.testing.assert_almost_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
