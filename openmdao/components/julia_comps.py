from types import MethodType

import openmdao.api as om

try:
    # The PythonCall/JuliaCall docs say I should create a new module to avoid polluting Main, but then the seval command below doesn't work.
    # Should change this to creating a new Julia module once this bug is fixed: https://github.com/cjdoris/PythonCall.jl/issues/170
    # import juliacall; jl = juliacall.newmodule("OMJL")
    import juliacall; jl = juliacall.Main
    HAVE_JULIA = True
except ImportError:
    JuliaExplicitComp = None
    JuliaImplicitComp = None
    HAVE_JULIA = False

if HAVE_JULIA:

    # This imports the Julia package OpenMDAOCore:
    jl.seval("using OpenMDAOCore: OpenMDAOCore")


    def _initialize_common(self):
        self.options.declare('jlcomp')


    def _setup_common(self):
        self._jlcomp = self.options['jlcomp']
        input_data, output_data, partials_data = jl.OpenMDAOCore.setup(self._jlcomp)

        for var in input_data:
            self.add_input(var.name, shape=var.shape, val=var.val,
                           units=var.units)

        for var in output_data:
            self.add_output(var.name, shape=var.shape, val=var.val,
                            units=var.units, lower=var.lower, upper=var.upper)

        for data in partials_data:
            self.declare_partials(data.of, data.wrt,
                                  rows=data.rows, cols=data.cols,
                                  val=data.val)


    class JuliaExplicitComp(om.ExplicitComponent):

        def initialize(self):
            _initialize_common(self)

        def setup(self):
            _setup_common(self)

            if jl.OpenMDAOCore.has_compute_partials(self._jlcomp):
                def compute_partials(self, inputs, partials):
                    inputs_dict = juliacall.convert(jl.Dict, dict(inputs))

                    partials_dict = {}
                    for of_wrt in self._declared_partials:
                        partials_dict[of_wrt] = partials[of_wrt]
                    partials_dict = juliacall.convert(jl.Dict, partials_dict)

                    jl.OpenMDAOCore.compute_partials_b(self._jlcomp, inputs_dict, partials_dict)

                # https://www.ianlewis.org/en/dynamically-adding-method-classes-or-class-instanc
                self.compute_partials = MethodType(compute_partials, self)
                # Hmm...
                self._has_compute_partials = True

            if jl.OpenMDAOCore.has_compute_jacvec_product(self._jlcomp):
                def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                    inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                    d_inputs_dict = juliacall.convert(jl.Dict, dict(d_inputs))
                    d_outputs_dict = juliacall.convert(jl.Dict, dict(d_outputs))

                    jl.OpenMDAOCore.compute_jacvec_product_b(self._jlcomp, inputs_dict, d_inputs_dict, d_outputs_dict, mode)

                self.compute_jacvec_product = MethodType(compute_jacvec_product, self)

        def compute(self, inputs, outputs):
            inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
            outputs_dict = juliacall.convert(jl.Dict, dict(outputs))

            jl.OpenMDAOCore.compute_b(self._jlcomp, inputs_dict, outputs_dict)

    class JuliaImplicitComp(om.ImplicitComponent):

        def initialize(self):
            _initialize_common(self)

        def setup(self):
            _setup_common(self)
            #self._julia_apply_nonlinear = get_py2jl_apply_nonlinear(self._jlcomp)
            #self._julia_linearize = get_py2jl_linearize(self._jlcomp)
            #self._julia_guess_nonlinear = get_py2jl_guess_nonlinear(self._jlcomp)
            #self._julia_solve_nonlinear = get_py2jl_solve_nonlinear(self._jlcomp)
            #self._julia_apply_linear = get_py2jl_apply_linear(self._jlcomp)
            ## Trying to avoid the exception
            ##
            ##     RuntimeError: AssembledJacobian not supported for matrix-free subcomponent.
            ##
            ## when the Julia component doesn't implement apply_linear!.
            #if self._julia_apply_linear:
            #    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
            #        inputs_dict = dict(inputs)
            #        outputs_dict = dict(outputs)
            #        d_inputs_dict = dict(d_inputs)
            #        d_outputs_dict = dict(d_outputs)
            #        d_residuals_dict = dict(d_residuals)

            #        self._julia_apply_linear(self._jlcomp,
            #                                 inputs_dict, outputs_dict,
            #                                 d_inputs_dict, d_outputs_dict,
            #                                 d_residuals_dict, mode)
            #    # https://www.ianlewis.org/en/dynamically-adding-method-classes-or-class-instanc
            #    self.apply_linear = MethodType(apply_linear, self)

            if jl.OpenMDAOCore.has_apply_nonlinear(self._jlcomp):
                def apply_nonlinear(self, inputs, outputs, residuals):
                    inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                    outputs_dict = juliacall.convert(jl.Dict, dict(outputs))
                    residuals_dict = juliacall.convert(jl.Dict, dict(residuals))

                    jl.OpenMDAOCore.apply_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict, residuals_dict)

                self.apply_nonlinear = MethodType(apply_nonlinear, self)

            if jl.OpenMDAOCore.has_solve_nonlinear(self._jlcomp):
                def solve_nonlinear(self, inputs, outputs):
                    inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                    outputs_dict = juliacall.convert(jl.Dict, dict(outputs))

                    jl.OpenMDAOCore.solve_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict)

                self.solve_nonlinear = MethodType(solve_nonlinear, self)

            if jl.OpenMDAOCore.has_linearize(self._jlcomp):
                def linearize(self, inputs, outputs, partials):
                    inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                    outputs_dict = juliacall.convert(jl.Dict, dict(outputs))

                    partials_dict = {}
                    for of_wrt in self._declared_partials:
                        partials_dict[of_wrt] = partials[of_wrt]
                    partials_dict = juliacall.convert(jl.Dict, partials_dict)

                    jl.OpenMDAOCore.linearize_b(self._jlcomp, inputs_dict, outputs_dict, partials_dict)

                self.linearize = MethodType(linearize, self)

            if jl.OpenMDAOCore.has_apply_linear(self._jlcomp):
                def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                    inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                    outputs_dict = juliacall.convert(jl.Dict, dict(outputs))
                    d_inputs_dict = juliacall.convert(jl.Dict, dict(d_inputs))
                    d_outputs_dict = juliacall.convert(jl.Dict, dict(d_outputs))
                    d_residuals_dict = juliacall.convert(jl.Dict, dict(d_residuals))

                    jl.OpenMDAOCore.apply_linear_b(self._jlcomp, inputs_dict, outputs_dict,
                            d_inputs_dict, d_outputs_dict, d_residuals_dict, mode)

                self.apply_linear = MethodType(apply_linear, self)

            if jl.OpenMDAOCore.has_solve_linear(self._jlcomp):
                def solve_linear(self, d_outputs, d_residuals, mode):
                    d_outputs_dict = juliacall.convert(jl.Dict, dict(d_outputs))
                    d_residuals_dict = juliacall.convert(jl.Dict, dict(d_residuals))

                    jl.OpenMDAOCore.solve_linear_b(self._jlcomp, d_outputs_dict, d_residuals_dict, mode)

        def _configure(self):
            super()._configure()

            if jl.OpenMDAOCore.has_guess_nonlinear(self._jlcomp):
                def guess_nonlinear(self, inputs, outputs, residuals):
                    inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                    outputs_dict = juliacall.convert(jl.Dict, dict(outputs))
                    residuals_dict = juliacall.convert(jl.Dict, dict(residuals))

                    jl.OpenMDAOCore.guess_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict, residuals_dict)

                self.guess_nonlinear = MethodType(guess_nonlinear, self)
                # Hmm...
                self._has_guess = True
