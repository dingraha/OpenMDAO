module ICompTest

using OpenMDAOCore

struct SimpleImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    a::TF
end

function OpenMDAOCore.setup(self::SimpleImplicit)
 
    n = self.n
    inputs = [
        VarData("x"; shape=n, val=[2.0]),
        VarData("y"; shape=(n,), val=3.0)]

    outputs = [
        VarData("z1"; shape=(n,), val=fill(2.0, n)),
        VarData("z2"; shape=n, val=3.0)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("z1", "x"; rows=rows, cols=cols),
        PartialsData("z1", "y"; rows, cols),
        PartialsData("z1", "z1"; rows, cols),
        PartialsData("z2", "x"; rows, cols),
        PartialsData("z2", "y"; rows, cols),          
        PartialsData("z2", "z2"; rows, cols)
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::SimpleImplicit, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]

    return nothing
end

function OpenMDAOCore.linearize!(self::SimpleImplicit, inputs, outputs, partials)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = -1.0
    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y

    @. partials["z2", "z2"] = -1.0
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0

    return nothing
end

struct SolveNonlinearImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    a::TF
end

function OpenMDAOCore.setup(self::SolveNonlinearImplicit)
 
    n = self.n
    inputs = [
        VarData("x"; shape=n, val=[2.0]),
        VarData("y"; shape=(n,), val=3.0)]

    outputs = [
        VarData("z1"; shape=(n,), val=fill(2.0, n)),
        VarData("z2"; shape=n, val=3.0)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("z1", "x"; rows=rows, cols=cols),
        PartialsData("z1", "y"; rows, cols),
        PartialsData("z1", "z1"; rows, cols),
        PartialsData("z2", "x"; rows, cols),
        PartialsData("z2", "y"; rows, cols),          
        PartialsData("z2", "z2"; rows, cols)
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::SolveNonlinearImplicit, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]

    return nothing
end

function OpenMDAOCore.solve_nonlinear!(self::SolveNonlinearImplicit, inputs, outputs)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = a*x*x + y*y
    @. outputs["z2"] = a*x + y

    return nothing
end

function OpenMDAOCore.linearize!(self::SolveNonlinearImplicit, inputs, outputs, partials)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = -1.0
    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y

    @. partials["z2", "z2"] = -1.0
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0

    return nothing
end

struct MatrixFreeImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    a::TF
end

function OpenMDAOCore.setup(self::MatrixFreeImplicit)
 
    n = self.n
    inputs = [
        VarData("x"; shape=n, val=[2.0]),
        VarData("y"; shape=(n,), val=3.0)]

    outputs = [
        VarData("z1"; shape=(n,), val=fill(2.0, n)),
        VarData("z2"; shape=n, val=3.0)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("z1", "x"; rows=rows, cols=cols),
        PartialsData("z1", "y"; rows, cols),
        PartialsData("z1", "z1"; rows, cols),
        PartialsData("z2", "x"; rows, cols),
        PartialsData("z2", "y"; rows, cols),          
        PartialsData("z2", "z2"; rows, cols)
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::MatrixFreeImplicit, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]

    return nothing
end

function OpenMDAOCore.apply_linear!(self::MatrixFreeImplicit, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    a = self.a
    x, y = inputs["x"], inputs["y"]
    z1, z2 = outputs["z1"], outputs["z2"]

    xdot = get(d_inputs, "x", nothing)
    ydot = get(d_inputs, "y", nothing)
    z1dot = get(d_outputs, "z1", nothing)
    z2dot = get(d_outputs, "z2", nothing)
    Rz1dot = get(d_residuals, "z1", nothing)
    Rz2dot = get(d_residuals, "z2", nothing)

    if mode == "fwd"
        # In forward mode, the goal is to calculate the derivatives of the
        # residuals wrt an upstream input, given the inputs and outputs and the
        # derivatives of the inputs and outputs wrt the upstream input.
        if Rz1dot !== nothing
            fill!(Rz1dot, 0)
            if xdot !== nothing
                @. Rz1dot += 2*a*x*xdot
            end
            if ydot !== nothing
                @. Rz1dot += 2*y*ydot
            end
            if z1dot !== nothing
                @. Rz1dot += -z1dot
            end
        end
        if Rz2dot !== nothing
            fill!(Rz2dot, 0)
            if xdot !== nothing
                @. Rz2dot += a*xdot
            end
            if ydot !== nothing
                @. Rz2dot += ydot
            end
            if z2dot !== nothing
                @. Rz2dot += -z2dot
            end
        end
    elseif mode == "rev"
        # In reverse mode, the goal is to calculate the derivatives of an
        # downstream output wrt the inputs and outputs, given the derivatives of
        # the downstream output wrt the residuals.
        if xdot !== nothing
            fill!(xdot, 0)
            if Rz1dot !== nothing
                @. xdot += 2*a*x*Rz1dot
            end
            if Rz2dot !== nothing
                @. xdot += a*Rz2dot
            end
        end
        if ydot !== nothing
            fill!(ydot, 0)
            if Rz1dot !== nothing
                @. ydot += 2*y*Rz1dot
            end
            if Rz2dot !== nothing
                @. ydot += Rz2dot
            end
        end
        if z1dot !== nothing
            fill!(z1dot, 0)
            if Rz1dot !== nothing
                @. z1dot += -Rz1dot
            end
        end
        if z2dot !== nothing
            fill!(z2dot, 0)
            if Rz2dot !== nothing
                @. z2dot += -Rz2dot
            end
        end
    end
end

struct GuessNonlinearImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    xguess::TF
    xlower::TF
    xupper::TF
end

function OpenMDAOCore.setup(self::GuessNonlinearImplicit)
    n = self.n
    xlower = self.xlower
    xupper = self.xupper
    inputs = [
        VarData("a"; shape=n, val=[2.0]),
        VarData("b"; shape=(n,), val=3.0),
        VarData("c"; shape=(n,), val=3.0)]

    outputs = [VarData("x"; shape=n, val=3.0, lower=xlower, upper=xupper)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("x", "a"; rows=rows, cols=cols),
        PartialsData("x", "b"; rows, cols),
        PartialsData("x", "c"; rows, cols),
        PartialsData("x", "x"; rows, cols),
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::GuessNonlinearImplicit, inputs, outputs, residuals)
    a = inputs["a"]
    b = inputs["b"]
    c = inputs["c"]
    x = outputs["x"]
    Rx = residuals["x"]

    @. Rx = a*x^2 + b*x + c

    return nothing
end

function OpenMDAOCore.linearize!(self::GuessNonlinearImplicit, inputs, outputs, partials)
    a = inputs["a"]
    b = inputs["b"]
    c = inputs["c"]
    x = outputs["x"]

    dRx_da = partials["x", "a"]
    dRx_db = partials["x", "b"]
    dRx_dc = partials["x", "c"]
    dRx_dx = partials["x", "x"]

    @. dRx_da = x^2
    @. dRx_db = x
    @. dRx_dc = 1
    @. dRx_dx = 2*a*x + b

    return nothing
end

function OpenMDAOCore.guess_nonlinear!(self::GuessNonlinearImplicit, inputs, outputs, residuals)
    @. outputs["x"] = self.xguess
    return nothing
end

end # module
