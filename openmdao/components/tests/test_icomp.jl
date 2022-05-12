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

end # module
