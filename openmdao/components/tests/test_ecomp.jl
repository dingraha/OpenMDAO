module EComp1Test

using OpenMDAOCore

struct EComp1 <: OpenMDAOCore.AbstractExplicitComp end

function OpenMDAOCore.setup(self::EComp1)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::EComp1, inputs, outputs)
    outputs["y"][1] = 2*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::EComp1, inputs, partials)
    partials["y", "x"][1] = 4*inputs["x"][1]
    @show 4*inputs["x"]
    return nothing
end

struct EComp2 <: OpenMDAOCore.AbstractExplicitComp
    a::Float64
end

function OpenMDAOCore.setup(self::EComp2)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::EComp2, inputs, outputs)
    outputs["y"][1] = 2*self.a*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::EComp2, inputs, partials)
    partials["y", "x"][1] = 4*self.a*inputs["x"][1]
    return nothing
end

struct EComp3 <: OpenMDAOCore.AbstractExplicitComp
    a::Vector{Float64}
end

EComp3(n::Integer) = EComp3(range(1.0, n; length=n) |> collect)

function OpenMDAOCore.setup(self::EComp3)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::EComp3, inputs, outputs)
    outputs["y"][1] = 2*self.a[1]*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::EComp3, inputs, partials)
    partials["y", "x"][1] = 4*self.a[1]*inputs["x"][1]
    return nothing
end

end # module
