abstract type AbstractMatrix end

struct SoA <: AbstractMatrix end
struct AoS <: AbstractMatrix end

"""
StateMatrix{F <: AbstractMatrix}(values::Matrix{T}) where {T <: AbstractFloat}
Holds a formated coordinate system. The format is given by the F type,
which can be SoA or AoS. The usage of this typed matrix allows the
correct retrieval of values in an uniform fashion, despite the format.
Example:
_coords = convert(Matrix{Float64}, [1 2 3; 4 5 6])
coords = LennardJones.StateMatrix{LennardJones.AoS, Float64}(_coords)
coords[1, 2] = 2.0
where 1 is the atom position and 2 is the desired coordinate;
"""
struct StateMatrix{F <: AbstractMatrix, T <: AbstractFloat}
    values::Matrix{T}
end

Base.getindex(M::StateMatrix{SoA, T}, i::Int, j::Int) where {T} = M.values[i, j]
Base.getindex(M::StateMatrix{AoS, T}, i::Int, j::Int) where {T} = M.values[j, i]
Base.getindex(M::StateMatrix{SoA, T}, i::Int) where {T} = M.values[i, :]
Base.getindex(M::StateMatrix{AoS, T}, i::Int) where {T} = M.values[:, i]
Base.setindex!(M::StateMatrix{SoA, T}, v, i::Int, j::Int) where {T <: AbstractFloat} = setindex!(M.values, v, i, j)
Base.setindex!(M::StateMatrix{AoS, T}, v, i::Int, j::Int) where {T <: AbstractFloat} = setindex!(M.values, v, j, i)
Base.length(M::StateMatrix) = max(size(M.values)...)

"""
    State(n::Int, coords::Matrix{T}, forces::Matrix{T}) where {T <: AbstractFloat}
    State(n::Int, coords::Matrix{T}) where {T <: AbstractFloat}
        Holds informating regarding the current coordinates and forces on a
        particle system with a given 'n' size. Input can be either in SoA or AoS
        format, in either case coordinates and forces can be retrieved and set
        using state.coords[i, j] (where i is the atom number and j is the
        coordinate position). If the 'forces' matrix is not given, a new Matrix
        is created with size equal to coords set to all zeros.
    Example:
        state = State(2, [1.0 2.0 3.0; 4.0 5.0 6.0], zeros(3, 2))
        state = State(2, [1.0 2.0 3.0; 4.0 5.0 6.0])
"""
mutable struct State{F <: AbstractMatrix, T <: AbstractFloat}
    n::Int
    coords::StateMatrix{F, T}
    forces::StateMatrix{F, T}

    function State(coords::Matrix{T}, forces::Matrix{T}) where {T <: AbstractFloat}
        if size(coords) != size(forces)
            error("Size of coords does not match size of forces")
        end
        if argmax(collect(size(coords))) == 2
            println("Creating state in AoS format")
            new{AoS, T}(max(size(coords)...), StateMatrix{AoS, T}(coords), StateMatrix{AoS, T}(forces))
        elseif argmax(collect(size(coords))) == 1
            println("Creating state in SoA format")
            new{SoA, T}(max(size(coords)...), StateMatrix{SoA, T}(coords), StateMatrix{SoA, T}(forces))
        else
            error("Could not infer the AoS or SoA format of input data.")
        end
    end
end

State(coords::Matrix{T}) where {T <: AbstractFloat} = State(coords, zeros(T, size(coords)))