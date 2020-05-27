abstract type AbstractMatrix end

Base.getindex(M::AbstractMatrix, i::UnitRange{Int}) = M.values[i]
Base.length(M::AbstractMatrix) = length(M.values)

"""
    AoSMatrix(values::Matrix{T}) where {T <: AbstractFloat} <: AbstractMatrix

        Holds a Matrix{T} in AoS format.
"""
mutable struct AoSMatrix{T <: AbstractFloat} <: AbstractMatrix
    values::Matrix{T}
end

Base.getindex(M::AoSMatrix{T}, i::Int) where {T <: AbstractFloat} = M.values[:, i]
Base.getindex(M::AoSMatrix{T}, i::Int, j::Int) where {T <: AbstractFloat} = M.values[j, i]
Base.setindex!(M::AoSMatrix{T}, v, i::Int, j::Int) where {T <: AbstractFloat} = setindex!(M.values, v, j, i)


"""
    SoAMatrix(values::Matrix{T}) where {T <: AbstractFloat} <: AbstractMatrix
    
        Holds a Matrix{T} is SoA format.
"""
mutable struct SoAMatrix{T <: AbstractFloat} <: AbstractMatrix
    values::Matrix{T}
end

Base.getindex(M::SoAMatrix{T}, i::Int) where {T <: AbstractFloat} = M.values[i, :]
Base.getindex(M::SoAMatrix{T}, i::Int, j::Int) where {T <: AbstractFloat} = M.values[i, j]
Base.setindex!(M::SoAMatrix{T}, v, i::Int, j::Int) where {T <: AbstractFloat} = setindex!(M.values, v, i, j)


"""
    State(n::Int, coords::Matrix{T}, forces::Matrix{T}) where {T <: AbstractFloat}

        Holds informating regarding the current coordinates and forces on a
        particle system with a given 'n' size. Input can be either in SoA or AoS
        format, in either case coordinates and forces can be retrieved and set
        using state.coords[i, j] (where i is the atom number and j is the
        coordinate position).

    Example:

        state = State(2, [1.0 2.0 3.0; 4.0 5.0 6.0], zeros(3, 2))
"""
mutable struct State{T <: AbstractFloat}
    n::Int
    coords::AbstractMatrix
    forces::AbstractMatrix

    function State(n::Int, coords::Matrix{T}, forces::Matrix{T}) where {T <: AbstractFloat}
        if size(coords) != size(forces)
            error("Size of coords does not match size of forces")
        end
        if argmax(collect(size(coords))) == 2
            println("Creating state in AoS format")
            new{T}(max(size(coords)...), AoSMatrix(coords), AoSMatrix(forces))
        elseif argmax(collect(size(coords))) == 1
            println("Creating state in SoA format")
            new{T}(max(size(coords)...), SoAMatrix(coords), SoAMatrix(forces))
        else
            error("Could not infer the AoS or SoA format of input data.")
        end
    end
end