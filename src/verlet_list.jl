using Base.Cartesian

"""
    @reduce(n::Int, op::Symbol, ex::Expr)

        Wrapping macro for @ncall. Is equivalent and generates
        op(ex_1, ..., ex_n)

    Example:

        @reduce 2 (+) u -> v_u^2
        > Generates +(v_1^2, v_2^2)
"""
macro reduce(n::Int, op::Symbol, ex::Expr)
    esc(:(@ncall($n, $op, $ex)))
end


"""
    VerletList(size::Int, capacity::Int, cutoff::Float64, offset::Vector{Int}, list::Vector{Int})
    VerletList(size::Int)

        Hold information regarding the neighbouring particles of each particle
        in the system.
"""
mutable struct VerletList
    size::Int              # number of particles the list refers to 
    capacity::Int          # capacity of list 
    cutoff::Float64        # cutoff (nm)
    offset::Vector{Int}    # pointer particle->list position
    list::Vector{Int}      # nblist
end
VerletList(size::Int) = VerletList(size, size, -1.0, zeros(Int, size), zeros(Int, size))


"""
    resize!(vlist::VerletList, n::Int)

        Resize a Verlet list to hold 'n' particles. If 'n' is higher than the
        current length of the list, appends 0's.

    Example:

        resize!(vl, 8)
"""
Base.resize!(vlist::VerletList, n::Int) = begin
    resize!(vlist.list, n)
    vlist.capacity = n
    vlist
end


"""
    get_coords(coords::Matrix{T}, u::Int, i::Int) where {T <: AbstractFloat}

        Returns, from the coordinates at array position (atom number) 'i', the
        dimendion number 'u' (i.e. dimension X is 1, Y is 2 and Z is 3). Allows
        for correct coordinates retrieval independent of AoS vs SoA format.

    Example:

        get_coords(coords, u, i)
"""
function get_coords(coords::Matrix{T}, u::Int, i::Int) where {T <: AbstractFloat}
    if argmin(collect(size(coords))) == 1
        return coords[u, i]
    else
        return coords[i, u]
    end
end


"""
    update_serial!(vlist::VerletList, coords::Matrix{T}) where {T <: AbstractFloat}

        Updates the Verlet list (using a serial approach) according to the
        defined 'vlist.cutoff' and the given coordinates 'coords' (can be in
        both AoS or SoA formats).

    Example:

        update_serial!(vl, xyz)
"""
@generated function update_serial!(vlist::VerletList, state::State{F, T}) where {T <: AbstractFloat, F <: AbstractMatrix}
    
    quote
        coords = state.coords.values
        @assert vlist.size == max(size(coords)...) "incompatible sizes"
        # println(size(coords))

        # cutoff squared
        cutsq = convert(T, vlist.cutoff*vlist.cutoff)
        
        offset = 1
        natoms = vlist.size
        @inbounds for i = 1:natoms

            vlist.offset[i] = offset
            # @nexprs 3 u -> xi_u = coords[i,u]
            $(if F == AoS
                quote
                    @nexprs 3 u -> xi_u = coords[u, i]
                end
            else
                quote
                    @nexprs 3 u -> xi_u = coords[i, u]
                end
            end)

            for j = (i+1):natoms

                # @nexprs 3 u -> vij_u = coords[j, u] - xi_u
                $(if F == AoS
                    quote
                        @nexprs 3 u -> vij_u = coords[u, j] - xi_u
                    end
                else
                    quote
                        @nexprs 3 u -> vij_u = coords[j, u] - xi_u
                    end
                end)
                dij_sq = @reduce 3 (+) u -> vij_u*vij_u
                
                if dij_sq < cutsq
                    vlist.list[offset] = j
                    offset += 1
                    if offset == vlist.capacity
                        resize!(vlist, vlist.capacity + natoms)
                    end
                end
            end

            vlist.list[offset] = -1
            offset += 1
            if (i < natoms) && (offset == vlist.capacity)
                resize!(vlist, vlist.capacity + natoms)
            end

        end
        
        vlist
    end
end

# TODO: update_simd4!
# TODO: update_simd8!