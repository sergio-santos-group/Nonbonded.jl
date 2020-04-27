using Base.Cartesian

"""
"""
function naive_aos(state::State{T}, cutoff::T, do_forces::Bool) where {T <: AbstractFloat}
    coords    = state.coords
    forces    = state.forces
    energy    = T(0)
    cutoff_sq = cutoff^2
    for i in 1:3:length(coords)
        for j in (i+3):3:length(coords)
            if do_forces
                forces[1:1+2] = zeros(3, 1)
                fi = zeros(3, 1)
            end
            rij = coords[i:i+2] - coords[j:j+2]
            dij_sq = sum(rij.^2)
            if dij_sq < cutoff_sq
                lj2 = T(1) / dij_sq
                lj6 = lj2*lj2*lj2
                energy += sum(Float32(1)*(lj6*lj6 - lj6))

                if do_forces
                    fc = T(24) * (lj6 - T(2) * lj6 * lj6) / dij_sq
                    fi .+= fc .* rij
                    forces[j:j+2] .-= fc .* rij
                end
            end
        end

        if do_forces
            forces[i:i+2] .+= fi
        end
    end
    
    return energy
end

"""
"""
@generated function serial_no_verlet_AoS(state::State{T}, cutoff::T, ::Type{Val{DoF}}) where {DoF, T <: AbstractFloat}
    quote
        coords = state.coords
        forces = state.forces
        natoms = state.size

        cutsq = convert(T, cutoff*cutoff)   # squared cutoff
        energy = T(0)                       # total energy
        σ = T(1)                            # LJ sigma
        
        @inbounds for i = 1:natoms-1

            # load coordinates for the i-th atom
            @nexprs 3 u -> ri_u = coords[u,i]
            
            #region FORCE_SECTION
            # zero the force accummulator for the i-th atom
            $(if DoF === true
                :(@nexprs 3 u -> fi_u = zero(T))
            end)
            #endregion
            
            for j = i+1:natoms

                # load coordinates for the j-th atom
                # and calculate the ij vector
                @nexprs 3 u -> rij_u = coords[u, j] - ri_u
                
                # calculate the squared distance. Skip
                # if greater than cutoff
                dij_sq = @reduce 3 (+) u -> rij_u*rij_u
                (dij_sq > cutsq) && continue
                
                # LJ potential
                lj2 = σ/dij_sq
                lj6 = lj2*lj2*lj2
                energy += lj6*lj6 - lj6
                
                #region FORCE_SECTION
                $(if DoF === true
                    quote
                        fc = T(24) * (lj6 - T(2) * lj6 * lj6) / dij_sq
                        @nexprs 3 u -> begin
                            # accumulate forces for atom i (in registers)
                            fi_u += fc * rij_u
                            # update forces for atom j
                            forces[u, j] -= fc * rij_u
                        end
                    end
                end)
                #endregion
            
            end
            #region FORCE_SECTION
            # update forces for the i-th atom
            $(if DoF === true
                :(@nexprs 3 u -> forces[u, i] += fi_u)
            end)
            #endregion
        end
        energy
    end
end


"""
"""
@generated function serial_no_verlet_SoA(state::State{T}, cutoff::T, ::Type{Val{DoF}}) where {DoF, T <: AbstractFloat}
    quote
        coords = state.coords
        forces = state.forces
        natoms = state.size

        cutsq = convert(T, cutoff*cutoff)   # squared cutoff
        energy = T(0)                       # total energy
        σ = T(1)                            # LJ sigma
        
        @inbounds for i = 1:natoms-1

            # load coordinates for the i-th atom
            @nexprs 3 u -> ri_u = coords[i, u]
            
            #region FORCE_SECTION
            # zero the force accummulator for the i-th atom
            $(if DoF === true
                :(@nexprs 3 u -> fi_u = zero(T))
            end)
            #endregion
            
            for j = i+1:natoms

                # load coordinates for the j-th atom
                # and calculate the ij vector
                @nexprs 3 u -> rij_u = coords[j, u] - ri_u
                
                # calculate the squared distance. Skip
                # if greater than cutoff
                dij_sq = @reduce 3 (+) u -> rij_u*rij_u
                (dij_sq > cutsq) && continue
                

                # LJ potential
                lj2 = σ/dij_sq
                lj6 = lj2*lj2*lj2
                energy += lj6*lj6 - lj6
                
                #region FORCE_SECTION
                $(if DoF === true
                    quote
                        fc = T(24) * (lj6 - T(2) * lj6 * lj6) / dij_sq
                        @nexprs 3 u -> begin
                            # accumulate forces for atom i (in registers)
                            fi_u += fc * rij_u
                            # update forces for atom j
                            forces[j, u] -= fc * rij_u
                        end
                    end
                end)
                #endregion
            
            end
            #region FORCE_SECTION
            # update forces for the i-th atom
            $(if DoF === true
                :(@nexprs 3 u -> forces[i, u] += fi_u)
            end)
            #endregion
        end
        energy
    end
end

# TODO: The difference between AoS and SoA versions of this funcion are very
# similar. The only thing that changes is the coordinate gathering lines:
# if argmin(collect(size(coords))) == 1
#     return coords[u, i]
# else
#     return coords[i, u]
# end
# Maybe a metaprogramming solution can be employed here