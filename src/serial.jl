using Base.Cartesian

"""
    naive(state::State{T}, cutoff::T, ::do_forces::Bool) where {DoF, T <: AbstractFloat}

        Calculates Lennard Jones energy of a State in a basic and inneficient
        way. If 'do_forces' is set to True, also calculate and update forces.
        Non bonded interactions are only considered for neighbouring atoms
        bellow the defined 'cutoff' (in nm). Returns the energy value.
            
	Example:
	
        naive(state, 1.2, true)
"""
function naive(state::State{F, T}, cutoff::T, do_forces::Bool) where {F <: AbstractMatrix, T <: AbstractFloat}
    coords    = state.coords
    forces    = state.forces
    energy    = T(0)
    cutoff_sq = cutoff^2
    for i in 1:(length(coords) - 1)
        if do_forces
            forces[i] = zeros(3, 1) # ERROR
            fi = zeros(3, 1) # ERROR
        end
        for j in (i+1):length(coords)
            rij = coords[i] - coords[j]
            dij_sq = sum(rij.^2)
            if dij_sq < cutoff_sq
                lj2 = T(1) / dij_sq
                lj6 = lj2*lj2*lj2
                energy += T(1)*(lj6*lj6 - lj6)

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
    serial_no_verlet(state::State{T}, cutoff::T, ::Type{Val{DoF}}) where {DoF, T <: AbstractFloat}

        Calculates Lennard Jones energy of a State. If Val{DoF} is set to True,
        also calculate and update forces. Non bonded interactions are only
        considered for neighbouring atoms bellow the defined 'cutoff' (in nm).
        Returns the energy value.
            
	Example:
	
        serial_no_verlet(state, 1.2, Val{true})
"""
@generated function serial(state::State{F, T}, cutoff::T, ::Type{Val{DoF}}) where {DoF, F <: AbstractMatrix, T <: AbstractFloat}
    quote
        coords = state.coords
        forces = state.forces
        natoms = state.n

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


@generated function serial(state::State{F, T}, vlist::VerletList, ::Type{Val{DoF}}) where {DoF, F <: AbstractMatrix, T <: AbstractFloat}
    quote
        coords = state.coords
        forces = state.forces
        natoms = state.n

        cutsq = convert(T, vlist.cutoff*vlist.cutoff)   # squared cutoff
        energy = T(0)                                   # total energy
        σ = T(1)                                        # LJ sigma
        
        @inbounds for i = 1:natoms-1

            # ptr -> location of the first neighbor of atom i
            # ptr_stop -> location of the last neighbor of atom i
            ptr = vlist.offset[i]
            if vlist.list[ptr] < 1
                continue
            end
            ptr_stop = vlist.offset[i+1]-2

            # load coordinates for the i-th atom
            @nexprs 3 u -> ri_u = coords[i, u]
            
            #region FORCE_SECTION
            # zero the force accummulator for the i-th atom
            $(if DoF === true
                :(@nexprs 3 u -> fi_u = zero(T))
            end)
            #endregion
            
            # for j = i+1:natoms
            while ptr <= ptr_stop
                j = vlist.list[ptr]

                # load coordinates for the j-th atom
                # and calculate the ij vector
                @nexprs 3 u -> rij_u = coords[j, u] - ri_u
                
                # calculate the squared distance. Skip
                # if greater than cutoff
                dij_sq = @reduce 3 (+) u -> rij_u*rij_u
                # (dij_sq > cutsq) && continue
                
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

                ptr += 1
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