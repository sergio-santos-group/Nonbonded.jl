using SIMD


@generated function simd(state::State{F, T}, cutoff::T, ::Type{Val{Forces}}, ::Type{Vec{N,T}}) where {Forces, N, T<:AbstractFloat, F <: AbstractMatrix}

    @assert isa(Forces, Bool) "Forces must be a boolean"
    # @assert T==Float32 "only Float32 is currently supported"
    @assert N ∈ (4,8) "only 4 or 8 wide vectors are allowed"

    quote
    
        σ = T(1)
        ϵ = ones(T, N)
        energy = T(0)
        natoms = state.n
    
        mlane = VecRange{N}(1)  # mask lane = vload(Vec{4, Float64}, X, 1)
        remaining_mask = Vec{4, T}((1, 1, 1, 0))

        coords = view(state.coords.values, :)
        cutsq = convert(T, cutoff*cutoff)

        #region FORCE_SECTION
        $(if Forces === true
            quote
                flane = VecRange{4}(0)
                forces = view(state.forces.values, :)
            end
        end)
        #endregion FORCE_SECTION
        
        $(if N==4
            quote
                m1 = (0, 4, 1, 5)
                m2 = (2, 6, 3, 7)
                mx = (0, 1, 4, 5)
                my = (2, 3, 6, 7)
            end
        else
            quote
                m1 = (0, 4,  8, 12,  1,  5,  9, 13)
                m2 = (2, 6, 10, 14,  3,  7, 11, 15)
                mx = (0, 1,  2,  3,  8,  9, 10, 11)
                my = (4, 5,  6,  7, 12, 13, 14, 15)
                mj = (0, 1,  2,  3,  4,  5,  6,  7)
            end
        end)

        @inbounds for i = 1:natoms-1
            
            $(if Forces=== true
                :(fi = Vec{4,T}(0))
            end)
    
            $(if F == AoS
                quote
                    # Atom number to atom position conversion
                    i1 = i<<2 - (2 + i)
                    vi = vload(Vec{4,T}, coords, i1) # Load XYZ (consecutive)
                end
            else
                quote
                    # Get X, Y and Z positions in SoA format
                    js = Vec((i, i+natoms, i+2*natoms, i))
                    vi = vgather(coords, js) # Gather XYZ
                end
            end)
    
            j = i
            while j+N <= natoms
                
                $(if F == AoS
                    quote
                        # @nexprs $N u -> println(j+(u-1))
                        @nexprs $N u ->  j_u = ((j+u)<<1) + (j+u-2)
                        @nexprs $N u -> vi_u = vload(Vec{4,T}, coords, j_u) - vi    # xi1, yi1, zi1, wi1
                    end
                else
                    quote
                        @nexprs $N u ->  js_u = Vec((j+u, j+u+natoms, j+u+2*natoms, j+u))
                        @nexprs $N u -> vi_u = vgather(coords, js_u) - vi    # xi1, yi1, zi1, wi1
                    end
                end)
                
                $(if N==4
                    quote
                        vs12_m1 = shufflevector(vi_1, vi_2, Val{m1})        # xi1, xi2, yi1, yi2
                        vs34_m1 = shufflevector(vi_3, vi_4, Val{m1})        # xi3, xi4, yi3, yi4
                        vs12_m2 = shufflevector(vi_1, vi_2, Val{m2})        # zi1, zi2, ?i1, ?i2
                        vs34_m2 = shufflevector(vi_3, vi_4, Val{m2})        # zi3, zi4, ?i3, ?i4
                        
                        xij = shufflevector(vs12_m1, vs34_m1, Val{mx})      # xi1, xi2, xi3, xi4
                        yij = shufflevector(vs12_m1, vs34_m1, Val{my})      # yi1, yi2, yi3, yi4
                        zij = shufflevector(vs12_m2, vs34_m2, Val{mx})      # zi1, zi2, zi3, zi4
                    end
                else
                    quote
                        vi12 = shufflevector(vi_1, vi_2, Val{mj})           # xi1, yi1, zi1, wi1, xi2, yi2, zi2, wi2
                        vi34 = shufflevector(vi_3, vi_4, Val{mj})           # xi3, yi3, zi3, wi3, xi4, yi4, zi4, wi4
                        vi56 = shufflevector(vi_5, vi_6, Val{mj})           # xi5, yi5, zi5, wi5, xi6, yi6, zi6, wi6
                        vi78 = shufflevector(vi_7, vi_8, Val{mj})           # xi7, yi7, zi7, wi7, xi8, yi8, zi8, wi8
                    

                        vs1234_m1 = shufflevector(vi12, vi34, Val{m1})      # xi1, xi2, xi3, xi4, yi1, yi2, yi3, yi4
                        vs5678_m1 = shufflevector(vi56, vi78, Val{m1})      # xi5, xi6, xi7, xi8, yi5, yi6, yi7, yi8
                        vs1234_m2 = shufflevector(vi12, vi34, Val{m2})      # zi1, zi2, zi3, zi4, wi1, wi2, wi3, wi4
                        vs5678_m2 = shufflevector(vi56, vi78, Val{m2})      # zi5, zi6, zi7, zi8, wi5, wi6, wi7, wi8

                        xij = shufflevector(vs1234_m1, vs5678_m1, Val{mx})  # xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8
                        yij = shufflevector(vs1234_m1, vs5678_m1, Val{my})  # yi1, yi2, yi3, yi4, yi5, yi6, yi7, yi8
                        zij = shufflevector(vs1234_m2, vs5678_m2, Val{mx})  # zi1, zi2, zi3, zi4, zi5, zi6, zi7, zi8
                    end
                end)
    
                dij_sq = xij*xij + yij*yij + zij*zij
                mask = dij_sq <= cutsq
                !any(mask) && continue
    
                ϵm = ϵ[mlane, mask]
    
                lj2 = σ / dij_sq
                lj6 = lj2*lj2*lj2
                energy += sum( ϵm*(lj6*lj6 - lj6) )
                
                #region FORCE_SECTION
                $(if Forces === true
                    quote
                        fc = ϵm * T(24.0) * (lj6 - T(2.0) * lj6 * lj6) / dij_sq
                        @nexprs $N u ->  if mask[u]
                            f = fc[u] * vi_u
                            forces[flane+j_u] -= f
                            fi += f
                        end
                    end
                end)

                # if j+N <= natoms && j != i+1
                if j+N <= natoms
                    j += N # Do the next N atoms
                    # println("$i vs [")
                else
                    break
                end
                #endregion FORCE_SECTION
            end # while ptr
            

            # do remaining pairs
            for _j = (j+1):natoms
                $(if F == AoS
                    quote
                        j1 = _j << 2 - (2+_j)
                        vi1 = (vload(Vec{4,T}, coords, j1) - vi) * remaining_mask
                    end
                else
                    quote
                        vi1 = (vgather(coords, Vec((_j, _j+natoms, _j+2*natoms, _j))) - vi) * remaining_mask
                    end
                end)

                sdij_sq = sum(vi1*vi1)
                (sdij_sq > cutsq) && continue
    
                slj2 = σ / sdij_sq
                slj6 = slj2*slj2*slj2
                energy += slj6*slj6 - slj6
    
                $(if Forces === true
                    quote
                        sfc = T(24.0) * (slj6 - T(2.0) * slj6 * slj6) / sdij_sq
                        f = sfc * vi1
                        forces[flane + j1] -= f
                        fi += f
                    end
                end)
            end
    
            #region FORCE_SECTION
            $(if Forces === true
                :(forces[flane + i1] += fi)
            end)
            #endregion FORCE_SECTION
    
        end # for i
    
        return energy
    end # quote
end # generated function


@generated function simd(state::State{F, T}, vlist::VerletList, ::Type{Val{Forces}}, ::Type{Vec{N,T}}) where {Forces, N, T<:AbstractFloat, F <: AbstractMatrix}

    @assert isa(Forces, Bool) "Forces must be a boolean"
    # @assert T==Float32 "only Float32 is currently supported"
    @assert N ∈ (4,8) "only 4 or 8 wide Float32 vectors are allowed"

    quote
    
        σ = T(1)
        ϵ = ones(T, N)
        energy = T(0)
        natoms = state.n
    
        mlane = VecRange{N}(1)  # mask lane = vload(Vec{4, Float64}, X, 1)
        remaining_mask = Vec{4, T}((1, 1, 1, 0))

        coords = view(state.coords.values, :)
        cutsq = convert(T, vlist.cutoff*vlist.cutoff)

        #region FORCE_SECTION
        $(if Forces === true
            quote
                flane = VecRange{4}(0)
                forces = view(state.forces.values, :)
            end
        end)
        #endregion FORCE_SECTION
        
        $(if N==4
            quote
                m1 = (0, 4, 1, 5)
                m2 = (2, 6, 3, 7)
                mx = (0, 1, 4, 5)
                my = (2, 3, 6, 7)
            end
        else
            quote
                m1 = (0, 4,  8, 12,  1,  5,  9, 13)
                m2 = (2, 6, 10, 14,  3,  7, 11, 15)
                mx = (0, 1,  2,  3,  8,  9, 10, 11)
                my = (4, 5,  6,  7, 12, 13, 14, 15)
                mj = (0, 1,  2,  3,  4,  5,  6,  7)
            end
        end)

        @inbounds for i = 1:natoms-1
            
            # ptr -> location of the first neighbor of atom i
            # ptr_stop -> location of the last neighbor of atom i
            ptr = vlist.offset[i]
            if vlist.list[ptr] < 1
                continue
            end
            ptr_stop = vlist.offset[i+1]-2
            
            $(if Forces=== true
                :(fi = Vec{4,T}(0))
            end)
    
            # META PROGRAMMING QUESTION !

            # if typeof(state.coords) == AoSMatrix{Float32}
            $(if F == AoS
                quote
                    # Atom number to atom position conversion
                    i1 = (i<<1) + (i-2)
                    vi = vload(Vec{4,T}, coords, i1) # Load XYZ (consecutive)
                end
            else
                quote
                    # Get X, Y and Z positions in SoA format
                    js = Vec((i, i+natoms, i+2*natoms, i))
                    vi = vgather(coords, js) # Gather XYZ
                end
            end)

            # To remove
            # i1 = (i<<1) + (i-2)
            # vi = vload(Vec{4,T}, coords, i1) # Load XYZ (consecutive)

    
            while ptr+(N-1) <= ptr_stop
                j = ptr
                ptr += N
                
                # META PROGRAMMING QUESTION !

                $(if F == AoS
                    quote
                        @nexprs $N u ->  j_u = vlist.list[j+(u-1)]<<2 - (2 + vlist.list[j+(u-1)])
                        @nexprs $N u -> vi_u = vload(Vec{4,T}, coords, j_u) - vi    # xi1, yi1, zi1, wi1
                    end
                else
                    quote
                        @nexprs $N u ->  js_u = Vec((vlist.list[j+(u-1)], vlist.list[j+(u-1)]+natoms, vlist.list[j+(u-1)]+2*natoms, vlist.list[j+(u-1)]))
                        @nexprs $N u -> vi_u = vgather(coords, js_u) - vi    # xi1, yi1, zi1, wi1
                    end
                end)

                # To remove
                # @nexprs $N u ->  j_u = vlist.list[j+(u-1)]<<2 - (2 + vlist.list[j+(u-1)])
                # @nexprs $N u -> vi_u = vload(Vec{4,T}, coords, j_u) - vi    # xi1, yi1, zi1, wi1

                
                $(if N==4
                    quote
                        vs12_m1 = shufflevector(vi_1, vi_2, Val{m1})        # xi1, xi2, yi1, yi2
                        vs34_m1 = shufflevector(vi_3, vi_4, Val{m1})        # xi3, xi4, yi3, yi4
                        vs12_m2 = shufflevector(vi_1, vi_2, Val{m2})        # zi1, zi2, ?i1, ?i2
                        vs34_m2 = shufflevector(vi_3, vi_4, Val{m2})        # zi3, zi4, ?i3, ?i4
                        
                        xij = shufflevector(vs12_m1, vs34_m1, Val{mx})      # xi1, xi2, xi3, xi4
                        yij = shufflevector(vs12_m1, vs34_m1, Val{my})      # yi1, yi2, yi3, yi4
                        zij = shufflevector(vs12_m2, vs34_m2, Val{mx})      # zi1, zi2, zi3, zi4
                    end
                else
                    quote
                        vi12 = shufflevector(vi_1, vi_2, Val{mj})           # xi1, yi1, zi1, wi1, xi2, yi2, zi2, wi2
                        vi34 = shufflevector(vi_3, vi_4, Val{mj})           # xi3, yi3, zi3, wi3, xi4, yi4, zi4, wi4
                        vi56 = shufflevector(vi_5, vi_6, Val{mj})           # xi5, yi5, zi5, wi5, xi6, yi6, zi6, wi6
                        vi78 = shufflevector(vi_7, vi_8, Val{mj})           # xi7, yi7, zi7, wi7, xi8, yi8, zi8, wi8
                    

                        vs1234_m1 = shufflevector(vi12, vi34, Val{m1})      # xi1, xi2, xi3, xi4, yi1, yi2, yi3, yi4
                        vs5678_m1 = shufflevector(vi56, vi78, Val{m1})      # xi5, xi6, xi7, xi8, yi5, yi6, yi7, yi8
                        vs1234_m2 = shufflevector(vi12, vi34, Val{m2})      # zi1, zi2, zi3, zi4, wi1, wi2, wi3, wi4
                        vs5678_m2 = shufflevector(vi56, vi78, Val{m2})      # zi5, zi6, zi7, zi8, wi5, wi6, wi7, wi8

                        xij = shufflevector(vs1234_m1, vs5678_m1, Val{mx})  # xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8
                        yij = shufflevector(vs1234_m1, vs5678_m1, Val{my})  # yi1, yi2, yi3, yi4, yi5, yi6, yi7, yi8
                        zij = shufflevector(vs1234_m2, vs5678_m2, Val{mx})  # zi1, zi2, zi3, zi4, zi5, zi6, zi7, zi8
                    end
                end)
    
                dij_sq = xij*xij + yij*yij + zij*zij
                mask = dij_sq <= cutsq
                !any(mask) && continue
    
                ϵm = ϵ[mlane, mask]
    
                lj2 = σ / dij_sq
                lj6 = lj2*lj2*lj2
                energy += sum( ϵm*(lj6*lj6 - lj6) )
                
                #region FORCE_SECTION
                $(if Forces === true
                    quote
                        fc = ϵm * T(24.0) * (lj6 - T(2.0) * lj6 * lj6) / dij_sq
                        @nexprs $N u ->  if mask[u]
                            f = fc[u] * vi_u
                            forces[flane+j_u] -= f
                            fi += f
                        end
                    end
                end)
                #endregion FORCE_SECTION
    
            end # while ptr
            

            # do remaining pairs
            for j = ptr:ptr_stop

                # META PROGRAMMING QUESTION !

                $(if F == AoS
                    quote
                        j1 = vlist.list[j]<<2 - (2 + vlist.list[j])
                        vi1 = (vload(Vec{4,T}, coords, j1) - vi) * remaining_mask
                    end
                else
                    quote
                        j1 = vlist.list[j]
                        vi1 = (vgather(coords, Vec((j1, j1+natoms, j1+2*natoms, j1))) - vi) * remaining_mask
                    end
                end)

                # To remove
                # j1 = vlist.list[j]<<2 - (2 + vlist.list[j])
                # vi1 = (vload(Vec{4,T}, coords, j1) - vi) * remaining_mask

                sdij_sq = sum(vi1*vi1)
                (sdij_sq > cutsq) && continue
    
                slj2 = σ / sdij_sq
                slj6 = slj2*slj2*slj2
                energy += slj6*slj6 - slj6
    
                $(if Forces === true
                    quote
                        sfc = T(24.0) * (slj6 - T(2.0) * slj6 * slj6) / sdij_sq
                        f = sfc * vi1
                        forces[flane + j1] -= f
                        fi += f
                    end
                end)
            end
    
            #region FORCE_SECTION
            $(if Forces === true
                :(forces[flane + i1] += fi)
            end)
            #endregion FORCE_SECTION
    
        end # for i
    
        return energy
    end # quote
end # generated function



@generated function simd_aos_sergio(state::State{F, T}, vlist::VerletList, ::Type{Val{Forces}}, ::Type{Vec{N,T}}) where {Forces, N, T<:AbstractFloat, F <: AbstractMatrix}

    @assert isa(Forces, Bool) "Forces must be a boolean"
    @assert T==Float32 "only Float32 is currently supported"
    @assert N ∈ (4,8) "only 4 or 8 wide Float32 vectors are allowed"

    quote
    
        σ = T(1)
        ϵ = ones(T, N)
        energy = T(0)
        natoms = state.n
    
        mlane = VecRange{N}(1)  # mask lane = vload(Vec{4, Float64}, X, 1)
        coords = view(state.coords.values, :) # Creates x1,y1,z1,x2,y2,z2,x3,y3,z3,...
        cutsq = convert(T, vlist.cutoff*vlist.cutoff)
        
        #region FORCE_SECTION
        $(if Forces === true
            quote
                flane = VecRange{4}(0)
                forces = view(state.forces, :)
            end
        end)
        #endregion FORCE_SECTION
        
        $(if N==4
            quote
                m1 = (0, 4, 1, 5)
                m2 = (2, 6, 3, 7)
                mx = (0, 1, 4, 5)
                my = (2, 3, 6, 7)
            end
        else
            quote
                m1 = (0, 4,  8, 12,  1,  5,  9, 13)
                m2 = (2, 6, 10, 14,  3,  7, 11, 15)
                mx = (0, 1,  2,  3,  8,  9, 10, 11)
                my = (4, 5,  6,  7, 12, 13, 14, 15)
                mj = (0, 1,  2,  3,  4,  5,  6,  7)
            end
        end)

        @inbounds for i = 1:natoms-1
            
            # ptr -> location of the first neighbor of atom i
            # ptr_stop -> location of the last neighbor of atom i
            ptr = vlist.offset[i]
            if vlist.list[ptr] < 1
                continue
            end
            ptr_stop = vlist.offset[i+1]-2
            
            $(if Forces=== true
                :(fi = Vec{4,T}(0))
            end)
    
            i1 = i<<2 - 3 # Allows me to travel every 4 numbers
            vi = vload(Vec{4,T}, coords, i1)
    
            while ptr+(N-1) <= ptr_stop
                j = ptr
                ptr += N
                
                # j = 4*(j-1) + 1 = 4j-4+1 = 4j-3 = j<<2 - 3
                @nexprs $N u ->  j_u = vlist.list[j+(u-1)]<<2 - 3
                @nexprs $N u -> vi_u = vload(Vec{4,T}, coords, j_u) - vi    # xi1, yi1, zi1, wi1
                
                $(if N==4
                    quote
                        vs12_m1 = shufflevector(vi_1, vi_2, Val{m1})        # xi1, xi2, yi1, yi2
                        vs34_m1 = shufflevector(vi_3, vi_4, Val{m1})        # xi3, xi4, yi3, yi4
                        vs12_m2 = shufflevector(vi_1, vi_2, Val{m2})        # zi1, zi2, ?i1, ?i2
                        vs34_m2 = shufflevector(vi_3, vi_4, Val{m2})        # zi3, zi4, ?i3, ?i4
                        
                        xij = shufflevector(vs12_m1, vs34_m1, Val{mx})      # xi1, xi2, xi3, xi4
                        yij = shufflevector(vs12_m1, vs34_m1, Val{my})      # yi1, yi2, yi3, yi4
                        zij = shufflevector(vs12_m2, vs34_m2, Val{mx})      # zi1, zi2, zi3, zi4
                    end
                else
                    quote
                        vi12 = shufflevector(vi_1, vi_2, Val{mj})           # xi1, yi1, zi1, wi1, xi2, yi2, zi2, wi2
                        vi34 = shufflevector(vi_3, vi_4, Val{mj})           # xi3, yi3, zi3, wi3, xi4, yi4, zi4, wi4
                        vi56 = shufflevector(vi_5, vi_6, Val{mj})           # xi5, yi5, zi5, wi5, xi6, yi6, zi6, wi6
                        vi78 = shufflevector(vi_7, vi_8, Val{mj})           # xi7, yi7, zi7, wi7, xi8, yi8, zi8, wi8

                        vs1234_m1 = shufflevector(vi12, vi34, Val{m1})      # xi1, xi2, xi3, xi4, yi1, yi2, yi3, yi4
                        vs5678_m1 = shufflevector(vi56, vi78, Val{m1})      # xi5, xi6, xi7, xi8, yi5, yi6, yi7, yi8
                        vs1234_m2 = shufflevector(vi12, vi34, Val{m1})      # zi1, zi2, zi3, zi4, wi1, wi2, wi3, wi4
                        vs5678_m2 = shufflevector(vi56, vi78, Val{m1})      # zi5, zi6, zi7, zi8, wi5, wi6, wi7, wi8

                        xij = shufflevector(vs1234_m1, vs5678_m1, Val{mx})  # xi1, xi2, xi3, xi4, xi5, xi6, xi7, xi8
                        yij = shufflevector(vs1234_m1, vs5678_m1, Val{my})  # yi1, yi2, yi3, yi4, yi5, yi6, yi7, yi8
                        zij = shufflevector(vs1234_m2, vs5678_m2, Val{mx})  # zi1, zi2, zi3, zi4, zi5, zi6, zi7, zi8
                    end
                end)
    
                dij_sq = xij*xij + yij*yij + zij*zij
                mask = dij_sq <= cutsq
                !any(mask) && continue
    
                ϵm = ϵ[mlane, mask]
    
                lj2 = σ / dij_sq
                lj6 = lj2*lj2*lj2
                energy += sum( ϵm*(lj6*lj6 - lj6) )
                
                #region FORCE_SECTION
                $(if Forces === true
                    quote
                        fc = ϵm * T(24.0) * (lj6 - T(2.0) * lj6 * lj6) / dij_sq
                        @nexprs $N u ->  if mask[u]
                            f = fc[u] * vi_u
                            forces[flane+j_u] -= f
                            fi += f
                        end
                    end
                end)
                #endregion FORCE_SECTION
    
            end # while ptr
            

            # do remaining pairs
            for j = ptr:ptr_stop
                # println("$i  $(vlist.list[j])")
                j1 = vlist.list[j]<<2 - 3
                vi1 = vload(Vec{4,T}, coords, j1) - vi
                # Take 4th element out ???

                sdij_sq = sum(vi1*vi1)
                (sdij_sq > cutsq) && continue
    
                slj2 = σ / sdij_sq
                slj6 = slj2*slj2*slj2
                energy += slj6*slj6 - slj6
    
                $(if Forces === true
                    quote
                        sfc = T(24.0) * (slj6 - T(2.0) * slj6 * slj6) / sdij_sq
                        f = sfc * vi1
                        forces[flane + j1] -= f
                        fi += f
                    end
                end)
            end
            # println("")
    
            #region FORCE_SECTION
            $(if Forces === true
                :(forces[flane + i1] += fi)
            end)
            #endregion FORCE_SECTION
    
        end # for i
        return energy
    
    end # quote
end # generated function
    
    
