using CUDAdrv, CUDAnative, CuArrays

# https://stackoverflow.com/questions/5611905/n-body-cuda-optimization
# https://devblogs.nvidia.com/even-easier-introduction-cuda/

function kernel(
    xyz::CuDeviceVector{T},
    forces::CuDeviceVector{T},
    energy::CuDeviceVector{T},
    N::Int,
    cutsq::T) where {T<:AbstractFloat}
    
    blocksize = blockDim().x
    @cuprint(" Block size: $blocksize\n")

    # thread ID
    tid = (blockIdx().x-1) * blocksize + threadIdx().x
    @cuprint(" Thread ID: $tid\n")
    
    if tid <= N
        id = threadIdx().x

        shmem = @cuDynamicSharedMem(T, 3*blocksize) # ?
        
        ii = tid<<2 - (2 + tid) # AoS
        @cuprint("$ii / $(length(xyz)) @ thread $(threadIdx().x)\n")
        
        e  = T(0)
        fx = T(0)
        fy = T(0)
        fz = T(0)
        
        xi = xyz[ii]
        yi = xyz[ii+1]
        zi = xyz[ii+2]

        for i=1:blocksize:N
            k = ((i+id)<<1) + (i+id-2) # AoS
            @cuprint("$ii <-> $k / $(length(xyz)) @ thread $(threadIdx().x)\n")
            if (i+id) <= N
                shmem[(id-1)*3+1] = xyz[k]
                shmem[(id-1)*3+2] = xyz[k+1]
                shmem[(id-1)*3+3] = xyz[k+2]

                sync_threads()

                #if i!=tid
                for j=1:blocksize
                    dx = shmem[(j-1)*3+1] - xi
                    dy = shmem[(j-1)*3+2] - yi
                    dz = shmem[(j-1)*3+3] - zi
                    dij_sq = dx*dx + dy*dy + dz*dz
                    
                    if dij_sq <= cutsq && dij_sq > 1e-12
                        lj2 = T(1) / dij_sq
                        lj6 = lj2*lj2*lj2
                        e += lj6*lj6 - lj6

                        fc = T(24.0) * (lj6 - T(2.0) * lj6 * lj6) / dij_sq
                        fx += fc*dx
                        fy += fc*dy
                        fz += fc*dz
                    end

                end
                #end
                sync_threads()
            end
        end

        energy[tid]  = e/2
        forces[ii  ] = fx
        forces[ii+1] = fy
        forces[ii+2] = fz
    end

    return nothing
end


function cuda(state::State{F, T}, cutoff::T) where {F <: AbstractMatrix, T <: AbstractFloat}
    
    N = state.n
    x_gpu = CuArray(view(state.coords.values, :))
    f_gpu = similar(x_gpu)
    e_gpu = CuArray{T}(undef, N)
    
    total_threads = min(N, 512)
    threads       = (total_threads, )
    blocks        = ceil.(Int, N ./ threads)

    # calculate size of dynamic shared memory
    shmem = 3 * sum(threads) * sizeof(T)
    cutsq = convert(T, cutoff*cutoff)

    @cuda blocks=blocks threads=threads shmem=shmem kernel(x_gpu, f_gpu, e_gpu, N, cutsq)

    e = sum(e_gpu)
    return e
end