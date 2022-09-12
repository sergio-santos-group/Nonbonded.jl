using Distributed
using Dates

start = now()

printstyled("> Starting serial simulation.\n"; color = :yellow)

push!(LOAD_PATH, "src")
using LennardJones
using SIMD

printstyled("> Setting up simulation environment.\n"; color = :yellow)
    
N = 7
T            = Float64
xyz_aos      = generate_cubic_lattice(100.0, N, LATTICE.face_centered) # This is in AoS
xyz_aos      = convert(Matrix{T}, xyz_aos)
n_atoms      = max(size(xyz_aos)...)
println("N atoms: $n_atoms")
state_aos    = State(xyz_aos)
vlist        = VerletList(n_atoms)
vlist.cutoff = T(10.0)
vlist        = update_serial!(vlist, state_aos)
N_steps      = 1_000
N            = 45 # Number of replicas

for j in 1:N
    println(" Consuming job $j on single worker")
    for i in 1:N_steps
        simd(state_aos, vlist, LennardJones.NOFORCES, Vec{8, T})
        # cuda(state_aos, T(10.0))
    end
end

printstyled("> Exiting serial simulation.\n"; color = :yellow)

# Measure elapsed time
_elapsed = now() - start
elapsed  = canonicalize(Dates.CompoundPeriod(_elapsed))
printstyled("> Elapsed time: $elapsed\n"; color = :red)
elapsed  = Dates.Second(div(_elapsed, 1000).value)
printstyled("> Elapsed time: $elapsed\n"; color = :red)
open("time.dat", "w") do io
    write(io, "$elapsed")
end