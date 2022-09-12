using Distributed
using Dates

start = now()

printstyled("> Starting distributed simulation.\n"; color = :yellow)

# `machines` is a vector of machine specifications;
# Workers are started for each specification;
# A machine specification is either a string `machine_spec`
# or a tuple (machine_spec, count);
# `machine_spec` is a string of the form [user@]host[:port] [bind_addr[:port]]

machines = Vector{Tuple{String, Int}}([ # 45/68 workers
    ("jpereira@192.168.179.59", 3),     # Feynman (Max: 8 )
    ("jpereira@192.168.179.86", 7),     # Buteo   (Max: 16) # For some reason Buteo allocates GPU memory...
    # ("jpereira@192.168.179.34", 40),    # Warbler (Max: 40)
])

printstyled("> Spawning workers.\n"; color = :yellow)
addprocs(machines,
    exename="julia",
    dir = "/home/jpereira",
    max_parallel=68)
 
printstyled("> Adding current path to workers.\n"; color = :yellow)
@everywhere mkpath("/home/jpereira/playground/Nonbonded.jl")
@everywhere cd("/home/jpereira/playground/Nonbonded.jl")

# printstyled("> Transfering Nonbonded.jl ...\n"; color = :yellow)
# for (machine, nprocs) in machines[2:end]
#     printstyled("  > Transfering Nonbonded.jl to $machine:$(pwd())\n"; color = :yellow)
#     run(`scp -r /home/jpereira/playground/Nonbonded.jl $machine:/home/jpereira/playground/`)
# end

printstyled("> Loading Nonbonded.jl ...\n"; color = :yellow)

@everywhere push!(LOAD_PATH, "src")
@everywhere using LennardJones
@everywhere using SIMD

printstyled("> Setting up simulation environment.\n"; color = :yellow)
@everywhere begin  
    
    T            = Float64
    xyz_aos      = generate_cubic_lattice(100.0, 2, LATTICE.face_centered) # This is in AoS
    xyz_aos      = convert(Matrix{T}, xyz_aos)
    n_atoms      = max(size(xyz_aos)...)
    println("N atoms: $n_atoms")
    state_aos    = State(xyz_aos)
    vlist        = VerletList(n_atoms)
    vlist.cutoff = T(10.0)
    vlist        = update_serial!(vlist, state_aos.coords.values)
    N_steps      = 100

    function start_simulation(job_cards::RemoteChannel, results::RemoteChannel)
        println("Starting simulation on worker $(myid()) - $(gethostname()) ...")
        while true
            job_n = take!(job_cards)
            println(" Consuming job $job_n on worker $(myid()) - $(gethostname())")
            for i in 1:N_steps
                simd(state_aos, vlist, LennardJones.NOFORCES, Vec{8, T})
                # cuda(state_aos, T(10.0))
            end
            put!(results, "OK")
        end
    end
end

printstyled("> Populating job queue.\n"; color = :yellow)

N = 450 # Number of replicas

job_queue = RemoteChannel(() -> Channel{Int16}(N))
for job_n in 1:N
    put!(job_queue, Int16(job_n))
end
results_queue = RemoteChannel(() -> Channel{Any}(N))

printstyled("> Starting replica simulations on all workers ...\n"; color = :yellow)
for p in workers()
    remote_do(start_simulation, p, job_queue, results_queue)
end

# * Save results
function retrieve_results(queue::RemoteChannel, n::Int)
    finished_jobs = 0
    while finished_jobs < n
        take!(queue)
        finished_jobs += 1
        printstyled(" > Finished jobs: $finished_jobs/$n \n"; color = :yellow)
    end
end

printstyled("> Simulations running. Waiting for results ...\n"; color = :yellow)
retrieve_results(results_queue, N)

printstyled("> Exiting distributed simulation.\n"; color = :yellow)

# Measure elapsed time
_elapsed = now() - start
elapsed  = canonicalize(Dates.CompoundPeriod(_elapsed))
printstyled("> Elapsed time: $elapsed\n"; color = :red)
elapsed  = Dates.Second(div(_elapsed, 1000).value)
printstyled("> Elapsed time: $elapsed\n"; color = :red)
open("time.dat", "w") do io
    write(io, "$elapsed")
end