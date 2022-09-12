push!(LOAD_PATH, "src")
using LennardJones
using SIMD
using Statistics

function calc_max_atoms_in_vlist(natoms::Int64, vlist::VerletList)

    counts = Vector{Int64}()
    for i in 1:natoms
        counter = 0
        ptr = vlist.offset[i]
        if ptr > 0
            while vlist.list[ptr] > 0
                counter += 1
                ptr += 1
            end
        end
        push!(counts, counter)
    end
    return maximum(counts)
end


function trial(expr::Expr, title::String)
    # println("\n$title ($n_atoms atoms ▶ $m_atoms interacting atoms)")
    eval(expr)
    t = @elapsed begin 
        for i in 1:n_samples
            eval(expr)
        end
    end
    energy_result = eval(expr)
    diff = energy - energy_result
    if abs(diff) < 1e-10
        diff = "0.0 (or less than 1e10)"
    end
    println("  Energy: $energy_result (Δe = $diff)")
    println("  Elapsed: $t")
    # return t
end

println("Starting Benchmark")
T         = Float64
n_samples = 1
N         = 7

# Generate states
# xyz_aos   = generate_cubic_lattice(100.0, parse(Int, ARGS[1]), LATTICE.primitive) # This is in AoS
xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered) # This is in AoS

xyz_aos   = convert(Matrix{T}, xyz_aos)
xyz_soa   = collect(xyz_aos')
n_atoms   = max(size(xyz_aos)...)

println(" Nº atoms: $n_atoms")
state_aos = State(xyz_aos)
state_soa = State(xyz_soa)

# Verlet list
cut_off   = T(21.0) # angstrom
vlist = VerletList(n_atoms)
vlist.cutoff = cut_off
println("\nUpdating vlist in serial mode")
vlist = update_serial!(vlist, state_aos)
m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)
println(" Max atoms in Verlet list: $m_atoms")

# Calculate the correct, known energies and forces
println("\nCalculating correct energy")
energy = serial(state_aos, T(Inf), LennardJones.NOFORCES)
print(" Energy: $energy\n")

println("Starting trials ...")

# trial(:(LennardJones.non_optimized(state_aos.coords.values, zeros(size(state_aos.coords.values)))), "Non-optimized - AoS")
# trial(:(serial(state_aos, cut_off, LennardJones.NOFORCES)), "Serial - No Verlet - AoS")
# trial(:(serial(state_soa, cut_off, LennardJones.NOFORCES)), "Serial - No Verlet - SoA")
# trial(:(serial(state_aos, vlist, LennardJones.NOFORCES)), "Serial - Verlet - AoS")
# trial(:(serial(state_soa, vlist, LennardJones.NOFORCES)), "Serial - Verlet - SoA")
t1 = trial(:(simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - No Verlet - AoS")
# t2 = trial(:(simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - No Verlet - SoA")
# t3 = trial(:(simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - No Verlet - AoS")
# t4 = trial(:(simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - No Verlet - SoA")
# t5 = trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - Verlet - AoS")
# t6 = trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - Verlet - SoA")
# t7 = trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - Verlet - AoS")
# t8 = trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - Verlet - SoA")
# t9 = trial(:(cuda(state_aos, cut_off)), "CUDA - SoA")