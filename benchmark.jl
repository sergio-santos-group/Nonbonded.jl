push!(LOAD_PATH, "src")
using LennardJones
using SIMD
using DataFrames

function calc_max_atoms_in_vlist(natoms::Int64, vlist::VerletList)

    counts = Vector{Int64}()
    for i in 1:natoms
        counter = 0
        ptr = vlist.offset[i]
        while vlist.list[ptr] > 0
            counter += 1
            ptr += 1
        end
        push!(counts, counter)
    end
    return maximum(counts)
end


function trial(expr::Expr, title::String)
    println("\n$title ($n_atoms atoms)")
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
    return t
end

println("Starting Benchmark")
T         = Float64
n_samples = 1
cut_off   = T(10.0) # angstrom

# Generate states
xyz_aos   = generate_cubic_lattice(100.0, parse(Int, ARGS[1]), LATTICE.primitive) # This is in AoS

xyz_aos   = convert(Matrix{T}, xyz_aos)
xyz_soa   = collect(xyz_aos')
n_atoms   = max(size(xyz_aos)...)

println(" Nº atoms: $n_atoms")
state_aos = State(xyz_aos)
state_soa = State(xyz_soa)

# Verlet list
vlist = VerletList(n_atoms)
vlist.cutoff = cut_off
println("\nUpdating vlist in serial mode")
vlist = update_serial!(vlist, state_aos.coords.values)
m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)
println(" Max atoms in Verlet list: $m_atoms")

# Calculate the correct, known energies and forces
println("\nCalculating correct energy")
energy = serial(state_aos, cut_off, LennardJones.NOFORCES)
print(" Energy: $energy\n")

results = DataFrame(
    N_ATOMS   = Int[],
    M_ATOMS   = Int[],
    S4_NV_AOS = Float64[],
    S4_NV_SOA = Float64[],
    S8_NV_AOS = Float64[],
    S8_NV_SOA = Float64[],
    S4_V_AOS  = Float64[],
    S4_V_SOA  = Float64[],
    S8_V_AOS  = Float64[],
    S8_V_SOA  = Float64[],
    CUDA      = Float64[],)

println("Starting trials ...")

# trial(:(serial(state_aos, cut_off, LennardJones.NOFORCES)), "Serial - No Verlet - AoS")
# trial(:(serial(state_soa, cut_off, LennardJones.NOFORCES)), "Serial - No Verlet - SoA")
# trial(:(serial(state_aos, vlist, LennardJones.NOFORCES)), "Serial - Verlet - AoS")
# trial(:(serial(state_soa, vlist, LennardJones.NOFORCES)), "Serial - Verlet - SoA")
t1 = trial(:(simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - No Verlet - AoS")
t2 = trial(:(simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - No Verlet - SoA")
t3 = trial(:(simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - No Verlet - AoS")
t4 = trial(:(simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - No Verlet - SoA")
t5 = trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - Verlet - AoS")
t6 = trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - Verlet - SoA")
t7 = trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - Verlet - AoS")
t8 = trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - Verlet - SoA")
t9 = trial(:(cuda(state_soa)), "CUDA - SoA")

push!(results, (n_atoms, m_atoms, t1, t2, t3, t4, t5, t6, t7, t8, t9))

display(results)