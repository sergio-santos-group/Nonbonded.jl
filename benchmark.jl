push!(LOAD_PATH, "src")
using LennardJones
using SIMD

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
    @time begin 
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
end

println("Starting Benchmark")
T = Float64
n_samples = 1
cut_off   = T(99999999.9)

# Generate states
xyz_aos   = generate_cubic_lattice(100.0, 23, LATTICE.primitive) # This is in AoS
# xyz_soa   = [1 2 3; 4 5 6; 7 8 9; 10 11 12; 13 14 15; 16 17 18; 19 20 21; 22 23 24; 25 26 27; 28 29 30]

xyz_aos   = convert(Matrix{T}, xyz_aos)
xyz_soa   = collect(xyz_aos')
n_atoms   = max(size(xyz_aos)...)

println(" Nº atoms: $n_atoms")
state_aos = State(xyz_aos)
state_soa = State(xyz_soa)

# energy3 = naive(state_soa, T(Inf), false)
# energy2 = cuda2(state_soa)
# print("$energy3 = $energy2 ($(energy3 == energy2))")
# exit(1)

# LennardJones.print_xyz(state_aos.coords.values, "test.xyz")

# Verlet list
vlist = VerletList(n_atoms)
vlist.cutoff = cut_off
println("\nUpdating vlist in serial mode")
vlist = update_serial!(vlist, state_aos.coords.values)
println(" Max atoms in Verlet list: $(calc_max_atoms_in_vlist(n_atoms, vlist))")

# Calculate the correct, known energies and forces
println("\nCalculating correct energy")
energy = serial(state_aos, cut_off, LennardJones.NOFORCES)
print(" Energy: $energy\n")

# trial(:(serial(state_aos, cut_off, LennardJones.NOFORCES)), "Serial - No Verlet - AoS")
# trial(:(serial(state_soa, cut_off, LennardJones.NOFORCES)), "Serial - No Verlet - SoA")
# trial(:(serial(state_aos, vlist, LennardJones.NOFORCES)), "Serial - Verlet - AoS")
# trial(:(serial(state_soa, vlist, LennardJones.NOFORCES)), "Serial - Verlet - SoA")
# trial(:(simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - No Verlet - AoS")
# trial(:(simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - No Verlet - SoA")
# trial(:(simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - No Verlet - AoS")
# trial(:(simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - No Verlet - SoA")
# trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - Verlet - AoS")
trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{4, T})), "SIMD_4 - Verlet - SoA")
# trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - Verlet - AoS")
trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{8, T})), "SIMD_8 - Verlet - SoA")
trial(:(cuda(state_soa)), "CUDA - SoA")