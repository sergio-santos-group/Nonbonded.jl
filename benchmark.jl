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
    println("  Energy: $energy_result (Δe = $diff)")
end

println("Starting Benchmark")
n_samples = 1
cut_off   = Float32(5.0)

# Generate states
xyz_aos   = generate_cubic_lattice(100.0, 31, LATTICE.primitive) # This is in AoS

xyz_aos   = convert(Matrix{Float32}, xyz_aos)
xyz_soa   = collect(xyz_aos')
n_atoms   = max(size(xyz_aos)...)

println("Nº atoms: $n_atoms")
state_aos = State(n_atoms, xyz_aos, zeros(Float32, 3, n_atoms))
state_soa = State(n_atoms, xyz_soa, zeros(Float32, n_atoms, 3))

# LennardJones.print_xyz(state_aos.coords.values, "test.xyz")

# Verlet list
vlist = VerletList(n_atoms)
vlist.cutoff = cut_off
println("Updating vlist in serial mode")
vlist = update_serial!(vlist, state_aos.coords.values)
println("Max atoms in Verlet list: $(calc_max_atoms_in_vlist(n_atoms, vlist))")

# Calculate the correct, known energies and forces
println("Calculating correct energy")
energy = simd(state_aos, vlist, LennardJones.NOFORCES, Vec{8, Float32})
print("Energy: $energy\n")

# trial(:(serial(state_aos, cut_off, LennardJones.NOFORCES)), "Serial - No Verlet - AoS")
# trial(:(serial(state_soa, cut_off, LennardJones.NOFORCES)), "Serial - No Verlet - SoA")
trial(:(serial(state_aos, vlist, LennardJones.NOFORCES)), "Serial - Verlet - AoS")
trial(:(serial(state_soa, vlist, LennardJones.NOFORCES)), "Serial - Verlet - SoA")
# trial(:(simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{4, Float32})), "SIMD_4 - No Verlet - AoS")
# trial(:(simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{4, Float32})), "SIMD_4 - No Verlet - SoA")
# trial(:(simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{8, Float32})), "SIMD_8 - No Verlet - AoS")
# trial(:(simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{8, Float32})), "SIMD_8 - No Verlet - SoA")
trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{4, Float32})), "SIMD_4 - Verlet - AoS")
trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{4, Float32})), "SIMD_4 - Verlet - SoA")
trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{8, Float32})), "SIMD_8 - Verlet - AoS")
trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{8, Float32})), "SIMD_8 - Verlet - SoA")
# trial(:(simd_aos_sergio(state_aos, vlist, LennardJones.NOFORCES, Vec{8, Float32})), "SIMD_8 - Verlet - AoS (Sérgio)")