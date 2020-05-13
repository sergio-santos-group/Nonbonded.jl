push!(LOAD_PATH, "src")
using LennardJones
using SIMD

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

# Generate states
xyz_aos   = generate_cubic_lattice(100.0, 15, LATTICE.primitive) # This is in AoS

xyz_aos   = convert(Matrix{Float32}, xyz_aos)
xyz_soa   = collect(xyz_aos')
n_atoms   = max(size(xyz_aos)...)

state_aos = State(n_atoms, xyz_aos, zeros(Float32, 3, n_atoms))
state_soa = State(n_atoms, xyz_soa, zeros(Float32, n_atoms, 3))

# LennardJones.print_xyz(state_aos.coords.values, "test.xyz")

# Verlet list
vlist = VerletList(n_atoms)
vlist.cutoff = Float32(Inf)
vlist = update_serial!(vlist, state_aos.coords.values)

# Calculate the correct, known energies and forces
energy = LennardJones.naive(state_aos, Float32(Inf), false)

trial(:(serial(state_aos, Float32(Inf), LennardJones.NOFORCES)), "Serial - No Verlet - AoS")
trial(:(serial(state_soa, Float32(Inf), LennardJones.NOFORCES)), "Serial - No Verlet - SoA")
trial(:(serial(state_aos, vlist, LennardJones.NOFORCES)), "Serial - Verlet - AoS")
trial(:(serial(state_soa, vlist, LennardJones.NOFORCES)), "Serial - Verlet - SoA")
trial(:(simd(state_aos, Float32(Inf), LennardJones.NOFORCES, Vec{4, Float32})), "SIMD_4 - No Verlet - AoS")
trial(:(simd(state_soa, Float32(Inf), LennardJones.NOFORCES, Vec{4, Float32})), "SIMD_4 - No Verlet - SoA")
trial(:(simd(state_aos, Float32(Inf), LennardJones.NOFORCES, Vec{8, Float32})), "SIMD_8 - No Verlet - AoS")
trial(:(simd(state_soa, Float32(Inf), LennardJones.NOFORCES, Vec{8, Float32})), "SIMD_8 - No Verlet - SoA")
trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{4, Float32})), "SIMD_4 - Verlet - AoS")
trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{4, Float32})), "SIMD_4 - Verlet - SoA")
trial(:(simd(state_aos, vlist, LennardJones.NOFORCES, Vec{8, Float32})), "SIMD_8 - Verlet - AoS")
trial(:(simd(state_soa, vlist, LennardJones.NOFORCES, Vec{8, Float32})), "SIMD_8 - Verlet - SoA")