push!(LOAD_PATH, "src")
using LennardJones

function trial(expr::Expr, title::String)
    println("\n$title ($n_atoms atoms)")
    eval(expr)
    @time begin 
        for i in 1:n_samples
            eval(expr)
        end
    end
    energy_result = eval(expr)
    result = energy - energy_result < 0.001
    println("  Energy: $energy_result ($result)")
end

println("Starting Benchmark")
n_samples = 10

xyz_aos = generate_cubic_lattice(100.0, 10, LATTICE.primitive)
xyz_aos = convert(Matrix{Float32}, xyz_aos)
n_atoms = max(size(xyz_aos)...)
state_aos = State(n_atoms, xyz_aos, zeros(Float32, 3, n_atoms))

energy = LennardJones.naive_aos(state_aos, Float32(Inf), false)

trial(:(LennardJones.naive_aos(state_aos, Float32(Inf), false)), "Serial - No Verlet - AoS (NAIVE)")
trial(:(serial_no_verlet_AoS(state_aos, Float32(Inf), LennardJones.NOFORCES)), "Serial - No Verlet - AoS")