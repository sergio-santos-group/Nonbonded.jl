module LennardJones

export generate_cubic_lattice, generate_lattice, LATTICE, State, VerletList,
       update_serial!, serial_no_verlet, serial_no_verlet,
       simd4_no_verlet_AoS, simd4_verlet_AoS, simd8_no_verlet_AoS,
       simd8_verlet_AoS

const FORCES = Val{true}
const NOFORCES = Val{false}

include("state.jl")       # Allows particle models to be saved as States
include("models.jl")      # Allows particle model generation and printing
include("verlet_list.jl") # Allows the usage of Verlet lists
include("serial.jl")      # Calculation of energy and forces in serial approach
include("simd4.jl")       # Calculation of energy and forces in SIMD 4 approach
include("simd8.jl")       # Calculation of energy and forces in SIMD 8 approach

end # module