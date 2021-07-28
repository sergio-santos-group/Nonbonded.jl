module LennardJones

export generate_cubic_lattice, generate_lattice, LATTICE, State, VerletList,
       update_serial!, serial, simd, naive, cuda

const FORCES = Val{true}
const NOFORCES = Val{false}

include("state.jl")       # Allows particle models to be saved as States
include("models.jl")      # Allows particle model generation and printing
include("verlet_list.jl") # Allows the usage of Verlet lists
include("serial.jl")      # Calculation of energy and forces with serial methods
include("simd.jl")        # Calculation of energy and forces with SIMD methods
include("cuda.jl")        # Calculation of energy and forces with CUDA methods      

end # module