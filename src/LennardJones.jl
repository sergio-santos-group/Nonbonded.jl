module LennardJones

export generate_cubic_lattice, generate_lattice, LATTICE, State, VerletList,
       update_serial!, serial_no_verlet_AoS

const FORCES = Val{true}
const NOFORCES = Val{false}

include("models.jl")      # Allows particle model generation and printing
include("verlet_list.jl") # Allows the usage of Verlet lists
include("serial.jl")      # Calculation of energy and forces in serial approach

end # module
