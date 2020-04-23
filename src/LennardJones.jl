module LennardJones

export generate_cubic_lattice, generate_lattice, LATTICE

greet() = print("Hello World!")

include("models.jl") # Allows particle model generation and printing

end # module
