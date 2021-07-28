using Printf

"""
    LATTICE

    Defines a lattice type. Supported lattice types include 'primitive',
    'body_centered' and 'face_centered' lattices.
"""
module LATTICE
    @enum TYPE begin
        primitive       = 1
        body_centered   = 2
        face_centered   = 3
    end
end


"""
    generate_template(cell_sizes::Vector{Float64}, type::LATTICE.TYPE)

        Generate a template for the given lattice 'type', with the cell size
        defined in 'cell_sizes' (should be a 3x1 Vector{Int64, 1}, with cell
        sizes in each of the 3 dimensions, in nm). Supported lattice types
        include 'primitive', 'body_centered' and 'face_centered' lattices.
        Returns an Array{Float64, 2}.

    Example:

        generate_template([1.0, 1.0, 1.0], LATTICE.primitive)
"""
function generate_template(cell_sizes::Vector{Float64}, type::LATTICE.TYPE)::Array{Float64, 2}
    hf = cell_sizes ./ 2
    atoms = [0.0 0.0 0.0]
    if type == LATTICE.body_centered
        atoms = vcat(atoms, [hf[1] hf[2] hf[3]])
    elseif type == LATTICE.face_centered
        atoms = vcat(atoms, [hf[1] hf[2] 0.0; hf[1] 0.0 hf[3]; 0.0 hf[2] hf[3]])
    end
    return atoms
end


"""
    calculate_n_atoms(rep::Vector{Int64}, type::LATTICE.TYPE)

        Calculate the number of atoms in a lattice of a given 'type', for the
        given amount of repetitions 'rep' (should be a 3x1 Vector{Int64, 1},
        with repetitions in each of the 3 dimensions). Supported lattice types
        include 'primitive', 'body_centered' and 'face_centered' lattices.
        Returns an Int64.

    Example:

        calculate_n_atoms([10, 10, 10], LATTICE.primitive)
"""
function calculate_n_atoms(rep::Vector{Int64}, type::LATTICE.TYPE)::Int64

    @assert size(rep) == (3,) "rep argument must be a 3x1 Array{Int64, 1}"

    if type == LATTICE.primitive
        n_atoms = (rep[1] + 1) * (rep[2] + 1) * (rep[3] + 1)
    elseif type == LATTICE.body_centered
        n_atoms = (rep[1] + 1) * (rep[2] + 1) * (rep[3] + 1)
        n_atoms += rep[1]*rep[2]*rep[3]
    elseif type == LATTICE.face_centered
        n_atoms = (rep[1] + 1) * (rep[2] + 1) * (rep[3] + 1)
        n_atoms += rep[1]*rep[2]*rep[3] * 3
        n_atoms += rep[1]*rep[2] + rep[2] * rep[3] + rep[3] * rep[1]
    end
    return n_atoms
end

"""
    generate_cubic_lattice(sizes::Float64, rep::Int64, type::LATTICE.TYPE)

        Generates a cubic lattice with all dimensions lenght set to 'sizes' (in
        nm). Each dimension contains 'rep' repeats of the particle template, of
        a specific 'type'. Supported lattice types include 'primitive',
        'body_centered' and 'face_centered' lattices.

        Returns a Matrix{Float64} in Array of Structures (AoS) format.
        Use 'convert' function to change the content type.
        Ex: convert(Matrix{Float32}, array)
        Use 'collect' function with the transpose to change the format to SoA.
        Ex: collect(array')

    Example:

        generate_cubic_lattice(10.0, 10, LATTICE.primitive)
"""
function generate_cubic_lattice(sizes::Float64, rep::Int64, type::LATTICE.TYPE)::Matrix{Float64}
        
    xyz = generate_lattice([sizes, sizes, sizes], [rep, rep, rep], type)
    return xyz
end

"""
    generate_lattice(sizes::Vector{Float64, 1}, rep::Vector{Int64, 1}, type::LATTICE.TYPE)

        Generates a lattice with dimensions lenght set to 'sizes' (should be a
        3x1 array with each dimanesion size, in nm). Each dimension contains
        'rep' repeats of the particle template of a specific 'type' (should be a
        3x1 array with each dimension repetitions). Supported lattice types
        include 'primitive', 'body_centered' and 'face_centered' lattices.

        Returns a Matrix{Float64} in Array of Structures (AoS) format. Use
        'convert function to change the content type.
        Ex: convert(Matrix{Float32}, array)

    Example:

        generate_lattice([10.0, 10.0, 5.0], [10, 10, 5], LATTICE.primitive)
"""
function generate_lattice(sizes::Vector{Float64}, rep::Vector{Int64},
    type::LATTICE.TYPE)::Matrix{Float64}

    @assert size(sizes) == (3,) "sizes argument must be a 3x1 Array{Float64, 1}"
    @assert size(rep) == (3,) "rep argument must be a 3x1 Array{Int64, 1}"

    # Generate the template
    cs = (sizes./rep)
    template = generate_template(cs, type)

    # Create empty list of atoms
    n_atoms  = calculate_n_atoms(rep, type)
    xyz = zeros(3, n_atoms)

    # Fill xyz with repetitions of template
    c = 1
    for i in 0:rep[1]
        for j in 0:rep[2]
            for k in 0:rep[3]
                for index in 1:size(template, 1)
                    new_pos = template[index, :] .+ [i*cs[1], j*cs[2], k*cs[3]]
                    if all(new_pos .<= sizes)
                        xyz[1:3, c] .= new_pos
                        c +=1
                    end
                end
            end
        end
    end

    return xyz
end

"""
    print_xyz(coords::Matrix{T}, [io::Union{String, Base.TTY} = Base.stdout]) where {T <: AbstractFloat}

        Print the contents of a Matrix{T}, where T is a Float and the format can
        be either AoS or SoA, in a XYZ readable format. By default, frints to
        stdout, but a file name can be given in argumento 'io' to print to file
        instead. 

    Example:
    
        print_xyz(coords)
        print_xyz(coords, "filename.xyz")
"""
function print_xyz(coords::Matrix{T}, io::Union{String, Base.TTY} = Base.stdout) where {T <: AbstractFloat}

    @assert min(size(coords)...) == 3 "Minimum size of matrix must be 3."
    AoS = false
    if argmin(collect(size(coords))) == 1
        AoS = true
    end

    if !isa(io, Base.TTY)
        io = open(io, "w")
    end

    for i in 1:max(size(coords)...)
        if AoS
            xyz = coords[:, i]
        else
            xyz = coords[i, :]
        end
        println(io, @sprintf "%-5s %8.2f %8.2f %8.2f" "H" xyz[1] xyz[2] xyz[3])
    end

    if !isa(io, Base.TTY)
        println(" Printed coordinates to $(io.name)")
        close(io)
    end
end