using Revise
push!(LOAD_PATH, "src")
using LennardJones
using SIMD
using BenchmarkTools
using Statistics
using Printf

# T       = Float64
# cut_off = T(99.0)
# # NS      = 1:21

# xyz_aos   = generate_cubic_lattice(15, LATTICE.face_centered)
# # xyz_aos   = collect(mapreduce(permutedims, vcat, [[y for x in 1:3] for y in 1:5])')
# xyz_aos   = convert(Matrix{T}, xyz_aos)
# state_aos = State(xyz_aos)
# n_atoms   = max(size(xyz_aos)...)
# # correct_energy = serial(state_aos, T(Inf), LennardJones.NOFORCES)
# measured_energy = cuda(state_aos, cut_off)
# @printf("cut-off: %5.1f | %5.2f (real) - %5.2f (measured) = %7.4f (N: %5d)\n",
#     cut_off, correct_energy, measured_energy, correct_energy - measured_energy, n_atoms)

# function calc_max_atoms_in_vlist(natoms::Int64, vlist::VerletList)

#     counts = Vector{Int64}()
#     for i in 1:natoms
#         counter = 0
#         ptr = vlist.offset[i]
#         if ptr > 0
#             while vlist.list[ptr] > 0
#                 counter += 1
#                 ptr += 1
#             end
#         end
#         push!(counts, counter)
#     end
#     return maximum(counts)
# end

T       = Float64
cut_off = T(11.0)
NS      = 1:21

# Generate state
for cut_off in Vector{T}(collect(5:30))
    xyz_aos   = generate_cubic_lattice(21, LATTICE.face_centered)
    xyz_aos   = convert(Matrix{T}, xyz_aos)
    xyz_soa   = collect(xyz_aos')
    state_aos = State(xyz_aos)
    state_soa = State(xyz_soa)
    n_atoms   = max(size(xyz_aos)...)
    vlist = VerletList(n_atoms)
    vlist.cutoff = cut_off
    vlist = update_serial!(vlist, state_aos)
    # m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)
    correct_energy = serial(state_soa, T(Inf), LennardJones.NOFORCES)
    measured_energy = cuda(state_aos, cut_off)
    # measured_energy = simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{4, T})
    @printf("cut-off: %5.1f | %5.2f (real) - %5.2f (measured) = %7.4f (N: %5d)\n", cut_off, correct_energy, measured_energy, correct_energy - measured_energy, n_atoms)
end

# # SIMD-4 | No Verlet | AoS
# for N in NS
#     # Generate state
#     global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
#     global xyz_aos   = convert(Matrix{T}, xyz_aos)
#     global xyz_soa   = collect(xyz_aos')

#     global state_aos = State(xyz_aos)
#     global state_soa = State(xyz_soa)
#     global n_atoms   = max(size(xyz_aos)...)

#     # Verlet list
#     global vlist = VerletList(n_atoms)
#     vlist.cutoff = cut_off
#     vlist = update_serial!(vlist, state_aos)
#     global m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)

#     b = @benchmark simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{4, T})
#     println("SIMD-4 | No Verlet | AoS | N: $N | # atoms: $n_atoms | # interacting atoms: $m_atoms | $(mean(b).time) ± $(std(b).time) ns")
# end

# # SIMD-4 | With Verlet | AoS
# for N in NS
#     # Generate state
#     global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
#     global xyz_aos   = convert(Matrix{T}, xyz_aos)
#     global xyz_soa   = collect(xyz_aos')

#     global state_aos = State(xyz_aos)
#     global state_soa = State(xyz_soa)
#     global n_atoms   = max(size(xyz_aos)...)

#     # Verlet list
#     global vlist = VerletList(n_atoms)
#     vlist.cutoff = cut_off
#     vlist = update_serial!(vlist, state_aos)
#     global m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)

#     b = @benchmark simd(state_aos, vlist, LennardJones.NOFORCES, Vec{4, T})
#     println("SIMD-4 | With Verlet | AoS | N: $N | # atoms: $n_atoms | # interacting atoms: $m_atoms | $(mean(b).time) ± $(std(b).time) ns")
# end

# # SIMD-8 | No Verlet | AoS
# for N in NS
#     # Generate state
#     global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
#     global xyz_aos   = convert(Matrix{T}, xyz_aos)
#     global xyz_soa   = collect(xyz_aos')

#     global state_aos = State(xyz_aos)
#     global state_soa = State(xyz_soa)
#     global n_atoms   = max(size(xyz_aos)...)

#     # Verlet list
#     global vlist = VerletList(n_atoms)
#     vlist.cutoff = cut_off
#     vlist = update_serial!(vlist, state_aos)
#     global m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)

#     b = @benchmark simd(state_aos, cut_off, LennardJones.NOFORCES, Vec{8, T})
#     println("SIMD-8 | No Verlet | AoS | N: $N | # atoms: $n_atoms | # interacting atoms: $m_atoms | $(mean(b).time) ± $(std(b).time) ns")
# end

# # SIMD-8 | With Verlet | AoS
# for N in NS
#     # Generate state
#     global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
#     global xyz_aos   = convert(Matrix{T}, xyz_aos)
#     global xyz_soa   = collect(xyz_aos')

#     global state_aos = State(xyz_aos)
#     global state_soa = State(xyz_soa)
#     global n_atoms   = max(size(xyz_aos)...)

#     # Verlet list
#     global vlist = VerletList(n_atoms)
#     vlist.cutoff = cut_off
#     vlist = update_serial!(vlist, state_aos)
#     global m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)

#     b = @benchmark simd(state_aos, vlist, LennardJones.NOFORCES, Vec{8, T})
#     println("SIMD-8 | With Verlet | AoS | N: $N | # atoms: $n_atoms | # interacting atoms: $m_atoms | $(mean(b).time) ± $(std(b).time) ns")
# end

# CUDA | AoS
for N in NS
    # Generate state
    global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
    global xyz_aos   = convert(Matrix{T}, xyz_aos)
    global xyz_soa   = collect(xyz_aos')

    global state_aos = State(xyz_aos)
    global state_soa = State(xyz_soa)
    global n_atoms   = max(size(xyz_aos)...)

    # Verlet list
    global vlist = VerletList(n_atoms)
    vlist.cutoff = cut_off
    vlist = update_serial!(vlist, state_aos)

    b = @benchmark cuda(state_aos, cut_off)
    println("CUDA | AoS | N: $N | T: $T | Cut-off: $cut_off | # atoms: $n_atoms | $(mean(b).time) ± $(std(b).time) ns")
end

# --- SOA ----------------------------------------------------------------------

# # SIMD-4 | No Verlet | SoA
# for N in NS
#     # Generate state
#     global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
#     global xyz_aos   = convert(Matrix{T}, xyz_aos)
#     global xyz_soa   = collect(xyz_aos')

#     global state_aos = State(xyz_aos)
#     global state_soa = State(xyz_soa)
#     global n_atoms   = max(size(xyz_aos)...)

#     # Verlet list
#     global vlist = VerletList(n_atoms)
#     vlist.cutoff = cut_off
#     vlist = update_serial!(vlist, state_aos)
#     global m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)

#     b = @benchmark simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{4, T})
#     println("SIMD-4 | No Verlet | SoA | N: $N | # atoms: $n_atoms | # interacting atoms: $m_atoms | $(mean(b).time) ± $(std(b).time) ns")
# end

# # SIMD-4 | With Verlet | SoA
# for N in NS
#     # Generate state
#     global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
#     global xyz_aos   = convert(Matrix{T}, xyz_aos)
#     global xyz_soa   = collect(xyz_aos')

#     global state_aos = State(xyz_aos)
#     global state_soa = State(xyz_soa)
#     global n_atoms   = max(size(xyz_aos)...)

#     # Verlet list
#     global vlist = VerletList(n_atoms)
#     vlist.cutoff = cut_off
#     vlist = update_serial!(vlist, state_aos)
#     global m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)

#     b = @benchmark simd(state_soa, vlist, LennardJones.NOFORCES, Vec{4, T})
#     println("SIMD-4 | With Verlet | SoA | N: $N | # atoms: $n_atoms | # interacting atoms: $m_atoms | $(mean(b).time) ± $(std(b).time) ns")
# end

# # SIMD-8 | No Verlet | SoA
# for N in NS
#     # Generate state
#     global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
#     global xyz_aos   = convert(Matrix{T}, xyz_aos)
#     global xyz_soa   = collect(xyz_aos')

#     global state_aos = State(xyz_aos)
#     global state_soa = State(xyz_soa)
#     global n_atoms   = max(size(xyz_aos)...)

#     # Verlet list
#     global vlist = VerletList(n_atoms)
#     vlist.cutoff = cut_off
#     vlist = update_serial!(vlist, state_aos)
#     global m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)

#     b = @benchmark simd(state_soa, cut_off, LennardJones.NOFORCES, Vec{8, T})
#     println("SIMD-8 | No Verlet | SoA | N: $N | # atoms: $n_atoms | # interacting atoms: $m_atoms | $(mean(b).time) ± $(std(b).time) ns")
# end

# # SIMD-8 | With Verlet | SoA
# for N in NS
#     # Generate state
#     global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
#     global xyz_aos   = convert(Matrix{T}, xyz_aos)
#     global xyz_soa   = collect(xyz_aos')

#     global state_aos = State(xyz_aos)
#     global state_soa = State(xyz_soa)
#     global n_atoms   = max(size(xyz_aos)...)

#     # Verlet list
#     global vlist = VerletList(n_atoms)
#     vlist.cutoff = cut_off
#     vlist = update_serial!(vlist, state_aos)
#     global m_atoms = calc_max_atoms_in_vlist(n_atoms, vlist)

#     b = @benchmark simd(state_soa, vlist, LennardJones.NOFORCES, Vec{8, T})
#     println("SIMD-8 | With Verlet | SoA | N: $N | # atoms: $n_atoms | # interacting atoms: $m_atoms | $(mean(b).time) ± $(std(b).time) ns")
# end

# CUDA | SoA
for N in NS
    # Generate state
    global xyz_aos   = generate_cubic_lattice(N, LATTICE.face_centered)
    global xyz_aos   = convert(Matrix{T}, xyz_aos)
    global xyz_soa   = collect(xyz_aos')

    global state_aos = State(xyz_aos)
    global state_soa = State(xyz_soa)
    global n_atoms   = max(size(xyz_aos)...)

    # Verlet list
    global vlist = VerletList(n_atoms)
    vlist.cutoff = cut_off
    vlist = update_serial!(vlist, state_aos)

    b = @benchmark cuda(state_soa, cut_off)
    println("CUDA | SoA | N: $N | T: $T | Cut-off: $cut_off | # atoms: $n_atoms | $(mean(b).time) ± $(std(b).time) ns")
end