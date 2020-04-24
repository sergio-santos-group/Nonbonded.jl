using Test
using LennardJones

# 1. Generating lattices
@test LennardJones.generate_template([1.0, 1.0, 1.0], LATTICE.primitive) == [0.0 0.0 0.0]
@test LennardJones.generate_template([1.0, 1.0, 1.0], LATTICE.body_centered) == [0.0 0.0 0.0; 0.5 0.5 0.5]
@test LennardJones.generate_template([1.0, 1.0, 1.0], LATTICE.face_centered) == [0.0 0.0 0.0; 0.5 0.5 0.0; 0.5 0.0 0.5; 0.0 0.5 0.5]
xyz_AoS = convert(Matrix{Float32}, generate_cubic_lattice(1.0, 1, LATTICE.primitive))
@test xyz_AoS == [[0.0, 0.0, 0.0] [0.0, 0.0, 1.0] [0.0, 1.0, 0.0] [0.0, 1.0, 1.0] [1.0, 0.0, 0.0] [1.0, 0.0, 1.0] [1.0, 1.0, 0.0] [1.0, 1.0, 1.0]]

# 2. Generating Verlet lists
# 2.1 AoS format
vl = VerletList(max(size(xyz_AoS)...))
vl.cutoff = 1.2
update_serial!(vl, xyz_AoS)
@test vl.list[1:end-4] == [2, 3, 5, -1, 4, 6, -1, 4, 7, -1, 8, -1, 6, 7, -1, 8, -1, 8, -1, -1]
# Note: Last 4 elements are not tested as they are not filled. resize! can either place 0's or random numbers

# 2.2 SoA format
xyz_SoA = collect(xyz_AoS')
vl = VerletList(max(size(xyz_SoA)...))
vl.cutoff = 1.2
update_serial!(vl, xyz_SoA)
@test vl.list[1:end-4] == [2, 3, 5, -1, 4, 6, -1, 4, 7, -1, 8, -1, 6, 7, -1, 8, -1, 8, -1, -1]
# Note: Last 4 elements are not tested as they are not filled. resize! can either place 0's or random numbers

# 3. Serial energy calculation

state_AoS = State(8, xyz_AoS, zeros(Float32, 3, 8))
energy = LennardJones.naive_aos(state_AoS, Float32(Inf), false)

@test serial_no_verlet_AoS(state_AoS, Float32(Inf), LennardJones.NOFORCES) == energy