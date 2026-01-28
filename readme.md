# One-site DMRG on Heisenberg Chain

## Project Overview

This is a Julia implementation of the DMRG (Density Matrix Renormalization Group) algorithm for finding ground states of the Heisenberg spin chain.

## Architecture

### Data Structures (src/MatrixProductStruct.jl)

- **MPS{T}**: Matrix Product State representing quantum states as a tensor train
  - `A::Vector{Array{T,3}}`: Array of 3D tensors with indices `(left_bond, physical, right_bond)`
  - `N::Int`: Number of sites in the chain
  - `d::Int`: Physical dimension (2 for spin-1/2)
  - Constructor `MPS{T}(N, d, D)` creates random MPS with bond dimension D

- **MPO{T}**: Matrix Product Operator representing quantum operators
  - `O::Vector{Array{T,4}}`: Array of 4D tensors with indices `(left_bond, right_bond, physical_in, physical_out)`
  - Bond dimensions: 1 at boundaries, 5 in bulk for Heisenberg chain

### Heisenberg Chain MPO (src/HeisenChainMPO.jl)

The `heisen_chain_MPO(N)` function constructs the MPO for the Heisenberg Hamiltonian using a specific tensor structure:

- Uses Pauli matrices: Sz (spin-z), Sp (spin-plus), Sm (spin-minus)
- Implements nearest-neighbor interactions via a 5-dimensional bond space
- Special handling for first, last, and bulk sites

### DMRG Algorithm (dmrg.jl)

The DMRG implementation uses a one-site sweeping algorithm:

1. **Preparation** (`l2r_DMRG_prep`): Computes all right environments before the first sweep
2. **Left-to-right sweep** (`l2r_DMRG`): Updates sites 1 to N-1, maintaining left-canonical form via QR decomposition
3. **Right-to-left sweep** (`r2l_DMRG`): Updates sites N to 2, maintaining right-canonical form via LQ decomposition
4. **Main loop** (`DMRG_loop`): Alternates between forward and backward sweeps until convergence

**Key functions:**

- `DMRG_1step(left_env, O, right_env)`: Constructs effective Hamiltonian and finds ground state via eigenvalue decomposition
- `_l2r_QR(A)` and `_r2l_LQ(A)`: Orthogonalization helpers that maintain canonical forms
- Environment tensors have indices `(bond_mps, bond_mpo, bond_mps)` and are updated incrementally

### Tensor Conventions

- All tensor contractions use the `@tensor` macro from TensorOperations.jl
- MPS tensors: `A[left, physical, right]`
- MPO tensors: `O[left, right, phys_in, phys_out]`
- Environment tensors: `env[mps_bond, mpo_bond, mps_bond]`

## Dependencies

- `TensorOperations`: Efficient tensor contractions
- `LinearAlgebra`: QR/LQ decompositions and eigenvalue solvers
- `KrylovKit`: For iterative eigensolving
- `Plots`, `Printf`: Visualization and output formatting
