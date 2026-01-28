using TensorOperations
using LinearAlgebra
using KrylovKit
include("src/MyPkg.jl")
using .MyPkg
using Plots, Printf

# %%
# Utility functions

function mps_norm(mps::MPS)::Float64
    N = mps.N
    C = ones(ComplexF64, 1, 1)
    for n in 1:N
        An = mps.A[n]
        An_dag = conj(An)
        @tensor C[k, l] := C[i, j] * An_dag[j, d, l] * An[i, d, k]
    end

    return sqrt(real(C[1, 1]))
end

function r2l_LQ!(mps::MPS)
    """Right-to-left LQ decomposition, modifies MPS in-place"""
    N = mps.N

    An = mps.A[N]
    Dl, d, Dr = size(An)
    An_mat = reshape(An, Dl, d * Dr)

    for n in N-1:-1:1
        L, Q = lq(An_mat)
        Q = Matrix(Q)

        mps.A[n+1] = reshape(Q, size(Q, 1), d, Dr)

        @tensor An[i, j, l] := mps.A[n][i, j, k] * L[k, l]
        Dl, d, Dr = size(An)
        An_mat = reshape(An, Dl, d * Dr)
    end

    mps.A[1] = An / norm(An)
    return nothing
end

function _l2r_QR(A::Array{T,3})::Array{T,3} where {T}
    Dl, d, Dr = size(A)
    A_mat = reshape(A, Dl * d, Dr)

    Q, _ = qr(A_mat)
    Q = Matrix(Q)

    return reshape(Q, Dl, d, size(Q, 2))
end

function _r2l_LQ(A::Array{T,3})::Array{T,3} where {T}
    Dl, d, Dr = size(A)
    A_mat = reshape(A, Dl, d * Dr)

    _, Q = lq(A_mat)
    Q = Matrix(Q)

    return reshape(Q, size(Q, 1), d, Dr)
end

# %%
# DMRG functions

function DMRG_1step(left_env::Array{T,3}, O::Array{T2,4}, right_env::Array{T,3}) where {T,T2}
    """Single DMRG optimization step using iterative eigensolver

    """
    @tensor H_eff[u, i, o, j, k, l] := left_env[u, y, j] * O[y, p, i, k] * right_env[o, p, l]

    H_eff_size = size(H_eff)
    dim1 = prod(H_eff_size[1:3])
    dim2 = prod(H_eff_size[4:6])


    # Find only the smallest eigenvalue using iterative method
    # :SR means "smallest real" eigenvalue
    H_eff_mat = reshape(H_eff, dim1, dim2)
    λs, vecs, _ = eigsolve(H_eff_mat, 1, :SR)

    λ = real(λs[1])
    An_new = reshape(vecs[1], H_eff_size[4:6])

    return An_new, λ
end

function l2r_DMRG!(mps::MPS, mpo::MPO,
    right_envs::Vector{Array{T,3}},
    left_envs::Vector{Array{T,3}},
    λs::Vector{Float64}) where {T}
    """Left-to-right DMRG sweep from site 1 to site N-1
    Modifies MPS in-place and reuses preallocated left_envs and λs arrays.
    """
    N = mps.N
    left_envs[1] = ones(1, 1, 1)

    for n in 1:N-1
        left_env = left_envs[n]
        right_env = right_envs[n]
        On = mpo.O[n]

        # update site n
        An_new, λ = DMRG_1step(left_env, On, right_env)

        # normalize and store
        Q = _l2r_QR(An_new)
        mps.A[n] = Q
        λs[n] = λ

        # Update left environment
        @tensor left_env_new[o, p, l] := left_env[u, y, j] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
        left_envs[n+1] = left_env_new
    end

    return nothing
end

function r2l_DMRG!(mps::MPS, mpo::MPO,
    left_envs::Vector{Array{T,3}},
    right_envs::Vector{Array{T,3}},
    λs::Vector{Float64}) where {T}
    """Right-to-left DMRG sweep from site N to site 2
    Modifies MPS in-place and reuses preallocated right_envs and λs arrays.
    """
    N = mps.N
    right_envs[N] = ones(1, 1, 1)

    for n in N:-1:2
        left_env = left_envs[n]
        right_env = right_envs[n]
        On = mpo.O[n]

        # update site n
        An_new, λ = DMRG_1step(left_env, On, right_env)

        # normalize and store
        Q = _r2l_LQ(An_new)
        mps.A[n] = Q
        λs[N+1-n] = λ

        # Update right environment
        @tensor right_env_new[u, y, j] := right_env[o, p, l] * conj(Q)[u, i, o] * On[y, p, i, k] * Q[j, k, l]
        right_envs[n-1] = right_env_new
    end

    return nothing
end

function l2r_DMRG_prep(mps::MPS{T}, mpo::MPO) where {T}
    """Prepare right environments for the first left-to-right sweep"""
    N = mps.N
    right_envs = Vector{Array{T,3}}(undef, N)
    right_envs[N] = ones(1, 1, 1)

    for n in N:-1:2
        On = mpo.O[n]
        An = mps.A[n]
        right_env = right_envs[n]
        @tensor right_env[u, y, j] := right_env[o, p, l] * conj(An)[u, i, o] * On[y, p, i, k] * An[j, k, l]
        right_envs[n-1] = right_env
    end
    return right_envs
end

function DMRG_loop!(mps::MPS{T}, mpo::MPO, times::Int, threshold::Float64) where {T}
    """Main DMRG loop 
    - Preallocates all arrays
    - Reuses environment tensors
    - Modifies MPS in-place
    """
    N = mps.N

    # Preallocate environments (reused across sweeps)
    left_envs = Vector{Array{T,3}}(undef, N)
    right_envs = l2r_DMRG_prep(mps, mpo)

    # Preallocate energy array with maximum possible size
    max_size = times * 2 * (N - 1)
    λs_all = Vector{Float64}(undef, max_size)

    # Preallocate temporary arrays for each sweep
    λs_lr = Vector{Float64}(undef, N - 1)
    λs_rl = Vector{Float64}(undef, N - 1)

    idx = 0 # index of last stored energy
    i = 0 # index of loops
    e = 100 # initial error

    while i < times && e > threshold
        # Left-to-right sweep
        l2r_DMRG!(mps, mpo, right_envs, left_envs, λs_lr)
        λs_all[idx+1:idx+N-1] .= λs_lr[1:N-1]
        idx += N - 1

        # Right-to-left sweep
        r2l_DMRG!(mps, mpo, left_envs, right_envs, λs_rl)
        λs_all[idx+1:idx+N-1] .= λs_rl[1:N-1]
        idx += N - 1

        # Check convergence
        if idx >= 2
            e = λs_all[idx-1] - λs_all[idx]
        end

        i += 1
    end

    return λs_all[1:idx]
end

# %%
# DMRG sweep

N = 50 # number of sites
d = 2 # physical dim
D = 20 # bond dim

# generate random mps
mps_rnd = MPS{Float64}(N, d, D)
r2l_LQ!(mps_rnd)

# generate mpo of N-site heisenberg chain
mpo = heisen_chain_MPO(N)

@time λs = DMRG_loop!(mps_rnd, mpo, 2, 1e-6)

println("Final energy: ", λs[end])

# %%
# Visualization

E_exact = (N - 1) * (0.25 - log(2))
E_err = abs(λs[end] - E_exact)
println("Theoretical ground energy: ", E_exact)

p = plot(λs; label="DMRG variational energy", xlabel="update steps", ylabel="λ",
    linewidth=2, marker=:circle, markersize=2)

yt, _ = yticks(p)[1]
new_yticks = sort(vcat(yt, E_exact))
ytick_labels = [@sprintf("%.2f", y) for y in new_yticks]
yticks!(p, new_yticks, ytick_labels)
hline!(p, [E_exact]; label="Theoretical ground energy", linestyle=:dash, linewidth=2)

title!(p, "$N sites, $D bond dim, error = $(round(E_err, digits=4))")