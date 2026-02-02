using TensorOperations
using LinearAlgebra
include("src/MyPkg.jl")
using .MyPkg
using Plots, Printf


# %%
# DMRG sweep

N = 50 # number of sites
d = 2 # physical dim
D = 20 # bond dim

# generate random mps
mps_rnd = MPS{Float64}(N, d, D)
r2l_LQ!(mps_rnd)

# generate mpo of N-site PBC heisenberg chain
BC = "PBC"
mpo = heisen_chain_MPO(N, BC)

@time λs = DMRG_loop!(mps_rnd, mpo, 2, 1e-6)

println("Final energy: ", λs[end])

# %%
# Visualization

# for OBC
# E_exact = N * (0.25 - log(2)) + (pi - 1 - 2 * log(2)) / 4

# for PBC
E_exact = N * (0.25 - log(2))

E_err = abs(λs[end] - E_exact)
println("Theoretical ground energy: ", E_exact)

p = plot(λs; label="DMRG variational energy", xlabel="update steps", ylabel="λ",
    linewidth=2, marker=:circle, markersize=2)

yt, _ = yticks(p)[1]
new_yticks = sort(vcat(yt, E_exact))
ytick_labels = [@sprintf("%.2f", y) for y in new_yticks]
yticks!(p, new_yticks, ytick_labels)
hline!(p, [E_exact]; label="Theoretical ground energy", linestyle=:dash, linewidth=2)

title!(p, "$N sites, $D bond dim, $BC, error = $(round(E_err, digits=4))")