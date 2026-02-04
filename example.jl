using TensorOperations
using LinearAlgebra
include("src/MyPkg.jl")
using .MyPkg
using Plots, Printf

# %%
# DMRG sweep

N = 100 # number of sites
d = 2 # physical dim
D = 30 # bond dim

BC = "PBC"
E_Bethe = heisen_chain_Bethe(N, BC)
# generate mpo of N-site PBC heisenberg chain
mpo = heisen_chain_MPO(N, BC)

# generate random mps
mps_rnd = MPS{Float64}(N, d, D)
r2l_LQ!(mps_rnd)


# @time λs = DMRG_loop!(mps_rnd, mpo, 2, 1e-12)
@time (E_dmrg, sites_updated) = DMRG_converge!(mps_rnd, mpo, 200, 1e-12)
println("Sites updated: ", sites_updated)


println("DMRG Final Energy:   ", E_dmrg)
println("Bethe Ansatz Energy: ", E_Bethe)

# %%
# plot error of different N and D
N_lst = [20, 40, 60, 80, 100]
D_lst = [10, 20, 40, 60]
E_lst = zeros(length(N_lst), length(D_lst))
err_lst = zeros(length(N_lst), length(D_lst))

BC = "PBC"
for (i, N) in enumerate(N_lst)
    E_Bethe = heisen_chain_Bethe(N, BC)
    mpo = heisen_chain_MPO(N, BC)
    for (j, D) in enumerate(D_lst)
        # generate random mps
        mps_rnd = MPS{Float64}(N, 2, D)
        r2l_LQ!(mps_rnd)
        (E_dmrg, sites_updated) = DMRG_converge!(mps_rnd, mpo, 500, 1e-12)
        # λs = DMRG_loop!(mps_rnd, mpo, 2, 1e-6)
        E_lst[i, j] = E_dmrg
    end
    err_lst[i, :] = (E_lst[i, :] .- E_Bethe) ./ E_Bethe
end

# %%

p = plot()
for (i, N) in enumerate(N_lst)
    plot!(p, D_lst, -err_lst[i, :]; label="N=$N", xlabel="Bond Dimension", ylabel="relative error", marker=:circle, markersize=2)
end
display(p)

# %%
# step-wise Visualization
# E_Bethe = heisen_chain_Bethe(N, BC)
# E_err = abs(E_dmrg - E_Bethe)
# println("Theoretical ground energy: ", E_Bethe)

# p = plot(λs; label="DMRG variational energy", xlabel="update steps", ylabel="λ",
#     linewidth=2, marker=:circle, markersize=2)

# yt, _ = yticks(p)[1]
# new_yticks = sort(vcat(yt, E_Bethe))
# ytick_labels = [@sprintf("%.2f", y) for y in new_yticks]
# yticks!(p, new_yticks, ytick_labels)
# hline!(p, [E_Bethe]; label="Bethe ansatz ground energy", linestyle=:dash, linewidth=2)

# title!(p, "$N sites, $D bond dim, $BC, error = $(round(E_err, digits=4))")