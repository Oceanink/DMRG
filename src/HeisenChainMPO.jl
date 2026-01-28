export heisen_chain_MPO

function heisen_chain_MPO(N::Int)::MPO
    hbar = 1
    Sz = (hbar / 2) * [1 0; 0 -1]
    Sp = hbar * [0 1; 0 0]
    Sm = hbar * [0 0; 1 0]
    I2 = [1 0; 0 1]

    d = 2

    row = zeros(4, d, d)
    row[1, :, :] = 0.5 * Sm
    row[2, :, :] = 0.5 * Sp
    row[3, :, :] = Sz
    row[4, :, :] = I2

    column = zeros(4, d, d)
    column[1, :, :] = I2
    column[2, :, :] = Sp
    column[3, :, :] = Sm
    column[4, :, :] = Sz


    D_vec = Vector{Int}(undef, N + 1)
    D_vec[1] = 1
    D_vec[N+1] = 1
    for i in 2:N
        D_vec[i] = 5
    end

    # (Dl, Dr, d, d)
    O = Vector{Array{Float64,4}}(undef, N)
    for i in 1:N
        O[i] = zeros(D_vec[i], D_vec[i+1], d, d)
    end

    O[1][1, 2:5, :, :] = row

    O[2][5, 2:5, :, :] = row
    O[2][1:4, 1, :, :] = column

    O[N][1:4, 1, :, :] = column

    for i in 3:N-1
        O[i] = O[2]
    end

    mpo = MPO{Float64}(O, N, d)

    return mpo
end