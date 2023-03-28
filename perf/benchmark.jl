include("../src/FlashAttention.jl")

using .FlashAttention
using CUDA, BenchmarkTools, Plots
using NNlib, NNlibCUDA

function bench_flash_attn(Q, K, V)
    CUDA.@sync begin
        flash_attention(Q, K, V)
    end
end

function ref_attention(Q, K, V)
    S = batched_mul(permutedims(K, (2, 1, 3, 4)), Q)
    P = softmax(S)
    O = batched_mul(V, P)
    return O
end

function main()
    seq_lens = 1 .<< collect(7:10)
    heads = 4
    batch = 1
    d = 4

    y_flash = Float64[]
    y_naive = Float64[]

    for seq_len in seq_lens
        Q = CUDA.rand(Float16, d, seq_len, heads, batch)
        K = CUDA.rand(Float16, d, seq_len, heads, batch)
        V = CUDA.rand(Float16, d, seq_len, heads, batch)
        t1 = @belapsed bench_flash_attn($Q, $K, $V)
        push!(y_flash, t1)

        CUDA.reclaim()
        t2 = @belapsed ref_attention($Q, $K, $V)
        push!(y_naive, t2)
    end

    plot(seq_lens, y_flash, label="FlashAttention")
    plot!(seq_lens, y_naive, label="Naive")
    xlabel!("Sequence Length")
    ylabel!("Time (s)")
end
