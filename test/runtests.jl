using FlashAttention
using Test
using NNlib, NNlibCUDA

import CUDA

if CUDA.functional()
    using CUDA  # exports CuArray, etc
    @info "starting CUDA tests"
else
    @info "CUDA not functional, testing via JLArrays"
    using JLArrays
    JLArrays.allowscalar(false)
    CUDA.cu(x) = jl(x)

end

# sanity check
function ref_attention(Q,K,V)
    S = batched_mul(permutedims(K, (2,1,3,4)), Q)
    P = softmax(S)
    O = batched_mul(V, P)
    return O
end

@testset "FlashAttention.jl" begin
    Q = CUDA.cu(rand(3,256,4,3));
    K = CUDA.cu(rand(3,256,4,3));
    V = CUDA.cu(rand(3,256,4,3));
    O = flash_attention(Q,K,V);
    O_ref = ref_attention(Q,K,V);
    @test O ≈ O_ref
end
