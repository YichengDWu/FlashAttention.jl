using FlashAttention
using Test
using CUDA, NNlib, NNlibCUDA

# sanity check
function ref_attention(Q,K,V)
    S = batched_mul(permutedims(K, (2,1,3,4)), Q)
    P = softmax(S)
    O = batched_mul(V, P)
    return O
end

@testset "FlashAttention.jl" begin
    Q = CUDA.rand(3,256,4,3);
    K = CUDA.rand(3,256,4,3);
    V = CUDA.rand(3,256,4,3);
    O = flash_attention(Q,K,V);
    O_ref = ref_attention(Q,K,V);
    @test O ≈ O_ref
end
