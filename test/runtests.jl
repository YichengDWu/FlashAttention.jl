using FlashAttention
using Test
using NNlib, NNlibCUDA

using CUDA

# sanity check
function ref_attention(Q, K, V)
    sq, sk = size(Q), size(K)
    Q = reshape(Q, sq[1], sq[2], sq[3] * sq[4])
    K = reshape(K, sk[1], sk[2], sk[3] * sk[4])
    S = CUBLAS.gemm_strided_batched('T', 'N', Q, K)
    S = reshape(S, sk[2], sq[2], sq[3], sq[4])
    P = softmax(S)
    O = batched_mul(V, P)
    return O
end

if CUDA.functional()
    @testset "FlashAttention.jl" begin
        Q = CUDA.rand(Float16, 3, 255, 4, 3)
        K = CUDA.rand(Float16, 3, 255, 4, 3)
        V = CUDA.rand(Float16, 3, 255, 4, 3)
        O = flash_attention(Q, K, V)
        O_ref = ref_attention(Q, K, V)
        @test O ≈ O_ref
    end
end
