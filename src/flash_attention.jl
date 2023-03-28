function flash_attention_kernel(Q, K, V, O)
    NK = size(K, 2)
    NQ = size(Q, 2)
    d = size(Q, 1)
    tx = threadIdx().x
    Bs = blockDim().x # assume Br == Bc
    idx = (blockIdx().x - 1) * Bs + tx
    T = eltype(Q)

    offset = 0
    q = CuDynamicSharedArray(T, (d, Bs), offset)
    offset += sizeof(q)
    o = CuDynamicSharedArray(T, (d, Bs), offset)
    offset += sizeof(o)
    k = CuDynamicSharedArray(T, (d, Bs), offset)
    offset += sizeof(k)
    lᵢ = CuDynamicSharedArray(T, (1, Bs), offset)
    offset += sizeof(lᵢ)
    mᵢ = CuDynamicSharedArray(T, (1, Bs), offset)
    offset += sizeof(mᵢ)
    s = CuDynamicSharedArray(T, (Bs, Bs), offset)
    offset += sizeof(s)
    m̃ᵢⱼ = CuDynamicSharedArray(T, (1, Bs), offset)
    offset += sizeof(m̃ᵢⱼ)
    P̃ᵢⱼ = CuDynamicSharedArray(T, (Bs, Bs), offset)

    # load Q to shared memory, note that this is done only once
    if idx <= NQ
        for i in 1:d
            q[i, tx] = Q[i, idx, blockIdx().y, blockIdx().z]
        end
    end

    # initialize lᵢ and mᵢ
    lᵢ[tx] = zero(T)
    mᵢ[tx] = -Inf

    # initialize o
    for i in 1:d
        o[i, tx] = zero(T)
    end

    sync_threads()

    # the inner loop is serial
    for j in 1:cld(NK, Bs)
        # load K to shared memory
        K_offset = (j-1) * Bs
        K_idx = K_offset + tx

        if K_idx <= NK
            for m in 1:d
                k[m, tx] = K[m, K_idx, blockIdx().y, blockIdx().z]
            end
        end
        sync_threads()

        # initialize m̃ᵢⱼ
        m̃ᵢⱼ = -Inf

        # compute s
        for n in 1:Bs
            if K_offset + n <= NK && idx <= NQ
                tmp = zero(T)
                for m in 1:d
                    tmp += k[m, n] * q[m, tx]
                end
                s[n, tx] = tmp
            else
                s[n, tx] = -Inf
            end
            m̃ᵢⱼ = max(m̃ᵢⱼ, s[n, tx])
        end

        sync_threads()
#=
       # m̃ᵢⱼ = maximum(s; dims=1)
        l̃ᵢⱼ = zero(T)
        for m in 1:Bs
            P̃ᵢⱼ[m, tx] = exp(s[m, tx] - m̃ᵢⱼ)
            l̃ᵢⱼ += P̃ᵢⱼ[m, tx]
        end

        mᵢⁿᵉʷ = max(mᵢ[tx], m̃ᵢⱼ)
        lᵢⁿᵉʷ = exp(mᵢ[tx]-mᵢⁿᵉʷ) * lᵢ[tx] + exp(m̃ᵢⱼ-mᵢⁿᵉʷ)*l̃ᵢⱼ
=#
        # Load V to shared memory, which is same as K
        if K_idx <= NK
            for m in 1:d
                k[m, tx] = V[m, K_idx, blockIdx().y, blockIdx().z]
            end
        end

        sync_threads()

        if blockIdx().y == 1 && blockIdx().z == 1
            @cushow j
            @cushow k[1, tx]
            @cushow k[2, tx]
        end

      #  w1 = lᵢ[tx] * exp(mᵢ[tx]-mᵢⁿᵉʷ)
     #   w2 = exp(m̃ᵢⱼ-mᵢⁿᵉʷ)

        # compute V * P
        for m in 1:d
            tmp = zero(T)
            for n in 1:Bs
                tmp += k[m, n] * s[n, tx]#P̃ᵢⱼ[n, tx]
            end
            o[m, tx] = tmp#(w1 * o[m,tx] + w2 * tmp) / lᵢⁿᵉʷ
        end

       # lᵢ[tx] = lᵢⁿᵉʷ
       # mᵢ[tx] = mᵢⁿᵉʷ
        sync_threads()
    end

    # write to O
    if idx <= NQ
        for m in 1:d
            O[m, idx, blockIdx().y, blockIdx().z] = o[m, tx]
        end
    end

    return nothing
end


function flash_attention(Q::CuArray{T,4},K::CuArray{T,4},V::CuArray{T,4}) where {T}
    d, N, H, B = size(Q)
    l = similar(Q, N, H, B)
    m = similar(l)
    Br =s
end

# sanity check
function ref_attention(Q,K,V)
    KQ = batched_mul(permutedims(K, (2,1,3,4)), Q)
    #P = softmax(KQ; dims=1)
    P = KQ
    O = batched_mul(V, P)
    return O
end
