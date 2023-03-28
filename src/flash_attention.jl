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
        for i = 1:d
            q[i, tx] = Q[i, idx, blockIdx().y, blockIdx().z]
        end
    end

    # initialize lᵢ and mᵢ
    lᵢ[tx] = zero(T)
    mᵢ[tx] = -Inf

    # initialize o
    for i = 1:d
        o[i, tx] = zero(T)
    end

    # the inner loop is serial
    for j = 1:cld(NK, Bs)
        # load K to shared memory
        K_offset = (j - 1) * Bs
        K_idx = K_offset + tx

        if K_idx <= NK
            for m = 1:d
                k[m, tx] = K[m, K_idx, blockIdx().y, blockIdx().z]
            end
        else
            for m = 1:d
                k[m, tx] = zero(T)
            end
        end
        sync_threads()

        # initialize m̃ᵢⱼ
        m̃ᵢⱼ = -Inf

        # compute s
        for n = 1:Bs
            if K_offset + n <= NK && idx <= NQ
                tmp = zero(T)
                for m = 1:d
                    tmp += k[m, n] * q[m, tx]
                end
                s[n, tx] = tmp
            else
                s[n, tx] = -Inf
            end
            m̃ᵢⱼ = max(m̃ᵢⱼ, s[n, tx])
        end

        l̃ᵢⱼ = zero(T)
        for n = 1:Bs
            tmp = exp(s[n, tx] - m̃ᵢⱼ)
            P̃ᵢⱼ[n, tx] = tmp
            l̃ᵢⱼ += tmp
        end

        mᵢⁿᵉʷ = max(mᵢ[tx], m̃ᵢⱼ)
        lᵢⁿᵉʷ = exp(mᵢ[tx] - mᵢⁿᵉʷ) * lᵢ[tx] + exp(m̃ᵢⱼ - mᵢⁿᵉʷ) * l̃ᵢⱼ

        # Load V to shared memory, which is same as K
        if K_idx <= NK
            for m = 1:d
                k[m, tx] = V[m, K_idx, blockIdx().y, blockIdx().z]
            end
        else
            for m = 1:d
                k[m, tx] = zero(T)
            end
        end

        sync_threads()

        w₁ = lᵢ[tx] * exp(mᵢ[tx] - mᵢⁿᵉʷ) / lᵢⁿᵉʷ
        w₂ = exp(m̃ᵢⱼ - mᵢⁿᵉʷ) / lᵢⁿᵉʷ

        # compute V * P
        for m = 1:d
            tmp = zero(T)
            for n = 1:Bs
                tmp += k[m, n] * P̃ᵢⱼ[n, tx]
            end
            o[m, tx] = w₁ * o[m, tx] + w₂ * tmp
        end

        lᵢ[tx] = lᵢⁿᵉʷ
        mᵢ[tx] = mᵢⁿᵉʷ
    end

    # write to O
    if idx <= NQ
        for m = 1:d
            O[m, idx, blockIdx().y, blockIdx().z] = o[m, tx]
        end
    end

    return nothing
end


function flash_attention(Q::CuArray{T,4}, K::CuArray{T,4}, V::CuArray{T,4}) where {T}
    O = similar(Q)
    d, N, H, B = size(Q)

    Bs = min(64, N) # block size
    threads = (Bs, 1, 1)
    blocks = (cld(N, Bs), H, B)
    shmem = compute_shmem_size(d, Bs, T)
    @cuda threads = threads blocks = blocks shmem = shmem flash_attention_kernel(Q, K, V, O)
    return O
end

@inline function compute_shmem_size(d, Bs, T)
    return (d * Bs * 3 + Bs * 3 + Bs * Bs * 2) * sizeof(T)
end
