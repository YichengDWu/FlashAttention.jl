function flash_attention_kernel(Q, K, V, O)
    NK = size(K, 2)
    NQ = size(Q, 2)
    d = size(Q, 1)
    tx = threadIdx().x
    Bs = blockDim().x # assume Br == Bc
    idx = (blockIdx().x - 1) * Bs + tx
    T = eltype(Q)

    sram_offset = 0
    q = CuDynamicSharedArray(T, (d, Bs), sram_offset)
    sram_offset += sizeof(q)
    o = CuDynamicSharedArray(T, (d, Bs), sram_offset)
    sram_offset += sizeof(o)
    k = CuDynamicSharedArray(T, (d, Bs), sram_offset)
    sram_offset += sizeof(k)
    s = CuDynamicSharedArray(T, (Bs, Bs), sram_offset)

    # load Q to shared memory, note that this is done only once
    if idx <= NQ
        for i in 1:d
            @inbounds q[i, tx] = Q[i, idx, blockIdx().y, blockIdx().z]
        end
    end

    # initialize lᵢ and mᵢ
    lᵢ = zero(T)
    mᵢ = -Inf16

    # initialize o
    for i in 1:d
        @inbounds o[i, tx] = zero(T)
    end

    # the inner loop is serial
    for j in 1:cld(NK, Bs)
        # load K to shared memory
        K_offset = (j - 1) * Bs
        K_idx = K_offset + tx

        if K_idx <= NK
            for m in 1:d
                @inbounds k[m, tx] = K[m, K_idx, blockIdx().y, blockIdx().z]
            end
        else
            for m in 1:d
                @inbounds k[m, tx] = zero(T)
            end
        end
        sync_threads()

        # initialize m̃ᵢⱼ
        m̃ᵢⱼ = -Inf16

        # compute s
        for n in 1:Bs
            if K_offset + n <= NK && idx <= NQ
                tmp = zero(T)
                for m in 1:d
                    @inbounds tmp = CUDA.fma(k[m, n], q[m, tx], tmp)
                end
                @inbounds s[n, tx] = tmp
            else
                @inbounds s[n, tx] = -Inf16
            end
            @inbounds m̃ᵢⱼ = max(m̃ᵢⱼ, s[n, tx])
        end

        # compute P̃ᵢⱼ and l̃ᵢⱼ
        l̃ᵢⱼ = zero(T)
        for n in 1:Bs
            @inbounds tmp = exp(s[n, tx] - m̃ᵢⱼ)
            @inbounds s[n, tx] = tmp
            l̃ᵢⱼ += tmp
        end

        mᵢⁿᵉʷ = max(mᵢ, m̃ᵢⱼ)
        lᵢⁿᵉʷ = CUDA.fma(exp(mᵢ - mᵢⁿᵉʷ), lᵢ, exp(m̃ᵢⱼ - mᵢⁿᵉʷ) * l̃ᵢⱼ)

        # Load V to shared memory, which is same as K
        if K_idx <= NK
            for m in 1:d
                @inbounds k[m, tx] = V[m, K_idx, blockIdx().y, blockIdx().z]
            end
        else
            for m in 1:d
                @inbounds k[m, tx] = zero(T)
            end
        end

        sync_threads()

        w₁ = lᵢ * exp(mᵢ - mᵢⁿᵉʷ) / lᵢⁿᵉʷ
        w₂ = exp(m̃ᵢⱼ - mᵢⁿᵉʷ) / lᵢⁿᵉʷ

        # compute V * P
        for m in 1:d
            tmp = zero(T)
            for n in 1:Bs
                @inbounds tmp = CUDA.fma(k[m, n], s[n, tx], tmp)
            end
            @inbounds o[m, tx] = CUDA.fma(w₁, o[m, tx], w₂ * tmp)
        end

        lᵢ = lᵢⁿᵉʷ
        mᵢ = mᵢⁿᵉʷ
    end

    # write to O
    if idx <= NQ
        for m in 1:d
            @inbounds O[m, idx, blockIdx().y, blockIdx().z] = o[m, tx]
        end
    end

    return nothing
end

function flash_attention(Q::CuArray{T, 4}, K::CuArray{T, 4}, V::CuArray{T, 4}) where {T}
    O = similar(Q)
    kernel = @cuda launch=false flash_attention_kernel(Q, K, V, O)

    d, N, H, B = size(Q)
    get_shmem(threads) = compute_shmem_size(d, threads, T)
    config = launch_configuration(kernel.fun; shmem=get_shmem)

    Bs = min(N, config.threads)
    threads = (Bs, 1, 1)
    blocks = (cld(N, Bs), H, B)
    shmem = get_shmem(Bs)

    kernel(Q, K, V, O; threads=threads, blocks=blocks, shmem=shmem)
    return O
end
