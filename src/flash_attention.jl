function flash_attention_kernel(Q, K, V, O)
    NK = size(K, 2)
    NQ = size(Q, 2)
    d = size(K, 1)
    tx = threadIdx().x
    Bs = blockDim().x # assume Br == Bc
    col = (blockIdx().x - 1) * Bs + tx

    # acllocate shared memory
    T = eltype(Q)
    sram_offset = 0
    q = CuDynamicSharedArray(T, (d, Bs), sram_offset)
    sram_offset += sizeof(q)
    o = CuDynamicSharedArray(T, (d, Bs), sram_offset)
    sram_offset += sizeof(o)
    k = CuDynamicSharedArray(T, (d, Bs), sram_offset)
    sram_offset += sizeof(k)
    s = CuDynamicSharedArray(T, (Bs, Bs), sram_offset)

    #  here we reshape Q to (d * Bs, gridDim().x, :, :)
    col_offset = d * Bs * (blockIdx().x - 1)
    Q_offset = col_offset + stride(Q, 3) * (blockIdx().y - 1) + stride(Q, 4) * (blockIdx().z - 1)
    # load Q to shared memory, note that this is done only once
    idx = tx
    max_idx = d * size(Q, 2) - col_offset
    for _ in 1:d
        if idx <= max_idx
            @inbounds q[idx] = Q[idx + Q_offset]
        end
        @inbounds o[idx] = zero(T) # initialize o to zero
        idx += Bs
    end
    sync_threads()

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
        row_offset = (j - 1) * Bs * d
        K_offset = row_offset + stride(K, 3) * (blockIdx().y - 1) + stride(K, 4) * (blockIdx().z - 1)
        idx = tx
        max_idx = d * Bs - row_offset
        for _ in 1:d
            if idx <= max_idx
                @inbounds k[idx] = K[idx + K_offset]
            end
            idx += Bs
        end
        sync_threads()




        K_offset = (j - 1) * Bs
        K_idx = K_offset + tx



        # initialize m̃ᵢⱼ
        m̃ᵢⱼ = -Inf16

        # compute s
        for n in 1:Bs
            if K_offset + n <= NK && col <= NQ
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
    if col <= NQ
        for m in 1:d
            @inbounds O[m, col, blockIdx().y, blockIdx().z] = o[m, tx]
        end
    end

    return nothing
end


# Assmue Bs divides NQ and NK
function flash_attention_no_padding_kernel(Q, K, V, O)
    NK = size(K, 2)
    d = size(K, 1)
    tx = threadIdx().x
    Bs = blockDim().x # assume Br == Bc

    # acllocate shared memory
    T = eltype(Q)
    sram_offset = 0
    q = CuDynamicSharedArray(T, (d, Bs), sram_offset)
    sram_offset += sizeof(q)
    o = CuDynamicSharedArray(T, (d, Bs), sram_offset)
    sram_offset += sizeof(o)
    k = CuDynamicSharedArray(T, (d, Bs), sram_offset)
    sram_offset += sizeof(k)
    s = CuDynamicSharedArray(T, (Bs, Bs), sram_offset)

    #  here we reshape Q to (d * Bs, gridDim().x, :, :)
    Q_offset = d * Bs * (blockIdx().x - 1) + stride(Q, 3) * (blockIdx().y - 1) + stride(Q, 4) * (blockIdx().z - 1)
    # load Q to shared memory, note that this is done only once
    for m in 0:d-1
        idx = tx + m * Bs
        @inbounds q[idx] = Q[idx + Q_offset]
        @inbounds o[idx] = zero(T) # initialize o to zero
    end
    sync_threads()

    # initialize lᵢ and mᵢ
    lᵢ = zero(T)
    mᵢ = -Inf16

    # the inner loop along seq_len_k is serial
    for j in 0:cld(NK, Bs)-1
        K_offset = (j - 1) * Bs

        # load K to shared memory
        K_offset =  d * Bs * j + stride(K, 3) * (blockIdx().y - 1) + stride(K, 4) * (blockIdx().z - 1)

        for m in 0:d-1
            idx = tx + m * Bs
            @inbounds k[idx] = K[idx + Q_offset]
        end
        sync_threads()

        # initialize m̃ᵢⱼ
        m̃ᵢⱼ = -Inf16

        # compute s
        for n in 1:Bs
            tmp = zero(T)
            for m in 1:d
                @inbounds tmp = CUDA.fma(k[m, n], q[m, tx], tmp)
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
        for m in 0:d-1
            idx = tx + m * Bs
            @inbounds k[idx] = V[idx + Q_offset]
        end
        sync_threads()

        w₁ = lᵢ * exp(mᵢ - mᵢⁿᵉʷ) / lᵢⁿᵉʷ
        w₂ = exp(m̃ᵢⱼ - mᵢⁿᵉʷ) / lᵢⁿᵉʷ

        # update o
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
    for m in 0:d-1
        idx = tx + m * Bs
        @inbounds O[idx + Q_offset] = o[idx]
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
