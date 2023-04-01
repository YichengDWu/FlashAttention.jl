function flash_attention_kernel(Q, K, V, O)
    d = size(K, 1)
    tx = threadIdx().x
    Bs = blockDim().x # assume Br == Bc
    col = (blockIdx().x - 1) * Bs + tx

    # acllocate shared memory
    T = eltype(Q)
    shmem_offset = 0
    q = CuDynamicSharedArray(T, (Bs + 2, d), shmem_offset) # pad 2 rows to avoid bank conflicts
    shmem_offset += sizeof(q)
    o = CuDynamicSharedArray(T, (Bs + 2, d), shmem_offset) # pad 2 rows to avoid bank conflicts
    shmem_offset += sizeof(o)
    k = CuDynamicSharedArray(T, (d, Bs), shmem_offset) # add 2 rows to avoid bank conflicts
    shmem_offset += sizeof(k)
    s = CuDynamicSharedArray(T, (Bs, Bs), shmem_offset)

    # load Q to shared memory, note that this is done only once
    Q_offset = d * Bs * (blockIdx().x - 1) +
               stride(Q, 3) * (blockIdx().y - 1) +
               stride(Q, 4) * (blockIdx().z - 1)
    K_offset = stride(K, 3) * (blockIdx().y - 1) + stride(K, 4) * (blockIdx().z - 1)
    for i in 0:(d - 1)
        idx = i * Bs + tx
        row = mod1_pow2(idx, d)
        col = div(idx - row, d) + 1
        @inbounds q[col, row] = Q[idx + Q_offset]
        @inbounds o[idx] = zero(T)
        @inbounds k[idx] = K[idx + K_offset]
    end

    sync_threads()

    # initialize lᵢ and mᵢ
    lᵢ = zero(T)
    mᵢ = -typemax(T)

    # the inner loop is serial
    for _ in 1:cld(size(K, 2), Bs)
        # initialize m̃ᵢⱼ
        m̃ᵢⱼ = -typemax(T)

        # compute s=Q^TK
        for n in 1:Bs
            tmp = zero(T)
            for m in 1:d
                @inbounds tmp = CUDA.fma(q[tx, m], k[m, n], tmp)
            end
            s[tx, n] = tmp
            @inbounds m̃ᵢⱼ = max(m̃ᵢⱼ, s[tx, n])
        end

        sync_threads()

        # compute P̃ᵢⱼ and l̃ᵢⱼ
        l̃ᵢⱼ = zero(T)
        for n in 1:Bs
            @inbounds tmp = exp(s[tx, n] - m̃ᵢⱼ)
            @inbounds s[tx, n] = tmp
            l̃ᵢⱼ += tmp
        end

        mᵢⁿᵉʷ = max(mᵢ, m̃ᵢⱼ)
        lᵢⁿᵉʷ = CUDA.fma(exp(mᵢ - mᵢⁿᵉʷ), lᵢ, exp(m̃ᵢⱼ - mᵢⁿᵉʷ) * l̃ᵢⱼ)

        # Load V to shared memory, which shares the same memory with k
        for i in 0:(d - 1)
            idx = i * Bs + tx
            row = mod1_pow2(idx, d)
            col = div(idx - row, d) + 1
            @inbounds k[row, col] = V[idx + K_offset]
        end

        sync_threads()

        w₁ = lᵢ * exp(mᵢ - mᵢⁿᵉʷ) / lᵢⁿᵉʷ
        w₂ = exp(m̃ᵢⱼ - mᵢⁿᵉʷ) / lᵢⁿᵉʷ

        # update o
        for m in 1:d
            tmp = zero(T)
            for n in 1:Bs
                @inbounds tmp = CUDA.fma(s[tx, n], k[m, n], tmp) # k[m, n] * s[n, tx]
            end
            @inbounds o[tx, m] = CUDA.fma(w₁, o[tx, m], w₂ * tmp) #  w₁ * o[m, tx] + w₂ * tmp
        end

        lᵢ = lᵢⁿᵉʷ
        mᵢ = mᵢⁿᵉʷ

        K_offset += Bs * d

        # update k
        for i in 0:(d - 1)
            idx = i * Bs + tx
            @inbounds k[idx] = K[idx + K_offset]
        end
        sync_threads()
    end

    # write to O
    for i in 0:(d - 1)
        idx = i * Bs + tx
        row = mod1_pow2(idx, d)
        col = div(idx - row, d) + 1
        @inbounds O[idx + Q_offset] = o[col, row]
    end

    return nothing
end

function flash_attention(Q::CuArray{T, 4}, K::CuArray{T, 4}, V::CuArray{T, 4}) where {T}
    _checkbounds(Q, K, V)
    O = similar(Q)
    kernel = @cuda launch=false flash_attention_kernel(Q, K, V, O)
    d, N, H, B = size(Q)
    get_shmem(threads) = compute_shmem_size(d, threads, T)
    config = launch_configuration(kernel.fun; shmem=get_shmem, max_threads=256)

    Bs = min(N, config.threads)
    threads = (Bs, 1, 1)
    blocks = (cld(N, Bs), H, B)
    shmem = get_shmem(Bs)

    kernel(Q, K, V, O; threads=threads, blocks=blocks, shmem=shmem)
    return O
end
