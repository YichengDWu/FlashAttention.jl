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
    q = CuDynamicSharedArray(T, (d + 2, Bs), sram_offset) # add 2 rows to avoid bank conflicts
    sram_offset += sizeof(q)
    o = CuDynamicSharedArray(T, (d + 2, Bs), sram_offset) # add 2 rows to avoid bank conflicts
    sram_offset += sizeof(o)
    k = CuDynamicSharedArray(T, (d + 2, Bs), sram_offset) # add 2 rows to avoid bank conflicts
    sram_offset += sizeof(k)
    s = CuDynamicSharedArray(T, (Bs, Bs), sram_offset)

    #  here we reshape Q to (d * Bs, gridDim().x, :, :)
    col_offset = d * Bs * (blockIdx().x - 1)
    Q_offset = col_offset +
               stride(Q, 3) * (blockIdx().y - 1) +
               stride(Q, 4) * (blockIdx().z - 1)
    # load Q to shared memory, note that this is done only once
    Q_max_idx = d * size(Q, 2) - col_offset
    for m in 0:(d - 1)
        Q_idx = m * Bs + tx
        q_idx = Q_idx + div(Q_idx - 1, d) << 1
        if Q_idx <= Q_max_idx
            @inbounds q[q_idx] = Q[Q_idx + Q_offset]
        end
        @inbounds o[q_idx] = zero(T) # initialize o to zero
    end
    sync_threads()

    # initialize lᵢ and mᵢ
    lᵢ = zero(T)
    mᵢ = -typemax(T)

    unrolled_end_N = 4 * div(d, 4)
    # the inner loop is serial
    for j in 1:cld(NK, Bs)
        # load K to shared memory
        j_offset = (j - 1) * Bs
        row_offset = j_offset * d
        K_offset = row_offset +
                   stride(K, 3) * (blockIdx().y - 1) +
                   stride(K, 4) * (blockIdx().z - 1)

        K_max_idx = d * size(K, 2) - row_offset
        for m in 0:(d - 1)
            K_idx = m * Bs + tx
            k_idx = K_idx + div(K_idx - 1, d) << 1
            if K_idx <= K_max_idx
                @inbounds k[k_idx] = K[K_idx + K_offset]
            end
        end
        sync_threads()

        # initialize m̃ᵢⱼ
        m̃ᵢⱼ = -typemax(T)

        # compute s
        for n in 1:Bs
            if j_offset + n <= NK && col <= NQ
                tmp = zero(T)
                for m in 1:4:unrolled_end_N
                    @inbounds tmp = muladd(k[m, n], q[m, tx], tmp)
                    @inbounds tmp = muladd(k[m + 1, n], q[m + 1, tx], tmp)
                    @inbounds tmp = muladd(k[m + 2, n], q[m + 2, tx], tmp)
                    @inbounds tmp = muladd(k[m + 3, n], q[m + 3, tx], tmp)
                end

                for m in (unrolled_end_N + 1):d
                    @inbounds tmp = muladd(k[m, n], q[m, tx], tmp)
                end
                @inbounds s[n, tx] = tmp
                @inbounds m̃ᵢⱼ = max(m̃ᵢⱼ, s[n, tx])
            else
                @inbounds s[n, tx] = -typemax(T)
            end
        end

        # compute P̃ᵢⱼ and l̃ᵢⱼ
        l̃ᵢⱼ = zero(T)
        for n in 1:Bs
            if n + j_offset <= NK
                @inbounds tmp = exp(s[n, tx] - m̃ᵢⱼ)
                @inbounds s[n, tx] = tmp
                l̃ᵢⱼ += tmp
            end
        end

        mᵢⁿᵉʷ = max(mᵢ, m̃ᵢⱼ)
        lᵢⁿᵉʷ = muladd(exp(mᵢ - mᵢⁿᵉʷ), lᵢ, exp(m̃ᵢⱼ - mᵢⁿᵉʷ) * l̃ᵢⱼ)

        # Load V to shared memory, which is same as K
        for m in 0:(d - 1)
            V_idx = m * Bs + tx
            v_idx = V_idx + div(V_idx - 1, d) << 1
            if V_idx <= K_max_idx
                @inbounds k[v_idx] = V[V_idx + K_offset]
            else
                @inbounds k[V_idx] = zero(T)
            end
        end
        sync_threads()

        w₁ = lᵢ * exp(mᵢ - mᵢⁿᵉʷ) / lᵢⁿᵉʷ
        w₂ = exp(m̃ᵢⱼ - mᵢⁿᵉʷ) / lᵢⁿᵉʷ

        # update o
        for m in 1:d
            if col <= NQ
                tmp = zero(T)
                for n in 1:Bs
                    @inbounds tmp = muladd(k[m, n], s[n, tx], tmp)
                end
                @inbounds o[m, tx] = muladd(w₁, o[m, tx], w₂ * tmp)
            end
        end

        lᵢ = lᵢⁿᵉʷ
        mᵢ = mᵢⁿᵉʷ

        sync_threads()
    end

    # write to O
    for m in 0:(d - 1)
        O_idx = m * Bs + tx
        o_idx = O_idx + div(O_idx - 1, d) << 1
        if O_idx <= Q_max_idx
            @inbounds O[O_idx + Q_offset] = o[o_idx]
        end
    end
    return nothing
end

function flash_attention(Q::CuArray{T, 4}, K::CuArray{T, 4}, V::CuArray{T, 4}) where {T}
    _checkbounds(Q, K, V)
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
