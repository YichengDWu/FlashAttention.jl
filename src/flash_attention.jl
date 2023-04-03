function flash_attention_kernel(Q, K, V, O)
    d = size(K, 1)
    power = trailing_zeros(d)
    tx = threadIdx().x
    Bs = blockDim().x # assume Br == Bc
    col = (blockIdx().x - 1) * Bs + tx

    # acllocate shared memory
    T = eltype(Q)
    shmem_offset = 0
    q = CuDynamicSharedArray(T, (Bs + 2, d), shmem_offset) # pad 2 rows to avoid bank conflicts
    shmem_offset += sizeof(q)
    o = CuDynamicSharedArray(T, (Bs + 2, d), shmem_offset) # pad 2 row to avoid bank conflicts
    shmem_offset += sizeof(o)
    k = CuDynamicSharedArray(T, (d, Bs), shmem_offset) # pad 2 rows to avoid bank conflicts
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
        col = (idx - row) >> power + 1
        @inbounds q[col, row] = Q[idx + Q_offset]
        @inbounds o[idx] = zero(T)
        @inbounds k[idx] = K[idx + K_offset]
    end

    sync_threads()

    # initialize lseᵢ and mᵢ
    lseᵢ = -typemax(T)
    mᵢ = -typemax(T)

    # the inner loop is serial
    for _ in 1:cld(size(K, 2), Bs)
        # initialize mᵢⱼ
        mᵢⱼ = lseᵢ

        # compute s=Q^TK
        for n in 1:Bs
            tmp = zero(T)
            for m in 1:d
                @inbounds tmp = CUDA.fma(q[tx, m], k[m, n], tmp)
            end
            s[tx, n] = tmp
            @inbounds mᵢⱼ = max(mᵢⱼ, s[tx, n])
        end

        sync_threads()

        # compute P̃ᵢⱼ and lᵢⱼ
        lᵢⱼ = zero(T)
        for n in 1:Bs
            @inbounds tmp = exp(s[tx, n] - mᵢⱼ)
            @inbounds s[tx, n] = tmp
            lᵢⱼ += tmp
        end

        # Load V to shared memory, which shares the same memory with k
        for i in 0:(d - 1)
            idx = i * Bs + tx
            row = mod1_pow2(idx, d)
            col = (idx - row) >> power + 1
            @inbounds k[row, col] = V[idx + K_offset]
        end

        sync_threads()

        # update o
        for m in 1:d
            tmp = o[tx, m] * exp(mᵢ - mᵢⱼ)
            for n in 1:Bs
                @inbounds tmp = CUDA.fma(s[tx, n], k[m, n], tmp) # k[m, n] * s[n, tx]
            end
            @inbounds o[tx, m] = tmp
        end

        mᵢ = mᵢⱼ
        lseᵢ = mᵢⱼ + log(exp(lseᵢ - mᵢⱼ) + lᵢⱼ)

        K_offset += Bs * d

        # update k
        for i in 0:(d - 1)
            idx = i * Bs + tx
            @inbounds k[idx] = K[idx + K_offset]
        end
        sync_threads()
    end

    for m in 1:d
        @inbounds o[tx, m] = o[tx, m] * exp(mᵢ - lseᵢ)
    end
    sync_threads()

    # write to O
    for i in 0:(d - 1)
        idx = i * Bs + tx
        row = mod1_pow2(idx, d)
        col = (idx - row) >> power + 1
        @inbounds O[idx + Q_offset] = o[col, row]
    end

    return nothing
end

function flash_attention(Q::CuArray{T, 4}, K::CuArray{T, 4}, V::CuArray{T, 4}) where {T}
    _checkbounds(Q, K, V)
    O = similar(Q)
    kernel = @cuda launch=false flash_attention_kernel(Q, K, V, O)
    d, N, H, B = size(Q)
    get_shmem = Base.Fix1(compute_shmem_size, d)
    config = launch_configuration(kernel.fun; shmem=get_shmem, max_threads=256)

    Bs = min(N, config.threads)
    threads = (Bs, 1, 1)
    blocks = (cld(N, Bs), H, B)
    shmem = get_shmem(Bs)

    kernel(Q, K, V, O; threads=threads, blocks=blocks, shmem=shmem)
    return O
end


function kernel(q,k,v)

    warpId = (threadIdx().x - 1) >> 5 + 1
    laneId = (threadIdx().x - 1) % 32 + 1

    block_shape_q = (D=64, N=256,H=1)
    warp_shape_q = (D=32, N=64,H=1)
    thread_shape_q = (D=32, N=1,H=1)

    block_tile_q = Tile(block_shape_q)
    warp_tile_q = Tile(warp_shape_q)
    thread_tile_q = Tile(thread_shape_q)

    shmem_q = CuDynamicSharedArray(Float16, (64,256))
    @unroll for warp_tile = parallellise(block_tile_q, warp_tile_q, warpId, 8)
        @unroll for thread_tile = parallellise(warp_tile, Tile(conf.mem_cd_thread), laneId, 32)
            x = Layout.load(conf.global_c_layout, c, translate_base(thread_tile, (M = block_i, N = block_j)))
            x = transf_gl2sh_c(x, thread_tile)
            Layout.store!(conf.shared_c_layout, shmem_c, x, thread_tile)
        end
    end
end
