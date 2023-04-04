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

using GemmKernels
using GemmKernels.Tiling
using GemmKernels.Layout
using GemmKernels.BLAS
using KernelAbstractions.Extras: @unroll
using GemmKernels: LocalArray

function fwd_kernel(q, k, v)
    warpId = (threadIdx().x - 1) >> 5 + 1
    laneId = (threadIdx().x - 1) & 31 + 1

    global_q_layout = BLAS.global_layout(CuArray{Float16}, Val(false))
    shared_q_layout = BLAS.shared_layout_ab(CuArray{Float16}, Val(false))

    # Constants
    block_i = (blockIdx().x - 1) * blockDim().x
    block_j = blockIdx().y - 1

    # Load Q to shared memory
    block_shape_q = (D=64, N=256, H=1)
    warp_shape_q = (D=64, N=4, H=1)
    thread_shape_q = (D=8, N=1, H=1)

    shmem_q = CuDynamicSharedArray(Layout.eltype(shared_q_layout),
                                   Layout.physical_size(shared_q_layout, (D=64, N=256)))

    @unroll for warp_tile in parallellise(Tile(block_shape_q), Tile(warp_shape_q), warpId,
                                          8)
        @unroll for thread_tile in parallellise(warp_tile, Tile(thread_shape_q), laneId, 32)
            x = Layout.load(global_q_layout, q,
                            translate_base(thread_tile, (D=0, N=block_i, H=block_j)))
            #x = transf_gl2sh_c(x, thread_tile)
            Layout.store!(shared_q_layout, shmem_q, x, thread_tile.DN)
        end
    end
    sync_threads()

    # Load K to shared memory
    return nothing
end
