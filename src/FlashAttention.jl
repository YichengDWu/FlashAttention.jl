module FlashAttention

using CUDA

include("utils.jl")
include("flash_attention.jl")

export flash_attention

function __init__()
    cap = CUDA.capability(device())
    @info "Detected GPU with compute capability $cap, setting maximum shared memory size..."

    # leave 2 KB free
    if cap.major == 0x07
        if cap.minor == 0x05
            setMaxShmem(62)
        else
            setMaxShmem(94)
        end
    elseif cap.major == 0x08
        if cap.minor == 0x00 || cap.minor == 0x07
            setMaxShmem(162)
        else
            setMaxShmem(96) # 96KB
        end
    elseif cap.major == 0x09
        setMaxShmem(226)
    else
        @warn "Unknown compute capability, the performance may be suboptimal."
    end
end

end
