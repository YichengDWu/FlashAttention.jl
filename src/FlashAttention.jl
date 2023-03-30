module FlashAttention

using CUDA

CUDA.@device_override Base.exp2(x::Float32) = Float16(CUDA.exp2(Float32(x)))

include("utils.jl")
include("flash_attention.jl")

export flash_attention, setMaxShmem

function __init__()
    cap = CUDA.capability(device())

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
