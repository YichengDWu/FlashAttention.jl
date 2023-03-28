using FlashAttention
using Documenter

DocMeta.setdocmeta!(
    FlashAttention,
    :DocTestSetup,
    :(using FlashAttention);
    recursive = true,
)

makedocs(;
    modules = [FlashAttention],
    authors = "MilkshakeForReal <yicheng.wu@ucalgary.ca> and contributors",
    repo = "https://github.com/YichengDWu/FlashAttention.jl/blob/{commit}{path}#{line}",
    sitename = "FlashAttention.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://YichengDWu.github.io/FlashAttention.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/YichengDWu/FlashAttention.jl", devbranch = "master")
