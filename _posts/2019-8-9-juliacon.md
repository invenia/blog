---
layout: post

title: "JuliaCon 2019"

author: "Eric Davies, Fernando Chorney, Glenn Moynihan, Lyndon White, Mary Jo Ramos, Nicole Epp, Sean Lovett, Will Tebbutt"

comments: false

tags: julia programming conference

---

Some of us at Invenia recently attended [JuliaCon 2019](https://juliacon.org/2019/), which was held at the University of Maryland, Baltimore, on July 21-26.

It was the biggest JuliaCon yet, with more than 370 participants, but still had a welcoming atmosphere as in previous years. There were plenty of opportunities to catch up with people that you work with regularly but almost never actually have a face-to-face conversation with.

We sent 16 developers and researchers from our Winnipeg and Cambridge offices.  As far as we know, this was the largest contingent of attendees after Julia Computing! We had a lot of fun and thought it would be nice to share some of our favourite parts.


## Julia is more than scientific computing

For someone who is relatively new to Julia, it may be difficult to know what to expect from a JuliaCon. Consider someone who has used Julia before in a few ways, but who has yet to fully grasp the power of the language.

One of the first things to notice is just how diverse the language is. It may initially seem as a very special tool for the scientific community, but the more one learns about it, the more evident it becomes that it could be used to tackle all sorts of problems.

Workshops were tailored for individuals with a range of Julia knowledge. Beginners had the opportunity to play around with Julia in basic workshops, and more advanced workshops covered topics from differential equations to parallel computing. There was something for everyone and no reason to feel intimidated.

The [Intermediate Julia for Scientific Computing](https://github.com/dpsanders/intermediate_julia_2019) workshop highlighted Julia's multiple dispatch and meta-programming capabilities. Both are very useful and interesting tools. As one learns more about Julia and sees the growth of the community, it becomes easier to understand the reasons why some love the language so much. It has the scientific usage without compromising on speed, readability, or usefulness.


## Julia is welcoming!

The various workshops and talks that encouraged diverse groups to utilize and contribute to the language were also great to see. It is uplifting to witness such a strong initiative to make everyone feel welcome.



The Diversity and Inclusion BOF provided a space for anyone to voice opinions, concerns and suggestions on how to improve the Julia experience for everyone. The [Diversity and Inclusion session](https://www.youtube.com/watch?v=9cedY6zyo8I) showcased how to foster the community's growth. The speakers were educators who shared their experiences using Julia as a tool to enhance education for women, minorities, and students with lower socioeconomic backgrounds. The inspirational work done by these individuals -- and everyone at JuliaCon -- prove that anyone can use Julia and succeed with the community's support.


## The community has a big role to play

[Heather Miller's keynote](https://www.youtube.com/watch?v=b_743P8XuvA) on Scala's experience as an open source project was another highlight. It is staggering to see what the adoption of open source software in industry looks like, as well as learning about the issues that come up with growth, and the importance of the community. Although Heather's snap polls at the end seemed broadly positive about the state of Julia's community, they suggested that the level to which we're mentoring newer members of the community is lower than ideal.


## Birds of a Feather Sessions

This year Birds of a Feather (BoF) sessions were a big thing. In previous JuliaCons, there were only one or two, but this year we had twelve. Each session was unique and interesting, and in almost every case people wished they could continue after the hour was up. The BoFs were an excellent chance to discuss and explore topics that are difficult to do in a formal talk. In many ways this is the best reason to go to a conference: to connect with people, make plans, and learn deeply. Normally this happens in cramped corridors or during coffee breaks, which certainly did happen this year still, but the BoFs gave the whole process just a little bit more structure, and also tables.

Each BoF was different and good in its own ways, and none would have been half as good without getting everyone in the same room. We mentioned the Diversity BoF earlier. The Parallelism BoF ended with everyone breaking into four groups, each of which produced notes on future directions for parallelism in Julia. Our own head of development, Curtis Vogt, ran the “Julia In Production" BoF which showed that both new and established companies are adopting Julia, and led to some useful discussion about better support for Julia in Cloud computing environments.

The Cassette BoF was particularly good. It had everyone talking about what they were using Cassette for. Our own Lyndon White presented his new Cassette-like project, [Arborist](https://github.com/oxinabox/Arborist.jl/), and some of the challenges it faces, including understanding of why the compiler behaves the way it does. We also learned that neither Jarret Revels (the creator of Cassette), nor Valentin Churavy (its current maintainer) know exactly what Tagging in Cassette does.

## Support tools are becoming well established

With the advent of [Julia 1.0](https://julialang.org/blog/2018/08/one-point-zero) at JuliaCon 2018 we saw the release of a new [Pkg.jl](https://www.youtube.com/watch?v=-yUiLCGegJs); a far more robust and user-friendly package manager. However, many core tools to support the maturing package ecosystem were yet to emerge until JuliaCon 2019. This year we saw the introduction of a new debugger, a formatter, and an impressive documentation generator.

For the past year, the absence of a debugger that equalled the pleasure of using Pkg.jl remained an open question. An answer was unveiled in [Debugger.jl](https://github.com/JuliaDebug/Debugger.jl), by the venerable Tim Holy, Sebastian Pfitzner, and Kristoffer Carlsson. They discussed  the ease with which Debugger.jl can be used to declare break points, enter functions, traverse lines of code, and modify environment variables, all from within a new debugger REPL mode! This is a very welcome addition to the Julia toolbox.

Style guides are easy to understand but it's all too easy to misstep in writing code. Dominique Luma [gave a simple walkthrough](https://www.youtube.com/watch?v=12z5MzoFQOM) of his [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) package, which formats the line width of source code according to specified parameters. The formatter spruces up and nests lines of code to present more aesthetically pleasing text. Only recently registered as a package, and not a comprehensive linter, it is still a good step in the right direction, and one that will save countless hours of code review for such a simple package.

Code is only as useful as its documentation and [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) is the canonical tool for generating package documentation. Morten Piibeleht gave an excellent overview of the API including docstring generation, example code evaluation, and using custom CSS for online manuals. A great feature is the inclusion of doc testing as part of a unit testing set up to ensure that examples match function outputs. While Documenter has had doctests for a long time, they are now much easier to trigger: just add `using Documenter, MyPackage; doctest(MyPackage)` to your `runtests.jl` file. Coupled with Invenia's own [PkgTemplates.jl](https://github.com/invenia/PkgTemplates.jl), creating a maintainable package framework has never been easier.


## Probabilistic Programming: Julia sure does do a lot of it

Another highlight was the somewhat unexpected [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming) track on Thursday afternoon. There were 5 presentations, each on a different framework and take on what probabilistic programming in Julia can look like. These included a [talk](https://www.youtube.com/watch?v=OO3BBkGEMV8) on [Stheno.jl](https://github.com/willtebbutt/Stheno.jl) from our own Will Tebbutt, which also contained a great introduction to [Gaussian Processes](https://en.wikipedia.org/wiki/Gaussian_process).

Particularly interesting was the [Gen](https://github.com/probcomp/Gen) for data cleaning [talk](https://www.youtube.com/watch?v=vUxrtqY84AM) by Alex Law. This puts the problem of data cleaning -- correcting miss-entered, or incomplete data -- into a probabilistic programming setting. Normally this is done via deterministic heuristics, for example by correcting spelling by using the [Damerau–Levenshtein distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance) to the nearest word in a dictionary. However, such approaches can have issues, for instance, when correcting town names, the nearest by spelling may be a tiny town with very small population, which is probably wrong. More complicated heuristics can be written to handle such cases, but they can quickly become unwieldy. An alternative to heuristics is to write statements about the data in a probabilistic programming language. For example, there is a chance of one typo, and a smaller chance of two typos, and further that towns with higher population are more likely to occur. Inference can then be run in the probabilistic model for the most likely cleaned field values. This is a neat idea based on a few recent publications. We’re very excited about all of this work, and look forward to further discussions with the authors of the various frameworks.



## Composable Parallelism

A particularly [exciting announcement](https://julialang.org/blog/2019/07/multithreading?source=techstories.org) was [composable multi-threaded parallelism](https://www.youtube.com/watch?v=HfiRnfKxI64) for Julia v1.3.0-alpha.

Safe and efficient thread parallelism has been on the horizon for a while now.  Previously, multi-thread parallelism was in an experimental form and generally very limited (see the `Base.Threads.@threads` macro). This involved dividing the work into blocks that execute independently and then joined into Julia's main thread. Limitations included not being able to use I/O or to do task switching inside the threaded for-loop of parallel work.

All that is changing in Julia v1.3. One of the most exciting changes is that all system-level I/O operations are now thread-safe. Functions like `print` can be used during a `@spawn` or `@threads` macro for-loop call. Additionally, the `@spawn` construct (which is like `@async`, but parallel) was introduced; this has a threaded meaning, moving towards parallelism rather than the pre-existing Distributed standard library export.

Taking advantage of hardware parallel computing capacities can lead to very large speedups for certain workflows, and now it will be much easier to get started with threads. The usual multithreading pitfalls of race conditions and potential deadlocks still exist, but these can generally be worked around with locks and atomic operations where needed. There are many other features and improvements to look forward to.



## Neural differential equations: machine learning and physics join forces

We were excited to see [DiffEqFlux.jl](https://github.com/JuliaDiffEq/DiffEqFlux.jl) presented at JuliaCon this year. This combines the excellent [DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl) and [Flux.jl](https://github.com/FluxML/Flux.jl) packages to implement Neural ODEs, SDEs, PDEs, DDEs, etc. All using state-of-the-art [time-integration methods](https://julialang.org/blog/2019/01/fluxdiffeq).

The [Neural Ordinary Differential Equations paper](https://arxiv.org/abs/1806.07366) caused a stir at [NeurIPS 2018](https://neurips.cc/Conferences/2018) for its successful combination of two seemingly-distinct techniques: differential equations and neural networks. More generally there has been a recent resurgence of interest in combining modern machine learning with traditional scientific modelling techniques (see [Hidden Physics Models](https://arxiv.org/abs/1708.00588), among others). There are many possible applications for these methods: they have potential for both enriching black-box machine learning models with physical insights, and conversely using data to learn unknown terms in structured physical models. For example, it is possible to use a differential equations solver as a layer embedded within a neural network, and train the resulting model end-to-end. In the latter case, it is possible to replace a term in a differential equation by a neural network. Both these use-cases, and many others, are now possible in DiffEqFlux.jl with just a few lines of code. The generic programming capabilities of Julia really shine through in the flexibility and composability of these tools, and we’re excited to see where this field will go next.



## The compiler has come a long way!

The Julia compiler and standard library has come a long way in the last two years. An interesting case came up while Eric Davies was working on [IndexedDims.jl](https://github.com/invenia/IndexedDims.jl) at the hackathon.

Take for example the highly-optimized code from [base/multidimensional.jl](https://github.com/JuliaLang/julia/blob/release-1.2/base/multidimensional.jl#L689):
​
```julia
@generated function _unsafe_getindex!(dest::AbstractArray, src::AbstractArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
   quote
       Base.@_inline_meta
       D = eachindex(dest)
       Dy = iterate(D)
       @inbounds Base.Cartesian.@nloops $N j d->I[d] begin
           # This condition is never hit, but at the moment
           # the optimizer is not clever enough to split the union without it
           Dy === nothing && return dest
           (idx, state) = Dy
           dest[idx] = Base.Cartesian.@ncall $N getindex src j
           Dy = iterate(D, state)
       end
       return dest
   end
end
```
It is interesting that this now has speed within a factor of two of the following simplified version:

```julia
function _unsafe_getindex_modern!(dest::AbstractArray, src::AbstractArray, I::Vararg{Union{Real, AbstractArray}, N}) where N
   @inbounds for (i, j) in zip(eachindex(dest), Iterators.product(I...))
       dest[i] = src[j...]
   end
   return dest
end
```
In Julia 0.6, this was five times slower than the highly-optimized version above!

It turns out that the type parameter `N` matters a lot. Removing this explicit type parameter causes performance to degrade by an order of magnitude. While Julia generally specializes on the types of the arguments passed to a method, there are a few cases in which Julia will avoid that specialization unless an explicit type parameter is added: the three main cases are `Function`, `Type`, and as in the example above, `Vararg` arguments.



## Some of our other favourite talks

Here are some of our other favourite talks not discussed above:
1. [Heterogeneous Agent DSGE Models](https://www.youtube.com/watch?v=Et-5AncK8TU)
2. [Solving Cryptic Crosswords](https://www.youtube.com/watch?v=SVsA55h8pdg)
3. [Differentiate All The Things!](https://www.youtube.com/watch?v=SVsA55h8pdg)
4. [Building a Debugger with Cassette](https://www.youtube.com/watch?v=SVsA55h8pdg)
5. [FilePaths](https://www.youtube.com/watch?v=SVsA55h8pdg)
6. [Ultimate Datetime](https://www.youtube.com/watch?v=HHjvv5yWMmc)
7. [Smart House with JuliaBerry](https://www.youtube.com/watch?v=1_S3aU3P3rE)
8. [Why writing C interfaces in Julia is so easy](https://www.youtube.com/watch?v=ez-KVi0leOw)
9. [Open Source Power System Modeling](https://www.youtube.com/watch?v=1TipY6g9IzE)
10. [What's Bad About Julia](https://www.youtube.com/watch?v=TPuJsgyu87U)
11. [The Unreasonable Effectiveness of Multiple Dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY)

It is always impressive how much is covered every JuliaCon, considering its size. It serves to show both how fast the community is growing and how versatile the language is. We look forward for another one in 2020, even bigger and broader.
