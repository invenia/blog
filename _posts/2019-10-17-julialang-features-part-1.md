---
layout: post
title: "The Emergent Features of JuliaLang: Part I"
author: Lyndon White
tags:
    - julia
---


## Introduction

*This blog post is based on a talk originally given at a Cambridge PyData Meetup, and also at a London Julia Users Meetup.*

Julia is known to be a very expressive language with many interesting features.
It is worth considering that some features were not planned,
but simply emerged from combinations of other features.
This post will describe how several interesting features are implemented:

1. Unit syntactic sugar,
2. Pseudo-OO objects with public/private methods,
3. Dynamic Source Tranformation / Custom Compiler Passes (Cassette),
4. Traits.

Some of these (1 and 4) should be used when appropriate,
while others (2 and 3) are rarely appropriate, but are instructive.
Part II of this post is about traits
because they are one of the most powerful and interesting Julia features.
They emerge from types, multiple dispatch,
and the ability to dispatch on the types themselves (rather than just instances).
We will review both the common use of traits on types,
and also the very interesting---but less common---use of traits on functions.

There are many other emergent features which are not discussed in this post.
For example, creating a vector using `Int[]` is an overload of
`getindex`, and constructors are overloads of `::Type{<:T}()`.

In Part I of this post we will cover topics 1-3, and in Part II, we will cover Traits.


## Juxtaposition Multiplication: Convenient Syntax for Units

Using units in certain kinds of calculations can be very helpful for protecting against arithmetic mistakes.
The package [Unitful](https://github.com/ajkeller34/Unitful.jl) provides the ability
to do calculations using units in a very natural way, and makes it easy to write
things like "2 meters" as `2m`. Here is an example of using Unitful units:

```julia
julia> using Unitful.DefaultSymbols

julia> 1m * 2m
2 m^2

julia> 10kg * 15m / 1s^2
150.0 kg m s^-2

julia> 150N == 10kg * 15m / 1s^2
true
```

How does this work? The answer is *Juxtaposition Multiplication*---a literal number
placed before an expression results in multiplication, for example:

```julia
julia> x = 0.5π
1.5707963267948966

julia> 2x
3.141592653589793

julia> 2sin(x)
2.0
```

To make this work, we need to overload multiplication to invoke the unit type's constructor.
Below is a simplified version of what goes on under the hood of Unitful.jl:

```julia
abstract type Unit<:Real end
struct Meter{T} <: Unit
    val::T
end

Base.:*(x::Any, U::Type{<:Unit}) = U(x)
```

Here we are overloading multiplication with a unit subtype, not an
instance of a unit subtype (`Meter(2)`), but with the subtype itself (`Meter`).
This is what `::Type{<:Unit}` means. We can see this if we write:

```julia
julia> 5Meter
Meter{Int64}(5)
```
This shows that we create a `Meter` object with `val=5`.

To get to a full units system, we then need to define methods for our unit types for everything that numbers need to work with,
such as addition and multiplication. The final result is units-style syntactic sugar.


## Closures give us "Classic Object-Oriented Programming"

First, it is important to emphasize: **don't do this in real Julia code**.
It is unidiomatic, and likely to hit edge-cases the compiler doesn't optimize well
(see for example the [infamous closure boxing bug](https://github.com/JuliaLang/julia/issues/15276)).
This is all the more important because it often requires boxing (see below).

"Classic OO" has classes with member functions (methods) that can see all the fields and methods,
but that outside the methods of the class, only public fields and methods can be seen.
This idea was originally posted for Julia in one of Jeff Bezanson's rare
[Stack Overflow posts](https://stackoverflow.com/a/39150509/179081),
which uses a classic functional programming trick.

Let's use closures to define a Duck type as follows:

```julia
function newDuck(name)
    # Declare fields:
    age=0

    # Declare methods:
    get_age() = age
    inc_age() = age+=1

    quack() = println("Quack!")
    speak() = quack()

    #Declare public:
    ()->(get_age, inc_age, speak, name)
end
```

This implements various things we would expect from "classic OO" encapsulation.
We can construct an object and call public methods:

```julia
julia> 🦆 = newDuck("Bill")
#7 (generic function with 1 method)

julia> 🦆.get_age()
0
```

Public methods can change private fields:

```julia
julia> 🦆.inc_age()
1

julia> 🦆.get_age()
1
```


However, outside the class we can't access private fields:

```julia
julia> 🦆.age
ERROR: type ##7#12 has no field age
```

We also can't access private methods:

```julia
julia> 🦆.quack()
ERROR: type ##7#12 has no field quack
```

However, accessing their functionality via public methods works:

```julia
julia> 🦆.speak()
Quack!
```

### How does this work?

Closures return singleton objects, with directly-referenced, closed variables as fields.
All the public fields/methods are directly referenced,
but the private fields (e.g `age`, `quack`) are not---they are closed over other methods that use them.
We can see those private methods and fields via accessing the public method closures:

```julia
julia> 🦆.speak.quack
(::var"#quack#10") (generic function with 1 method)

julia> 🦆.speak.quack()
Quack!
```

```julia
julia> 🦆.inc_age.age
Core.Box(1)
```

### An Aside on Boxing

The `Box` type is similar to the `Ref` type in function and purpose.
`Box` is the type Julia uses for variables that are closed over, but which might be rebound to a new value.
This is the case for primitives (like `Int`) which are rebound whenever they are incremented.
It is important to be clear on the difference between mutating the contents of a variable,
and rebinding that variable name.

```julia
julia> x = [1, 2, 3, 4];

julia> objectid(x)
0x79eedc509237c203

julia> x .= [10, 20, 30, 40];  # mutating contents

julia> objectid(x)
0x79eedc509237c203

julia> x = [100, 200, 300, 400];  # rebinding the variable name

julia> objectid(x)
0xfa2c022285c148ed
```

In closures, boxing applies only to rebinding, though the [closure bug](https://github.com/JuliaLang/julia/issues/15276) means Julia will sometimes over-eagerly box variables because it believes that they might be rebound.
This has no bearing on what the code does, but it does impact performance.

While this kind of code itself should never be used (since Julia has a perfectly
functional system for dispatch and works well without "Classic OO"-style encapsulation),
knowing how closures work opens other opportunities to see how they can be used.
In our [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/) project,
we are considering the use of closures as callable named tuples as part of
a [difficult problem](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/53#issuecomment-533833058)
in extensibility and defaults.


## Cassette, etc.

<img src="https://raw.githubusercontent.com/jrevels/Cassette.jl/master/docs/img/cassette-logo.png" width="256" style="display: inline"/>

Cassette/IRTools (Zygote) are built around a notable Julia feature, which goes by several names:
Custom Compiler Passes, Contextual Dispatch, Dynamically-scoped Macros or Dynamic Source Rewriting.
This is an incredibly powerful and general feature, and was the result of a very specific
[Issue #21146](https://github.com/JuliaLang/julia/issues/21146) and very casual
[PR #22440](https://github.com/JuliaLang/julia/pull/22440) suggestion that it might be useful for one particular case. These describe everything about Cassette, but only in passing.

The Custom Compiler Pass feature is essential for:
 - AutoDiff tools (ForwardDiff2, Zygote, Yota),
 - Mocking tools (Mocking.jl),
 - Debuggers (MagneticReadHead),
 - Code-proof related tools (ConcolicFuzzer),
 - Generally rewriting all the code (GPUifyLoops).

### Call Overloading

One of the most basic elements is the ability to effectively overload what it means to call a function.
Call overloading is much more general than operator overloading, as it applies to every call special-cased as appropriate, whereas operator overloading applies to just one call and just one set of types.

To give a concrete example of how call overloading is more general,
operator overloading/dispatch (multiple or otherwise) would allow us to
to overload, for example, `sin(::T)` for different types `T`.
This way `sin(::DualNumber)` could be specialized to be different from `sin(::Float64)`,
so that with `DualNumber` as input it could be set to calculate the nonreal part using the `cos` (which is the  derivative of sin). This is exactly how
[DualNumbers](https://en.wikipedia.org/wiki/Dual_number#Differentiation) operate.
However, operator overloading can't express the notion that, for all functions `f`, if the input type is `DualNumber`, that `f` should calculate the dual part using the deriviative of `f`.
Call overloading allows for much more expressivity and massively simplifies the implementation of automatic differentiation.

Let's look at an example with regular functions:

```julia
function merge(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    names = merge_names(an, bn)
    types = merge_types(names, typeof(a), typeof(b))
    NamedTuple{names,types}(map(n->getfield(sym_in(n, bn) ? b : a, n), names))
end
```

This checks at runtime what fields each `NamedTuple` has, to decide what will be in the merge.
However, note that we know all this information based on the types alone.

### Generated Functions

[Generated functions](https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions-1)
take types as inputs and return the AST (Abstract Syntax Tree) for what code should run, based on information in the types. This is a kind of metaprogramming.
Take for example a function `f` with input an `N`-dimentional array (type `AbstractArray{T, N}`). Then a generated function for `f` might construct code with `N` nested loops to process each element.
It is then possible to generate code that only accesses the fields we want, which gives substantial performance improvements.

```julia
@generated function merge(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    names = merge_names(an, bn)
    types = merge_types(names, a, b)
    vals = Any[ :(getfield($(sym_in(n, bn) ? :b : :a), $(QuoteNode(n)))) for n in names ]
    :( NamedTuple{$names,$types}(($(vals...),)) )
end
```

It is important to note that Casette is not baked into the compiler.
`@generated` functions can return an `Expr` **or** a `CodeInfo`.
We return a `CodeInfo` based on a modified version of one for a function argument.
We can use `@code_lowered` to look up what the original `CondeInfo` would have been.
`@code_lowered` gives back one particular representation of the Julia code: the **Untyped IR**.

### Layers of Representation

Julia has many representations of the code it moves through during compilation.
 - Source code (`CodeTracking.definition(String,....)`),
 - AST: (`CodeTracking.definition(Expr, ...`),
 - Untyped IR: `@code_lowered`,
 - Typed IR: `@code_typed`,
 - LLVM: `@code_llvm`,
 - ASM: `@code_native`.
We can retrieve the different representations using the functions/macros indicated.

Looking at Untyped IR, this is basically a linearization of the AST, with the following properties:
 - Only 1 operation per statement (nested expressions get broken up);
 - the return values for each statement are accessed as `SSAValue(index)`;
 - variables become Slots;
 - control-flow becomes jumps (like Goto);
 - function names become qualified as `GlobalRef(mod, func)`.
It is ok to read, but can be very difficult to work with or write.
IRTools and Cassette exist to make this easier, but to properly understand how it works, lets run through a manual example (originally from a [JuliaCon talk on MagneticReadHead.jl](https://www.youtube.com/watch?v=lTR6IPjDPlo)).

Let's define a generated function `rewritten`, that makes a copy of the Untyped IR (a `CodeInfo` object that it gets back from `@code_lowered`) and then mutates it, replacing each call with a call to the function `call_and_print`. Finally, this returns the new `CodeInfo` to be run when it is called.

```julia
call_and_print(f, args...) = (println(f, " ", args); f(args...))

@generated function rewritten(f)
    ci = deepcopy(@code_lowered f.instance())
    for ii in eachindex(ci.code)
        if ci.code[ii] isa Expr && ci.code[ii].head==:call
            func = GlobalRef(Main, :call_and_print)
            ci.code[ii] = Expr(:call, func, ci.code[ii].args...)
        end
    end
    return ci
end
```

We can see that this works:

```julia
julia> foo() = 2*(1+1)
foo (generic function with 1 method)

julia> rewritten(foo)
+ (1, 1)
* (2, 2)
4
```

Rather than replacing each call with `call_and_print`,
we could instead call a function that does the work we are interested in,
and then call `rewriten` on this function.
So, not only does the function we call get rewritten,
but so does every function it calls, all the way down.

```julia
function work_and_recurse(f, args...)
    println(f, " ", args)
    rewritten(f, args...)
end
```

This is how Cassette and IRTools work.
There are a few complexities and special cases that need to be taken care of,
but this is the core of it: recursive invocation of generated functions that rewrite the IR, similar to what is returned by `@code_lowered`.


## Wrapping up Part I

In this part we covered the basics of:
1. Unit syntactic sugar,
2. Pseudo-OO objects with public/private methods,
3. Dynamic Source Tranformation / Custom Compiler Passes (Cassette).

In Part II, we will discuss Traits!
