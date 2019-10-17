---
layout: default
title: "The Emergent Features of JuliaLang: Part I"
Author: Lyndon White
tags:
    - julia
---


# Introduction

*This blog post is based on a talk originally given at a Cambridge PyData Meetup, and also at a London Julia Users Meetup.*

Julia is known to be a very expressive language with many interesting features.
It is worth considering that some features were not planned,
but simply emerged from the combinations of other features. 
This post will describe how several interesting features are implemented:

1. Unit syntactic sugar,
2. Pseudo-OO objects with public/private methods,
3. Dynamic Source Tranformation / Custom Compiler Passes (Cassette),
4. Traits,
 
Some of these (1 and 4) should be used when appropriate, 
while others (2 and 3) are rarely appropriate, but are instructive.
A particularly large portion of this post is about traits
because they are one of the most powerful and interesting Julia features.
They emerge from types, multiple dispatch,
and the ability to dispatch on the types themselves (rather than just instances).
We will review both the common use of traits on types,
and also the very interesting---but less common---use of traits on functions.

There are many other features that are similarly emergent, which are not discussed in this post.
For example, creating a vector using `Int[]` is an overload of 
`getindex`, and constructors are overloads of `::Type{<:T}()`.

In Part I of this post we will cover topics 1-3, and in Part II, we will cover Traits.


# Juxtaposition Multiplication: Convenient Syntax for Units

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

How does this work? The answer is *Juxtaposition Multiplication*, a literal number 
placed before an expression results in multiplication, for example:

```julia
julia> x = 0.5π
1.5707963267948966

julia> 2x
3.141592653589793

julia> 2sin(x)
2.0
```

To make this work, we need to overload the multiplication with the constructor, 
in order to invoke that constructor. 
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

To get to a full units system, we then need to overload everything that numbers need to work with, 
such as addition and multiplication. The final result is units-style syntactic sugar.

# Closures give us "Classic OO"

First it is important to emphasize: **don't do this in real Julia code**.
It is unidiomatic, and likely to hit edge-cases the compiler doesn't optimize well 
(see for example the [infamous closure boxing bug](https://github.com/JuliaLang/julia/issues/15276)).
This is all the more important because it often requires boxing (see below); so the boxing of things won't even be a bug.

"Classic OO" has classes with member functions (methods) that can see all the fields and methods,
but that outside the class' methods, only public fields and methods can be seen. 
This idea was originally posted for Julia in one of Jeff Bezanson's rare 
[Stack Overflow posts](https://stackoverflow.com/a/39150509/179081).
Though in it is a classic functional programming trick.

Consider a Duck type: we can use closures to define it as follows:

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

This can do various things we would expect from classic "OO" encapsulation.
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

Closures return singleton objects, with directly referenced closed variables as fields.
All the public fields/methods are directly referenced, 
but the private fields (e.g `age`, `quack`) are not directly referenced, they are closed over other methods that use them.
We can actually see those private methods and fields via accessing the public method closures:

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

### An aside on Boxing

The `Box` type is similar to the `Ref` type, in function and purpose.
`Box` is the type Julia uses for variables that are closed over, but which might be rebound.
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

In closures, boxing applies only to rebinding, though the [closure bug](https://github.com/JuliaLang/julia/issues/15276)
does mean Julia will sometimes over-eagerly box variables because it thinks they might be rebound.
It has no change on what the code does, but it does impact performance.

While this kind of code itself should never be used, since Julia has a perfectly 
functional system for dispatch and seems to get along fine without Classic OO-style encapsulation,
knowing how closures work opens other opportunities to see how they can be used.
In our [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/) project, 
we are considering the use of closures as callable named tuples as part of 
a [difficult problem](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/53#issuecomment-533833058) 
in extensibility and defaults, where the default is to call the closure, 
which could be extended by using it like a named tuple and accessing its fields.


# Cassette, etc.

 <img src="https://raw.githubusercontent.com/jrevels/Cassette.jl/master/docs/img/cassette-logo.png" width="256" style="display: inline"/>

Cassette/IRTools (Zygote) is a notable Julia feature.
This feature is sometimes called:
Custom Compiler Passes,
Contextual Dispatch,
Dynamicaly-scoped Macros,
Dynamic-source rewritting.
It does not work the way you might think it does.
The compiler knows nothing of "custom compiler passes".

This is an incredibly powerful and general feature, and was the result of a very specific 
issue and very casual PR suggesting that it might be useful for one particular case.
Issue [#21146](https://github.com/JuliaLang/julia/issues/21146)
<img src="{{ site.baseurl }}/public/images/cassette-issue.png"/>

PR [#22440](https://github.com/JuliaLang/julia/pull/22440)
<img src="{{ site.baseurl }}/public/images/cassette-pr.png"/>

Technically, the above describe everything about Cassette, but only in passing.

This "custom compiler pass" feature is essential for:
 - AutoDiff tools (ForwardDiff2, Zygote, Yota);
 - Mocking tools (Mocking.jl);
 - Debuggers (MagneticReadHead);
 - Code-proof related tools (ConcolicFuzzer);
 - Generally rewriting all the code (GPUifyLoops).

One of the most basic features is to effectively overload what it means to call a function.
Call overloading is much more general than operator overloading.
Since it applies to every call special-cased as appropriate,
whereas operator overloading applies to just one call and just one set of types.

To give a concrete example of how call overloading is more general:
operator overloading/dispatch  (multiple or otherwise) would allow one to
to overload, for example, `sin(::T)` for different types `T`,
so `sin(::DualNumber)` could be specialized to be different from `sin(::Float64)`---
such that in that case the it could be set to calculate the nonreal part using the `cos` (which is the  derivative of sin)---[that is what DualNumbers do](https://en.wikipedia.org/wiki/Dual_number#Differentiation).
However, operator overloading can't express the notion that, for all functions `f`, if the input type is `DualNumber`,
that `f` should calculate the dual part using the deriviative of `f`.
Call overloading allows for much more expressivity and massively simplifies the implementation of automatic differentiation.

Lets take a look at an example with normal functions:
```julia
function merge(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    names = merge_names(an, bn)
    types = merge_types(names, typeof(a), typeof(b))
    NamedTuple{names,types}(map(n->getfield(sym_in(n, bn) ? b : a, n), names))
end
```

This checks at runtime what fields each nametuple has, to decide what will be in the merge.
However, note that we know all this information based on the types alone.

Next, [generated functions](https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions-1) 
take types as inputs and return the AST (Abstract Syntax Tree) for what code should run.
It is a kind of metaprogramming.
So, based on information in the types it worked out what code should be run.
A simple example is for function taking as input an `N`-dimentional array (type `AbstractArray{T, N}`)
a generated function might construct code with `N` nested loops to process each elements.
It can generate code that only accesses the fields we want.

```julia
@generated function merge(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    names = merge_names(an, bn)
    types = merge_types(names, a, b)
    vals = Any[ :(getfield($(sym_in(n, bn) ? :b : :a), $(QuoteNode(n)))) for n in names ]
    :( NamedTuple{$names,$types}(($(vals...),)) )
end
```

This gives substantial preformance improvements.

Casette is not magic, and is not baked into the compiler.
`@generated` functions can return an `Expr` **or** a `CodeInfo`
We return a `CodeInfo` based on a modified version of one for a function argument.
We can use `@code_lowered` to look up what the original `CondeInfo` would have been.
`@code_lowered` gives back one particular representation of the Julia code: the **Untyped IR**.


### Layers of Representation

Julia has many representations of the code it moves through during compilation.

 - Source code (`CodeTracking.definition(String,....)`)
 - AST: (`CodeTracking.definition(Expr, ...`)
 - **Untyped IR**: `@code_lowered`
 - Typed IR: `@code_typed`
 - LLVM: `@code_llvm`
 - ASM: `@code_native`
 
You can retrieve the different representations using the functions/macros indicated.


### Untyped IR: this is what we are working with

This is Basically a linearization of the AST.
 - Only 1 operation per statement (nested expressions get broken up);
 - the return values for each statement are accessed as `SSAValue(index)`;
 - variables become Slots;
 - control-flow becomes jumps (like Goto);
 - function names become qualified as `GlobalRef(mod, func)`.

It isn't great to work-with.
It is ok to read, but writing it gives a special kind of headache.
That is why IRTools and Cassette exist, to take some of that pain away,
but we want to do it manually to understand how it works.

This example originally showed up in my [JuliaCon talk on MagneticReadHead.jl](https://www.youtube.com/watch?v=lTR6IPjDPlo)


### Manual pass

We define a generated function `rewritten`,
that makes a copy of the untyped IR, a  `CodeInfo` object, that it gets back from `@code_lowered`
and then mutates it,
replacing each call with a call to the function `call_and_print`.
It then returns the new `CodeInfo` to be run when it is called.

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

### Result of our manual pass

We can see that this works:
```julia
julia> foo() = 2*(1+1)
foo (generic function with 1 method)

julia> rewritten(foo)
+ (1, 1)
* (2, 2)
4
```

### Overdub/recurse

Rather than replacing each call with `call_and_print`,
we could make it call something that would do the work we are interested in,
and then call `rewriten` on that function.
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
but that is the core of it.
Recursive invocation of generated functions that rewrite the IR, like what is returned by `@code_lowered`.

This covers topics 1-3. 
In Part II, we will discuss Traits!
