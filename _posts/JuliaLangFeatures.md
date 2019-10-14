---
layout: default
title: "The Emergent Features of JuliaLang (Especially Traits)"
Author: Lyndon White
tags:
    - julia
---

This blog post is based on a talk originally given at a Cambridge PyData Meetup, and also at a London Julia Users Meetup.

## Introduction

Julia is known to be a very expressive language with many interesting and cool features.
It is worth considering that some features were not planned,
but simply emerged from the combinations of other features. 
This post will describe how several interesting features are implemented:

 - ✅ Unit synatic sugar (`2kg`),
 - ✅ Traits,
 - ❌ Pseudo-OO objects with public/private methods,
 - ❌ Dynamic Source Tranformation / Custom Compiler Passes (Cassette). 
 
Some of these (✅) should be used when appropriate,  
while others (❌) are rarely appropriate, but are instructive.
A particularly large portion of this post is about traits,
because they are one of the most powerful and interesting julia features.
They emerge from types, multiple dispatch,
and the ability to dispatch on the types themselves (rather than just instances).
We will talk about both the common use of traits on types,
and also the very interesting---but less common---use of traits on functions.
The latter is an emergent feature of functions being instances of singleton types.

There are many other features that are similarly emergent, which are not discussed in this post.
For example, creating a vector using `Int[]` is an overload of 
`getindex`, and constructors are overloads of `::Type{<:T}()`.


## ✖️ Juxtaposition multiplication, convenient syntax for Units

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

So, how does this work? The answer is "Juxtaposition Multiplication": a literal number 
placed before an expression results in multiplication.

```julia
julia> x = 0.5π
1.5707963267948966

julia> 2x
3.141592653589793

julia> 2sin(x)
2.0
```

The following is a simplified version of what goes on under the hood of Unitful.jl.
We need to overload the multiplication with the constructor in order to invoke that constructor.

```julia
abstract type Unit<:Real end
struct Meter{T} <: Unit
    val::T
end

Base.:*(x::Any, U::Type{<:Unit}) = U(x)
```

In the above code, we are overloading multiplication with a unit subtype, not an 
instance of a unit subtype, but with the subtype itself: `Meter` not `Meter(2)`.
This is what `::Type{<:Unit}` means. We can see this if we try out the above code:
```julia
julia> 5Meter
Meter{Int64}(5)
```
This shows that we create a `Meter` object with `val=5`.

To get to a full units system, we then need to overload everything that numbers need to work with, 
such as addition and multiplication. The final result is some units-style syntactic sugar.


## Traits

Inheritance is a gun with only one bullet, at least in the case of single-inheritance like Julia uses.
Single-inheritance means that, once something has a super-type, there can't be another.
Traits are one way to get something similar to multiple inheritance under such conditions.
They allow for a form of [polymorphism](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)), 
that is orthogonal to the inheritence tree, and can be implemented using functions of types.
In an earlier [blog post](https://white.ucc.asn.au/2018/10/03/Dispatch,-Traits-and-Metaprogramming-Over-Reflection.html#part-2-aslist-including-using-traits) 
this is explained using a slightly different take.

Sometimes people claim that Julia doesn't have traits, which is not correct.
Julia does not have _syntatic sugar_ for traits, or ubiqutious traits.
However, traits are used in the Base standard library for iterators: 
`IteratorSize` and `IteratorEltype`, and for several other interfaces. [[Docs]](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration-1).

These are commonly called Holy traits, not out of religious glorification, but after [Tim Holy](https://github.com/timholy).
They were originally proposed to [make StridedArrays extensible](https://github.com/JuliaLang/julia/issues/2345#issuecomment-54537633).
Ironically, even though they are fairly well established today, `StridedArray` continues to not use them.
[There are on-going efforts](https://github.com/JuliaDiffEq/ArrayInterface.jl/) 
to add more traits to arrays, which one day will no doubt lead to powerful and general BLAS type functionality. 

There are a few ways to implement traits, though all are broadly similar.
Here we will focus on the implementation based on concrete types. 
Similar things can be done with `Type{<:SomeAbstractType}` (ugly, but flexible),
or even with values if they are of types that [constant-fold](https://en.wikipedia.org/wiki/Constant_folding) (like `Bool`), 
particularly if you are happy to wrap them in `Val` when dispatching on them (an example of this will be shown later).

There are various advantages of traits:
 - Can be used after the type is declared (unlike a supertype).
 - Don't have to be created upfront, so we can add new types later (unlike a `Union`).
 - Otherwise unrelated types (unlike a supertype) can be used.


### A motivating example from Python: the  `AsList` function

In Python TensorFlow, `_AsList` is a helper function:

```python
def _AsList(x):
    return x if isinstance(x, (list, tuple)) else [x]
```

This converts scalars to single item lists, and is useful because it only needs 
code that deals with lists. This is not really idiomatic python code, but TensorFlow uses it in the wild.
However, AsList fails for numpy arrays:

```python
>>> _AsList(np.asarray([1,2,3]))
[array([1, 2, 3])]
```

This can be fixed in the following way:

```python
def _AsList(x):
    return x if isinstance(x, (list, tuple, np.ndarray)) else [x]
```

But where will it end? What if other packages want to extend this?  
What about other functions that also depend on if something is a list or not?
The answer here is traits, which allow for marking types as having particular properties. 
In julia, in particular, dispatch works on these traits. 
Often they will compile out of existence (due to static dispatch, during specialization).

Note that at some point we have to document what properties a trait requires   
(e.g. what methods must be implemented).
There is no static type-checker enforcing this, so it may be helpful to write a 
test suite for anything that has a trait which checks that it works properly.


### The Parts of Traits

Traits have a few key parts: 
 - The trait types: these are the different traits a type can have.
 - The trait function: this tells you what traits a type has.
 - Trait dispatch: using the traits.

To understand how traits work, it is important to understand the type of types in Julia.
Types are values, so themselves have a type: `DataType`.
However, they also have the special pseudo-supertype `Type`, so a type `T` acts like `T<:Type{T}`.

```julia
typeof(String) = DataType
String isa Type{String} = true
String isa Type{<:AbstractString} = true
```

We can dispatch on `Type{T}` and have it resolved at compile time.


#### Trait Type

This is the type that is used to attribute a particular trait. 
In the following example we will consider a trait that highlights the properties of a type for statistical modeling.   
This is similar to the MLJ Sci-type, or the StatsModel schema.

```julia
abstract type StatQualia end

struct Continuous <: StatQualia end
struct Ordinal <: StatQualia end
struct Categorical <: StatQualia end
struct Normable <: StatQualia end
```
 
#### Trait Function

The trait function takes a type as input, and returns an instance of the trait type.  
We use the trait function to declare what traits a particular type has.

For example, we can say things like floats are continuous, booleans are categorical, etc.
```julia
statqualia(::Type{<:AbstractFloat}) = Continuous()
statqualia(::Type{<:Integer}) = Ordinal()

statqualia(::Type{<:Bool}) = Categorical()
statqualia(::Type{<:AbstractString}) = Categorical()

statqualia(::Type{<:Complex}) = Normable()
```

#### Using Traits

To use a trait we need to re-dispatch upon it.
This is where we take the type of an input,
and invoke the trait function on it to get objects of the trait type,
then dispatch on those.

In the following example we are going to define a `bounds` function.
This function will define some indication of the range of values a particular type has.
It will be defined on a collection of objects with a particular trait,
and it will be defined differently depending on which `statqualia` they have.

```julia
using LinearAlgebra
bounds(xs::AbstractVector{T}) where T = bounds(statqualia(T), xs)

bounds(::Categorical, xs) = unique(xs)
bounds(::Normable, xs) = maximum(norm.(xs))
bounds(::Union{Ordinal, Continuous}, xs) = extrema(xs)
```

Using the above:

```julia
julia> bounds([false, false, true])
2-element Array{Bool,1}:
 false
 true

julia> bounds([false, false, false])
1-element Array{Bool,1}:
 false

julia> bounds([1,2,3,2])
(1, 3)

julia> bounds([1+1im, -2+4im, 0+-2im])
4.47213595499958
```

We can also extend traits after the fact, so, for example, if we want to add the
property that vectors have norms defined, we could define

```julia
julia> statqualia(::Type{<:AbstractVector}) = Normable()
statqualia (generic function with 6 methods)

julia> bounds([[1,1], [-2,4], [0,-2]])
4.47213595499958
```


### Back to `AsList`

Lets first define a trait type and trait function:

```julia
struct List end
struct Nonlist end

islist(::Type{<:AbstractVector}) = List()
islist(::Type{<:Tuple}) = List()
islist(::Type{<:Number}) = Nonlist()
```

The we can define trait dispatch:

```julia
aslist(x::T) where T = aslist(islist(T), x)
aslist(::List, x) = x
aslist(::Nonlist, x) = [x]
```

The above then allows us to do:

```julia
julia> aslist(1)
1-element Array{Int64,1}:
 1

julia> aslist([1,2,3])
3-element Array{Int64,1}:
 1
 2
 3

julia> aslist([1])
1-element Array{Int64,1}:
 1
```

As discussed above this is fully extensible.


### Dynamic dispatch as fallback.

All the traits discussed so far have been fully-static, and they compile away.
We can also write runtime code, at a small runtime cost.
The following makes a runtime call to `hasmethod` to look up if the given type 
has an `iterate` method defined. 
(There are [plans](https://github.com/JuliaLang/julia/pull/32732) to make `hasmethod` compile time. 
But for now it can only be done at runtime.)
Similar code to this can be used to dispatch on the values of objects.

```julia
islist(T) = hasmethod(iterate, Tuple{T}) ? List() : Nonlist()
```

We can see that this works on strings, as it does not wrap the following into an array.
```julia
julia> aslist("ABC")
"ABC"
```


## Traits on functions

We can  also attach traits to functions, because functions are instances of singleton types,
e.g. `foo::typeof(foo)`.
We can use this idea for declarative input transforms.

As an example, we could have different functions expect the arrangement of observations to be different.
More specifically, different machine learning models might expect the inputs be:
- Iterator of Observations.
- Matrix with Observations in Rows.
- Matrix with Observations in Columns.

This isn't a matter of personal perference or different field standards; 
there are good performance related reasons to choose one the above options depending on what operations are needed.
As a user of a model however, we should not have to think about this.

The following examples use [LIBSVM.jl](https://github.com/mpastell/LIBSVM.jl), and [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl).
An alternative would be to use the [MLJ interface](https://github.com/alan-turing-institute/MLJ.jl) instead, 
which takes care of this kind of thing.

First we need some basic functions to deal with our data, 
for example `get_true_classes` expects an iterator of observations.

```julia
using Statistics
is_large(x) = mean(x) > 0.5
get_true_classes(xs) = map(is_large, xs)

inputs = rand(100, 1_000);  # 100 features, 1000 observations
labels = get_true_classes(eachcol(inputs));
```

For simplicity, we will simply test on our training data,
which should be avoided in general (an exception is to validate that training worked).
First lets try LIBSVM, which expect the data to be a matrix with 1 observation per column.

```julia
using LIBSVM
svm = svmtrain(inputs, labels)

estimated_classes_svm, probs = svmpredict(svm, inputs)
mean(estimated_classes_svm .== labels)
```

Next, lets try DecisionTree.jl, which expects data to be a matrix with 1 observation per row.  

```julia
using DecisionTree
tree = DecisionTreeClassifier(max_depth=10)
fit!(tree, permutedims(inputs), labels)

estimated_classes_tree = predict(tree, permutedims(inputs))
mean(estimated_classes_tree .== labels)
```

As we can see above, we had to know what each function needed, and use `eachcol` and `permutedims` to modify the data.
The user should not need to remember these details; they should simply be encoded in the program.


### Traits to the Rescue

We will attach a trait to each function that needed to rearrange its inputs.
There is a more sophisticated version of this,
which also attaches traits to inputs saying how the observations are currently arranged; or lets the user specify.
For simplicity, we will assume the data starts out as a matrix with 1 observations per column.

We are considering 3 possible ways a function might like its data to be arranged:
```julia
abstract type ObsArrangement end

struct IteratorOfObs <: ObsArrangement end
struct MatrixColsOfObs <: ObsArrangement end
struct MatrixRowsOfObs <: ObsArrangement end
```

We can encode this information about each function expectation into a trait,
rather than force the user to look it up from the documentation.

```julia
# Out intial code:
obs_arrangement(::typeof(get_true_classes)) = IteratorOfObs()

# LIBSVM
obs_arrangement(::typeof(svmtrain)) = MatrixColsOfObs()
obs_arrangement(::typeof(svmpredict)) = MatrixColsOfObs()

# DecisionTree
obs_arrangement(::typeof(fit!)) = MatrixRowsOfObs()
obs_arrangement(::typeof(predict)) = MatrixRowsOfObs()
```

We are also going to attach some simple traits to the data types to say whether or not they contain observations.
We will use [value types](https://docs.julialang.org/en/v1/manual/types/index.html#%22Value-types%22-1) for this, 
rather than fully declare the trait-types, so we can just skip straight to declaring the trait functions:

```julia
# All matrices contain observations
isobs(::AbstractMatrix) = Val{true}()

# It must be iterator of vectors, else it doesn't contain observations
isobs(::T) where T = Val{eltype(T) isa AbstractVector}()
```

Next, we can define `model_call`: a function which uses the traits to decide how 
to rearrane the observations before calling the function, based on the function's type and the argument's types.

```julia
function model_call(func, args...; kwargs...)
    return func(maybe_organise_obs.(func, args)...; kwargs...)
end

# trait re-dispatch: don't rearrange things that are not observations
maybe_organise_obs(func, arg) = maybe_organise_obs(func, arg, isobs(arg))
maybe_organise_obs(func, arg, ::Val{false}) = arg
function maybe_organise_obs(func, arg, ::Val{true})
    organise_obs(obs_arrangement(func), arg)
end

# The heavy lifting for rearranging the observations
organise_obs(::IteratorOfObs, obs_iter) = obs_iter
organise_obs(::MatrixColsOfObs, obsmat::AbstractMatrix) = obsmat

organise_obs(::IteratorOfObs, obsmat::AbstractMatrix) = eachcol(obsmat)
function organise_obs(::MatrixColsOfObs, obs_iter)
    reduce(hcat, obs_iter)
end

function organise_obs(::MatrixRowsOfObs, obs)
    permutedims(organise_obs(MatrixColsOfObs(), obs))
end
```

Now, rather than calling things directly, we can use `model_call`
which takes care of rearranging things.
Note that the code no longer needs to be aware of the particular cases for each library!

```julia
inputs = rand(100, 1_000);  # 100 features, 1000 observations
labels = model_call(get_true_classes, inputs);
```

```julia
using LIBSVM
svm = model_call(svmtrain, inputs, labels)

estimated_classes_svm, probs = model_call(svmpredict, svm, inputs)
mean(estimated_classes_svm .== labels)
```

```julia
using DecisionTree
tree = DecisionTreeClassifier(max_depth=10)
model_call(fit!, tree, inputs, labels)

estimated_classes_tree = model_call(predict, tree, inputs)
mean(estimated_classes_tree .== labels)
```

**In short: Traits: Useful**


## Closures give us "Classic OO"

"Classic OO" has classes with member functions (methods) that can see all the fields and methods,
but that outside the class' methods, only public fields and methods can be seen. 
Achieving this with closures is a classic functional programming trick, and it is fairly elegant in julia.
As an example see one of Jeff Bezanson's rare [Stack Overflow posts](https://stackoverflow.com/a/39150509/179081).

Consider a Duck type: we can use closures to define it as follows:

```julia
function newDuck(name)
    age=0
    get_age() = age
    inc_age() = age+=1
    
    quack() = println("Quack!")
    speak() = quack()
    
    #Declare public:
    ()->(get_age, inc_age, speak, name)
end

```

Now we caan construct an object and call public methods:

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

`Box` is the type julia uses for variables that are closed over, but which might be rebound.
This is the case for primatives (like `Int`) which are rebound whenever they are incremented.
This is a nice example of how closures are basically callable NamedTuples.

While this kind of code itself should never be used since Julia has a perfectly 
functional system for dispatch and seems to get along fine without Classic OO style encapsulation,
knowing how closures work opens other opertunities to see how they can be used.
In our [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/) project, 
we are considering the use of closures as callable named tuples as part of 
a [difficult problem](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/53#issuecomment-533833058) 
in extensibility and defaults, where the default is to call the closure, 
which could be extended by using it like a named tuple and accessing its fields.


# Cassette

 <img src="https://raw.githubusercontent.com/jrevels/Cassette.jl/master/docs/img/cassette-logo.png" width="256" style="display: inline"/>

## The compiler knows nothing of these "custom compiler passes"

Cassette / IRTools (Zygote) is a notable julia feature.
This feature is sometimes called:
Custom Compiler Passes,
Contextual Dispatch,
Dynamicaly-scoped Macros,
Dynamic-source rewritting.
It does not work the way you might think it does.
The compiler knows nothing of Cassette.

This incredibly powerful and general feature
And it came out of a very specific issue, and  very casual issue PR suggesting it might be useful for one particular case.
Issue [#21146](https://github.com/JuliaLang/julia/issues/21146)
<img src="{{ site.baseurl }}/public/images/cassette-issue.png"/>

PR [#22440](https://github.com/JuliaLang/julia/pull/22440)
<img src="{{ site.baseurl }}/public/images/cassette-pr.png"/>

### This is super-powerful

This capacity allows one to build:
 - AutoDiff tools (ForwardDiff2, Zygote, Yota)
 - Mocking tools (Mocking.jl)
 - Debuggers (MagneticReadHead)
 - Code-proof related tools (ConcolicFuzzer)
 - Generally rewriting all the code (GPUifyLoops)

and more.

One of the most basic features is to effectively overload what it means to call a function.
Call overloading is much more general than operator overloading.
Since it applies to every call special cased as appropriate,
whereas operator overloading applies to just one call and just one set of types.

### Consider Normal Functions

```julia
function merge(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    names = merge_names(an, bn)
    types = merge_types(names, typeof(a), typeof(b))
    NamedTuple{names,types}(map(n->getfield(sym_in(n, bn) ? b : a, n), names))
end
```

It checks at runtime what fields each nametuple has, to decide what will be in the merge.

But we know all that information based on the types alone.

### Now think about Generated Functions

[Generated functions](https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions-1) take types as inputs and 
return the AST (Abstract Syntax Tree) for what code should run.
It is a kind of metaprogramming.
So, as a computation of the types we can workout exactly what to return.
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

## How Does Cassette Work?
It is not magic, Cassette is not specially baked into the compiler.
`@generated` function can return a `Expr` **or** a `CodeInfo`
We return a `CodeInfo` based on a modified version of one for a function argument.
We can use `@code_lowered` to look up what the original `CondeInfo` would have been.
`@code_lowered` gives back one particular representation of the Julia code: the **Untyped IR**.

## Julia: layers of representenstation:

Julia has many representations of the code it moves through during compilation.

 - Source code (`CodeTracking.definition(String,....)`)
 - AST: (`CodeTracking.definition(Expr, ...`)
 - **Untyped IR**: `@code_lowered`
 - Typed IR: `@code_typed`
 - LLVM: `@code_llvm`
 - ASM: `@code_native`
 
You can retrieve the different representations using the functions/macros indicated in brackets.

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
that makes a copy of the untyped IR, a  `CodeInfo` object, that it gets back from `@code_lowered`,
and then mutates it,
replacing each call with a call to the function `call_and_print`.
It then returns the and new `CodeInfo` to be run when it is called.

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

### Result of our manual pass:
We can see that this works:
```julia
julia> foo() = 2*(1+1)
foo (generic function with 1 method)

julia> rewritten(foo)
+ (1, 1)
* (2, 2)
4
```

### Overdub/recurse.
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

## Conclusion: JuliaLang is not magic
 - Features give rise to other features.
 - Types turn out to be very powerful.
 - Especially with multiple dispatch.

All just fall out of the combination of other features.
 - Unit syntax.
 - Traits.
 - Closure-based Objects.
 - Contextual Compiler Passes.