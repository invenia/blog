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
They emerge from types, and multiple dispatch,
and the ability to dispatch on the types themselves (rather than just instances).
We will talk about both the common use of traits on types,
and also the very interesting (but less common) use of traits on functions.
The latter of which is an emergent feature of functions being instances of singleton types.

There are lots of other cool features that are similarly emergent, which are not discussed in this post.
For example, creating a vector using `Int[]` is actually just an overload of `getindex`, and constructors are overloads of `(::Type{<:T}()`.


## ✖️ Juxtaposition multiplication, convenient syntax for Units

 - [Unitful](https://github.com/ajkeller34/Unitful.jl) is really cool.
 - Units are helpful for protecting against arithmetic mistakes.
 - It is easy to write "2 meters" as `2m`, and units are used.
 - It is not magic.

Example of using Unitful units:
```julia
julia> using Unitful.DefaultSymbols

julia> 1m * 2m
2 m^2

julia> 10kg * 15m / 1s^2
150.0 kg m s^-2

julia> 150N == 10kg * 15m / 1s^2
true
```


### How does this work? Juxtaposition Multiplication
A literal number placed before an expression results in multiplication.


```julia
julia> x = 0.5π
1.5707963267948966

julia> 2x
3.141592653589793

julia> 2sin(x)
2.0
```


### How do we use this to make Units work?
So, to make this work, we are going to make juxtaposition multiplication work for us.
This is a simplified version of what goes on under the hood of Unitful.jl.
We need to overload the multiplication with the constructor, to invoke that constructor.

```julia
abstract type Unit<:Real end
struct Meter{T} <: Unit
    val::T
end

Base.:*(x::Any, U::Type{<:Unit}) = U(x)
```

So, in the above code, we are overloading multiplication with a unit subtype.
Not an instance of a unit subtype, but with the subtype itself.
I.e. with `Meter` not with `Meter(2)`.
That is what `::Type{<:Unit}` says.

We can see that if we try out the above code:
```julia
julia> 5Meter
Meter{Int64}(5)
```
It shows that we create a `Meter` object with `val=5`.

From there, to get to a full unit system, one needs to overload all the stuff that numbers need, like addition and multiplication.
But that is the core trick.

Now we have our own units-style syntactic sugar

## Traits
Inheritance is a gun with only one bullet, at least in the case of single-inheritance like Julia uses.
Once you have a super-type, you can't have another.
Traits are one method of achieving something like multiple inheritiance.
They allow for a form of [polymorphism](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)), that is orthogonal to the inheritence tree.
They can be implemented using functions of types.
 
I have earlier written [another blog post](https://white.ucc.asn.au/2018/10/03/Dispatch,-Traits-and-Metaprogramming-Over-Reflection.html#part-2-aslist-including-using-traits) explaining them, this is a slightly different take, but still similar.
Some parts (including the initial motivating example), are very similar.

### Sometimes people say Julia doesn't have traits

This isn't true, what is true is that:
 - Julia doesn't have syntatic sugar for traits.
 - Julia doesn't have ubiqutious traits.
 - Traits are even used in the Base standard library for iterators: `IteratorSize` and `IteratorEltype`, and for several other interfaces. [[Docs]](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration-1).

### 🙋🏻‍♂️ These are commonly called Holy traits.
Not out of religious glorification, but after [Tim Holy](https://github.com/timholy).
Who is pretty great, to be fair, but still his ideas are not themselves religious objects.
He originally proposed them to [make StridedArrays extensible](https://github.com/JuliaLang/julia/issues/2345#issuecomment-54537633).
Ironically, even though they are fairly well established today, `StridedArray` continues to not use them.
One day that will be fixed. [There are on-going efforts](https://github.com/JuliaDiffEq/ArrayInterface.jl/) to add more traits to arrays, which one-day no-doubt will lead to powerful and general BLAS type functionality. 

### Different ways to implement traits
There are a few different ways to implement them, though all are broadly similar.
 - We're going to talk about the way based on concrete types.
 - But you can do similar with `Type{<:SomeAbstractType}` (ugly, but flexible).
 - Or even with values if they constantly fold (like `Bool`), particularly if you are happy to wrap them in `Val` when you want to dispatch on them (an example of this will be shown later).

### AsList a Python example
In Python TensorFlow, this is a helper function, `_AsList`:

```python
def _AsList(x):
    return x if isinstance(x, (list, tuple)) else [x]
```


This is supposed to convert scalars to single item lists. 
It is useful for only needing to write code that deals with lists.

This is not really idiomatic python code, but TensorFlow uses it in the wild, so it runs on thousands of computers.

#### AsList fails for numpy arrays

```python
>>> _AsList(np.asarray([1,2,3]))
[array([1, 2, 3])]
```

#### Can we fix it?

```python
def _AsList(x):
    return x if isinstance(x, (list, tuple, np.ndarray)) else [x]
```

But where will it end?
What if other packages want to extend this?  
What about other functions that also depend on if something is a list or not?

### Answer: Traits

 - Traits let you mark types as having particular properties.
 - In julia, in particular, you can dispatch on these traits.
 - Often, they will compile out of existence (due to static dispatch, during specialization.).

⚠️ Do note that at some point you have to document what properties a trait requires   
(e.g. what methods must be implemented).
There is no static type-checker enforcing it.
You may want to write a test suite for anything that has a trait that checks it works right.

#### Advantages of Traits
 - You can do this after the type is declared (unlike a supertype).
 - You don't have to do it upfront and can add new types later (unike a `Union`).
 - You can have otherwise unrelated types (unlike a supertype).
 

#### Traits have a few parts: 
 - The trait types: these are the different traits a type can have.
 - The trait function: this tells you what traits a type has.
 - Trait dispatch: using the traits.

#### The type of types

Types are values, and so themselves have a type (`DataType`).
However, they also have the special pseudo-supertype `Type`.
A type `T` acts like `T<:Type{T}`.

```julia
typeof(String) = DataType
String isa Type{String} = true
String isa Type{<:AbstractString} = true
```

We can dispatch on `Type{T}` and have it resolved at compile time

### Trait Type
This is the type that is used to make having the particular trait.
 
In this exanmple we will consider a trait that highlights the properties of a type for statistical modeling.   
Like MLJ's Sci-type, or StatsModel's schema.

```julia
abstract type StatQualia end

struct Continuous <: StatQualia end
struct Ordinal <: StatQualia end
struct Categorical <: StatQualia end
struct Normable <: StatQualia end
```
 
### Trait function
The trait function take a type as input, and returns an instance of the trait type.  
We use the trait function to declare what traits a particular type has.

So we are going to say things like floats are continous, booleans are categorical etc.
```julia
statqualia(::Type{<:AbstractFloat}) = Continuous()
statqualia(::Type{<:Integer}) = Ordinal()

statqualia(::Type{<:Bool}) = Categorical()
statqualia(::Type{<:AbstractString}) = Categorical()

statqualia(::Type{<:Complex}) = Normable()
```

### Using our traits
To use a trait we need to re-dispatch upon it.
This is where we take a the type of an input,
and invoke the trait function on it, to get objects of the trait type,
then dispatch on those.

For this example we are first going to define a `bounds` function.
This bounds function will define some indication of the range of values a particular type has.
It will be defined on a collection of objects with a particular trait,
and it will be defined differently depending on which `statqualia` they have.

```julia
using LinearAlgebra
bounds(xs::AbstractVector{T}) where T = bounds(statqualia(T), xs)

bounds(::Categorical, xs) = unique(xs)
bounds(::Normable, xs) = maximum(norm.(xs))
bounds(::Union{Ordinal, Continuous}, xs) = extrema(xs)
```

Example of use:

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

We can extend traits after the fact.
So, if we wanted to add that vectors have norms defined, we could do:

```julia
julia> statqualia(::Type{<:AbstractVector}) = Normable()
statqualia (generic function with 6 methods)

julia> bounds([[1,1], [-2,4], [0,-2]])
4.47213595499958
```


### So back to `AsList`

#### Define our trait type and trait function:

```julia
struct List end
struct Nonlist end

islist(::Type{<:AbstractVector}) = List()
islist(::Type{<:Tuple}) = List()
islist(::Type{<:Number}) = Nonlist()
```


#### Define our trait dispatch:

```julia
aslist(x::T) where T = aslist(islist(T), x)
aslist(::List, x) = x
aslist(::Nonlist, x) = [x]
```


#### Demo:
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

And, as discussed, it is fully extensible.

### Dynamic dispatch as fallback.

All the traits discussed so far have been fully-static, and they compile away.

But we can also write runtime code, at a small runtime cost.
The following makes a runtime call to `hasmethod` to lookup if the given type has an `iterate` method defined. (There are [plans](https://github.com/JuliaLang/julia/pull/32732) to make `hasmethod` compile time. But for now it can only be done at runtime.)
Similar code to this can be used to dispatch on the values of objects.

```julia
islist(T) = hasmethod(iterate, Tuple{T}) ? List() : Nonlist()
```

We can see that it works on strings, as it does not wrap the following into an array.
```julia
julia> aslist("ABC")
"ABC"
```


## Traits on functions
We can  also attach traits to functions,
becuase functions are instances of singleton types,
e.g. `foo::typeof(foo)`.
We can use this to do declarative input transforms.

### Example: different functions expect the arrangement of observations to be different
Difference ML Models might expect the inputs be:
    - Iterator of Observations.
    - Matrix with Observations in Rows.
    - Matrix with Observations in Columns.

This isn't even a matter of personal perference or different field standards.
For performance reasons there are good reasons to use the different options depending on what operations you are doing.

We shouldn't have to deal with this as user, though.


The following examples use [LIBSVM.jl](https://github.com/mpastell/LIBSVM.jl), and [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl).
One could largely avoid this by using [MLJ's interface](https://github.com/alan-turing-institute/MLJ.jl) instead, which takes care of this kind of thing for you.

First we have some basic functions for dealing with out data.
`get_true_classes` expects an iterator of observations.
```julia
using Statistics
is_large(x) = mean(x) > 0.5
get_true_classes(xs) = map(is_large, xs)

inputs = rand(100, 1_000);  # 100 features, 1000 observations
labels = get_true_classes(eachcol(inputs));
```

For examples sake, we are going to test on our training data,
not something you should ever do to test a real model, except to validate that training worked.
First lets try LIBSVM.
LIBSVM expects the data to come in as a matrix with 1 observation per column.

```julia
using LIBSVM
svm = svmtrain(inputs, labels)

estimated_classes_svm, probs = svmpredict(svm, inputs)
mean(estimated_classes_svm .== labels)
```

Then we can also try DecisionTree.jl.
This library expects data to come as a matrix with 1 observation per row.  
```julia
using DecisionTree
tree = DecisionTreeClassifier(max_depth=10)
fit!(tree, permutedims(inputs), labels)

estimated_classes_tree = predict(tree, permutedims(inputs))
mean(estimated_classes_tree .== labels)
```

What a mess.
We had to know what each different function needed and use `eachcol`, and `permutedims` to change the data around for it.
Why should the user need to remember these details?
We should encode them into the program.

### Lets solve this with traits

We  will attach a trait to each function that needed to rearrange its inputs.

There is a more sophisticated version of this that we could do,
which also attachs traits to inputs saying how the observations are currently arranged; or lets user specify.
But for simplicity we will assume the data starts out, as a matrix with 1 observations per column.

#### Trait types
So we are considering 3 possible ways a function might like its data to be arranged:
```julia
abstract type ObsArrangement end

struct IteratorOfObs <: ObsArrangement end
struct MatrixColsOfObs <: ObsArrangement end
struct MatrixRowsOfObs <: ObsArrangement end
```

#### Trait functions
Now we encode that knowledge about each functions expectations into a trait.
Rather than force the user to look it up from the documentation.

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

We are also going to attach some simple traits to the data types.
To say whether or not they contain observations.
We are just going to use [value types](https://docs.julialang.org/en/v1/manual/types/index.html#%22Value-types%22-1) for this, 
rather than go and fully declare the trait-types.
So we just skip strait to declaring the trait-functions:

```julia
# All matrixes contain observations
isobs(::AbstractMatrix) = Val{true}()

# It must be iterator of vectors, else it doesn't contain observations
isobs(::T) where T = Val{eltype(T) isa AbstractVector}()
```


#### Trait dispatch
Now we define `model_call`: a function which uses the traits to decide how to rearrane the observations before calling the function.
It does this based on the function's type and the argument's types.

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

#### Demo
Now rather than calling things directly we use `model_call`
which takes care of rearranging things.
See how the code is now not having to be aware of the particular cases for each different library?

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

It is basically the same code for each of them.

**In short: Traits: Useful**

## Closures, they give you "Classic OO"
"Classic OO" has classes, with member functions (method)
that can see all the fields and methods,
but that outside the class's methods,
only public fields and methods can be seen.

Achieving this with closures is a classic functional programming trick,
but I first saw it applied to julia in one of   
[Jeff Bezanson's rare Stack Overflow posts](https://stackoverflow.com/a/39150509/179081).
It is pretty elegant in Julia.

### Consider a Duck type.
We use closes to define it as follows:

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


##### 🏗 Can construct an object and can call public methods

```julia
julia> 🦆 = newDuck("Bill")
#7 (generic function with 1 method)

julia> 🦆.get_age()
0
```


##### Public methods can change private fields

```julia
julia> 🦆.inc_age()
1

julia> 🦆.get_age()
1
```


##### Outside the class we can't access private fields

```julia
julia> 🦆.age
ERROR: type ##7#12 has no field age
```

##### Can't access private methods

```julia
julia> 🦆.quack()
ERROR: type ##7#12 has no field quack
```

##### But accessing their functionality via public methods works: 
```julia
julia> 🦆.speak()
Quack!
```

### How does this work?
Closures return singleton objects, with the directly referenced closed variables as fields.
All our public fields/methods are directly referenced.
But our private fields (e.g `age`, `quack`) is not directly referenced, but are closed over other methods that use them.

#### So we can actually see those private methods and fields via accessing the public method closures

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
`Box` is the type julia uses for variables that closed over, but that might be rebound.
Which is the case for primatives (like `Int`) which are rebound whenever they are incremented.

This is a neat example of how closures are basically callable namedtuples.
While this kind of code itself should never be done since Julia has a perfectly functional system for dispatch, and seems to get along fine without Classic OO style encapsulation,
knowing how closures work opens other opertunities to see how they can be used.
In our [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/) project, we are considering the use of closures as callable named tuples as part of a [difficult problem](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/53#issuecomment-533833058) in extensibility and defaults.
Where the default is to call the closure, but you could extend it by using it like a named tuple and accessing its fields.

# How Cassette etc. works

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

### This super-powerful

This capacity allows one to build:
 - AutoDiff tools (ForwardDiff2, Zygote, Yota)
 - Mocking tools (Mocking.jl)
 - Debuggers (MagneticReadHead)
 - Code-proof related tools (ConcolicFuzzer)
 - Generally rewriting all the code (GPUifyLoops)

and more.

One of the most basic features is to effectively overload what it means to call a function.
Call overloading as much more general than operator overloading.
Since it applies to every call special cased as appropriate,
where as operator overloading applies to just one call and just one set of types.

### Consider Normal Functions

```julia
function merge(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    names = merge_names(an, bn)
    types = merge_types(names, typeof(a), typeof(b))
    NamedTuple{names,types}(map(n->getfield(sym_in(n, bn) ? b : a, n), names))
end
```

It at runtime checks the what fields each nametuple has.
To decide what will be in the merge.

But we know all that information base on the types alone.

### Now think about Generated Functions

[Generated functions](https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions-1) take types as inputs and 
return the AST (Abstract Syntax Tree) for what code should run.
It is a kind of metaprogramming.
So as a computation of the types we can workout exactly what to return.
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
We can use `@code_lowered` to look up what the original `CondeInfo` would have been
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
 - Only 1 operation per statement (Nested expressions get broken up) 
 - the return values for each statement is accessed as `SSAValue(index)`
 - Variables become Slots
 - Control-flow becomes jumps (like Goto)
 - function names become qualified as `GlobalRef(mod, func)`

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
We could make it call something that would do the work we are interested in,
and then call `rewriten` on that function.
So the not only does the function we call get rewritten,
but so does every function it calls get written, all the way down.

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
 - Features give rise to other features
 - Types turn out to be very powerful
 - Especially with multiple dispatch

All just fall out of the combination of other features.
 - Unit syntax
 - Traits
 - Closure-based Objects
 - Contextual Compiler Passes