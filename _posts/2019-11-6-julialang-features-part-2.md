---
layout: post
title: "The Emergent Features of JuliaLang: Part II - Traits"
author: Lyndon White
tags: julia
---

# Introduction

*This blog post is based on a talk originally given at a Cambridge PyData Meetup, and also at a London Julia Users Meetup.*

This is the second of a two-part series of posts on the Emergent Features of JuliaLang.
That is to say, features that were not designed, but came into existence from a combination of other features.
While [Part I]({{ site.baseurl }}{% post_url 2019-10-30-julialang-features-part-1 %}) was more or less a grab-bag of interesting tricks,
Part II (this post), is focused solely on traits, which are, in practice, much more useful than anything discussed in Part I.
This post is fully indepedent from Part I.


# Traits

Inheritance can be a gun with only one bullet, at least in the case of single inheritance like Julia uses.
Single-inheritance means that, once something has a super-type, there can't be another.
Traits are one way to get something similar to multiple inheritance under such conditions.
They allow for a form of [polymorphism](https://en.wikipedia.org/wiki/Polymorphism_(computer_science)),
that is orthogonal to the inheritence tree, and can be implemented using functions of types.
In an earlier [blog post](https://white.ucc.asn.au/2018/10/03/Dispatch,-Traits-and-Metaprogramming-Over-Reflection.html#part-2-aslist-including-using-traits) this is explained using a slightly different take.

Sometimes people claim that Julia doesn't have traits, which is not correct.
Julia does not have _syntactic sugar_ for traits, nor does it have the ubiqutious use of traits that some languages feature.
But it does have traits, and in fact they are even used in the standard library.
In particular for iterators, `IteratorSize` and `IteratorEltype`, and for several other [interfaces](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-iteration-1).

These are commonly called Holy traits, not out of religious glorification, but after [Tim Holy](https://github.com/timholy).
They were originally proposed to [make StridedArrays extensible](https://github.com/JuliaLang/julia/issues/2345#issuecomment-54537633).
Ironically, even though they are fairly well established today, `StridedArray` still does not use them.
[There are on-going efforts](https://github.com/JuliaDiffEq/ArrayInterface.jl/)
to add more traits to arrays, which one day will no doubt lead to powerful and general BLAS-type functionality.

There are a few ways to implement traits, though all are broadly similar.
Here we will focus on the implementation based on concrete types.
Similar things can be done with `Type{<:SomeAbstractType}` (ugly, but flexible),
or even with values if they are of types that [constant-fold](https://en.wikipedia.org/wiki/Constant_folding) (like `Bool`),
particularly if you are happy to wrap them in `Val` when dispatching on them (an example of this will be shown later).

There are a few advantages to using traits:
 - Can be defined after the type is declared (unlike a supertype).
 - Don't have to be created up-front, so types can be added later (unlike a `Union`).
 - Otherwise-unrelated types (unlike a supertype) can be used.


### A motivating example from Python: the  `AsList` function

In Python TensorFlow, `_AsList` is a helper function:

```python
def _AsList(x):
    return x if isinstance(x, (list, tuple)) else [x]
```

This converts scalars to single item lists, and is useful because it only needs code that deals with lists.
This is not really idiomatic python code, but TensorFlow uses it in the wild.
However, `_AsList` fails for numpy arrays:

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
What about other functions that also depend on whether something is a list or not?
The answer here is traits, which give the ability to mark types as having particular properties.
In Julia, in particular, dispatch works on these traits.
Often they will compile out of existence (due to static dispatch, during specialization).

Note that at some point we have to document what properties a trait requires (e.g. what methods must be implemented).
There is no static type-checker enforcing this, so it may be helpful to write a test suite for anything that has a trait which checks that it works properly.


### The Parts of Traits

Traits have a few key parts:
 - Trait types: the different traits a type can have.
 - Trait function: what traits a type has.
 - Trait dispatch: using the traits.

To understand how traits work, it is important to understand the type of types in Julia.
Types are values, so they have a type themselves: `DataType`.
However, they also have the special pseudo-supertype `Type`, so a type `T` acts like `T<:Type{T}`.

```julia
typeof(String) === DataType
String isa Type{String} === true
String isa Type{<:AbstractString} === true
```

We can dispatch on `Type{T}` and have it resolved at compile time.


#### Trait Type

This is the type that is used to attribute a particular trait.
In the following example we will consider a trait that highlights the properties of a type for statistical modeling.
This is similar to the MLJ [ScientificTypes.jl](https://github.com/alan-turing-institute/ScientificTypes.jl), or the [StatsModel](https://github.com/JuliaStats/StatsModels.jl) schema.


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
This is where we take the type of an input, and invoke the trait function on it to get objects of the trait type, then dispatch on those.

In the following example we are going to define a `bounds` function, which will define some indication of the range of values a particular type has.
It will be defined on a collection of objects with a particular trait, and it will be defined differently depending on which `statqualia` they have.

```julia
using LinearAlgebra

# This is the trait re-dispatch; get the trait from the type
bounds(xs::AbstractVector{T}) where T = bounds(statqualia(T), xs)

# These functions dispatch on the trait
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

We can also extend traits after the fact: for example, if we want to add the
property that vectors have norms defined, we could define:

```julia
julia> statqualia(::Type{<:AbstractVector}) = Normable()
statqualia (generic function with 6 methods)

julia> bounds([[1,1], [-2,4], [0,-2]])
4.47213595499958
```


### Back to `AsList`

First, we define the trait type and trait function:

```julia
struct List end
struct Nonlist end

islist(::Type{<:AbstractVector}) = List()
islist(::Type{<:Tuple}) = List()
islist(::Type{<:Number}) = Nonlist()
```

Then we define trait dispatch:

```julia
aslist(x::T) where T = aslist(islist(T), x)
aslist(::List, x) = x
aslist(::Nonlist, x) = [x]
```

This allows `aslist` to work as expected.

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


### Traits on functions

We can  also attach traits to functions, because functions are instances of singleton types,
e.g. `foo::typeof(foo)`.
We can use this idea for declarative input transforms.

As an example, we could have different functions expect the arrangement of observations to be different.
More specifically, different machine learning models might expect the inputs be:
- Iterator of Observations.
- Matrix with Observations in Rows.
- Matrix with Observations in Columns.

This isn't a matter of personal perference or different field standards;
there are good performance-related reasons to choose one the above options depending on what operations are needed.
As a user of a model, however, we should not have to think about this.

The following examples use [LIBSVM.jl](https://github.com/mpastell/LIBSVM.jl), and [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl).
In practice we can just use the [MLJ interface](https://github.com/alan-turing-institute/MLJ.jl) instead,
which takes care of this kind of thing.

In this demonstration of the simple use of a few ML libraries, first we need some basic functions to deal with our data.
In particular we want to define `get_true_classes` which as it would naturally be written would expect an iterator of observations.

```julia
using Statistics
is_large(x) = mean(x) > 0.5
get_true_classes(xs) = map(is_large, xs)

inputs = rand(100, 1_000);  # 100 features, 1000 observations
labels = get_true_classes(eachcol(inputs));
```

For simplicity, we will simply test on our training data,
which should be avoided in general (outside of simple validation that training is working).

The first library we will consider is LIBSVM, which expect the data to be a matrix with 1 observation per *column*.

```julia
using LIBSVM
svm = svmtrain(inputs, labels)

estimated_classes_svm, probs = svmpredict(svm, inputs)
mean(estimated_classes_svm .== labels)
```

Next, lets try DecisionTree.jl, which expects data to be a matrix with 1 observation per *row*.

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
which also attaches traits to inputs saying how the observations are currently arranged, or lets the user specify.
For simplicity, we will assume the data starts out as a matrix with 1 observations per column.

We are considering three possible ways a function might like its data to be arranged.
Each of these will define a different trait type.

```julia
abstract type ObsArrangement end

struct IteratorOfObs <: ObsArrangement end
struct MatrixColsOfObs <: ObsArrangement end
struct MatrixRowsOfObs <: ObsArrangement end
```

We can encode this information about each function expectation into a trait,
rather than force the user to look it up from the documentation.

```julia
# Our intial code:
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
rather than fully declare the trait types, so we can just skip straight to declaring the trait functions:

```julia
# All matrices contain observations
isobs(::AbstractMatrix) = Val{true}()

# It must be iterator of vectors, else it doesn't contain observations
isobs(::T) where T = Val{eltype(T) isa AbstractVector}()
```

Next, we can define `model_call`: a function which uses the traits to decide how
to rearrange the observations before calling the function, based on the type of the function and on the type of the argument.

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

Now, rather than calling things directly, we can use `model_call`,
which takes care of rearranging things.
Notice that the code no longer needs to be aware of the particular cases for each library, which makes things much easier for the end-user: just use `model_call` and don't worry about how the data is arranged.

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

# Conclusion: JuliaLang is not magic

In this series of posts we have seen a few examples of how certain features can give rise to other features.
Unit syntax, Closure-based Objects, Contextual Compiler Passes, and Traits, all just fall out of the combination of other features.
We have also seen that Types turn out to be very powerful, especially with multiple dispatch.
