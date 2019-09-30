---
layout: default
title: "The Emergent Features of JuliaLang"
tags:
    - julia
---
By Lyndon White (Research Software Engineer)

This blog post is based on a talk originally given a Cambridge PyData Meetup, and also at London Julia Users Meetup.

### Some features were not planned, but simply emerged from the combinations of other features.

I will show you how these features are implemented:

 - ✅ Unit synatic sugar (`2kg`)
 - ✅ Traits
 - ❌ pseudo-OO objects with public/private methods
 - ❌ Dynamic Source Tranformation / Custom Compiler Passes (Cassette) 
 
Some of these you should do (✅) when appropriate,  
others are rarely approriate (❌) but are interesting to understand how things work

## ✖️ Juxtaposition multiplication, convenient syntax for Units

 - Unitful is really cool
 - units can protect you against artithmatic mistakes
 - You get to write `2m` for 2 meters, and units are use
 - It is not magic

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
using Unitful.DefaultSymbols

1N
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
1m * 2m
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
10kg * 15m / 1s^2
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
150N == 10kg * 15m / 1s^2 
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

### 🖼🗿✖️ Juxtaposition Multiplication
 - A literal number placed before an expression results in multiplication

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
x = 0.5π
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
2sin(x)
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

### 🤔𐄷 How do we use this to make Units work?
 - We just need to overload multiplication
 - In particular we will overload the multiplication with the constructor

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
abstract type Unit end
struct Meter{T} <: Unit
    val::T
end

Base.:*(x::Any, unit::Type{<:Unit}) = unit(x)
{% endhighlight %}
</div>

#### 🍧 Now we have our own units-style syntactic sugar

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
4Meter
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
5.1Meter
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

# ✍️ Traits
 - (single) inheritances is a gun with only one bullet
 - Once you have a super-type, you can't have another
 - Traits are one method of multiple inheritance
 - They can be implemented using functions of types.
 
This is based on [a blog post I wrote a while back explaining traits](https://white.ucc.asn.au/2018/10/03/Dispatch,-Traits-and-Metaprogramming-Over-Reflection.html#part-2-aslist-including-using-traits)

### 🙊 Some times people say julia doesn't have traits
 - Julia doesn't have syntatic sugar for traits
 - Julia doesn't have ubiqutious traits
 - But: traits are use in Base julia for iterators: `IteratorSize` and `IteratorEltype`.

### 🙋🏻‍♂️ These are commonly called (Tim) Holy traits.
<img src="https://avatars1.githubusercontent.com/u/1525481?s=160&v=4"/>

### 👨‍👨‍👦‍👦 Different ways to implement traits
There are a few different ways to implement them, though all are broadly similar.
 - We're going to talk about the way based-on concrete types.
 - But you can do similar with `Type{<:SomeAbstractType}`, (Ugly, but flexible).
 - or even with values if they constant fold (like bools) particularly if you are happy to wrap them in `Val` when you want to dispatch on them.

### 🐍 AsList
In Python TensorFlow, these is a helper function, `_AsList`:

```python
def _AsList(x):
    return x if isinstance(x, (list, tuple)) else [x]
```

 - Supposed to converts scalars to single item lists. 
 - Useful. E.g. for only needing to write code that deals with lists.
 
(Not really idiomatic python code, but eh TensorFlow uses it.   
Should have just called `list(x)`, or fully trusted the 🦆s)

### 😨 AsList fails for numpy arrays
```python
>>> _AsList(np.asarray([1,2,3]))
[array([1, 2, 3])]
```

### 🔨 Fix it?

```python
def _AsList(x):
    return x if isinstance(x, (list, tuple, np.ndarray)) else [x]
```

**But where will it end?**  
**What if other packages want to extend this?**  
**What about other functions that also depend on is something is a list or not?**


### ☑️✅ Answer: Traits

 - Traits let you mark types as having particular properties
 - In julia in particular you can dispatch on these traits
 - and often they will compile out of existance (due to static dispatch)

 - ⚠️ At some point you do have to document what properties a trait requires   
 (e.g. what methods must be implemented)


### ➕🙂  Advantages of Traits
 - You can do this after the type is declared (unlike a supertype)
 - You don't have to do it upfront and can add new types later (unike a `Union`)
 - and you can have otherwise unrelated types (unlike a supertype)
 

### 🧩 Traits have a few parts: 
 - The trait types: these are the different traits a type can have
 - The trait function: this tells you what traits a type has
 - Trait dispatch: using the traits

### ⌨️ 📇 The type of types
 - Types are values, and so themselves have a type (`DataType`).
 - Though, they also act like `T<:Type{T}`
 
```julia
typeof(String) = DataType
String isa Type{String} = true
String isa Type{<:AbstractString} = true
```

We can dispatch on this and have it resolve at compile-time

### 🔪⌨️ Trait Type
This is the type that is used to make having the particular trait.
 
We will consider a trait that highlights the properties of a type for statistical modeling.   
Like MLJ's Sci-type, or StatsModels schema.

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
abstract type StatQualia end

struct Continuous <: StatQualia end
struct Ordinal <: StatQualia end
struct Categorical <: StatQualia end
struct Normable <: StatQualia end
{% endhighlight %}
</div>

## 🔪➡️ Trait function
The trait function take a type as input, and returns an instance of the trait type.  
This is how we declare what traits something has.


**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
statqualia(::Type{<:AbstractFloat}) = Continuous()
statqualia(::Type{<:Integer}) = Ordinal()

statqualia(::Type{<:Bool}) = Categorical()
statqualia(::Type{<:AbstractString}) = Categorical()

statqualia(::Type{<:Complex}) = Normable()
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

### 🏁 Using our traits
To use a trait we need to re-dispatch upon it.

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
using LinearAlgebra
bounds(xs::AbstractVector{T}) where T = bounds(statqualia(T), xs)

bounds(::Categorical, xs) = unique(xs)
bounds(::Normable, xs) = maximum(norm.(xs))
bounds(::Union{Ordinal, Continuous}, xs) = extrema(xs)
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
bounds([false, false])
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
bounds([1,2,3])
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
bounds([1+1im, -2+4im, 0+-2im])
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
statqualia(::Type{<:AbstractVector}) = Normable()

bounds([[1,1], [-2,4], [0,-2]])
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

### 🔙 🐍 So back to `AsList`

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
struct List end
struct Nonlist end

islist(::Type{<:AbstractVector}) = List()
islist(::Type{<:Tuple}) = List()
islist(::Type{<:Number}) = Nonlist()
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

### 🔀 Define our trait dispatch:

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
aslist(x::T) where T = aslist(islist(T), x)
aslist(::List, x) = x
aslist(::Nonlist, x) = [x]
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
aslist(1)
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
aslist([1,2,3])
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

### 🕳🔙 Dynamic dispatch as fallback.

All the traits so far have been fully-static, 
and they compile-away.

But we can also write runtime code,
(at a small runtime cost.)

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
islist(T) = hasmethod(iterate, Tuple{T}) ? List() : Nonlist()
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
aslist("ABC")
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

### 📇🔛 🧮 Traits on functions

 - We can attach traits to functions
 - becuase functions are instances of singleton types
 - `foo::typeof(foo)`
 - We can use this to do declarative input transforms.

### 💻🧠 Example: different functions expect the arrangement of observations to be different

 - ML Models might expect the inputs be:
    - Iterator of Observations
    - Matrix with Observations in Rows
    - Matrix with Observations in Columns
 - For performance reasons there is good reasons to use the different options depending on what operations you are doing.

 - We shouldn't have to deal with this as user though.

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
using Statistics
is_large(x) = mean(x) > 0.5
get_true_classes(xs) = map(is_large, xs)


inputs = rand(100, 1_000);  # 100 features, 1000 observations
labels = get_true_classes(eachcol(inputs));
{% endhighlight %}
</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
using LIBSVM
svm = svmtrain(inputs, labels)

estimated_classes_svm, probs = svmpredict(svm, inputs)
mean(estimated_classes_svm .== labels)
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
using DecisionTree
tree = DecisionTreeClassifier(max_depth=10)
fit!(tree, permutedims(inputs), labels)

estimated_classes_tree = predict(tree, permutedims(inputs))
mean(estimated_classes_tree .== labels)
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

### 🕵️‍♂️ 🔪 Lets solve this with traits

So we will attach a trait to each function that needed to rearrange its inputs.

(There is a more sophisticated version of this that also attachs traits to inputs saying how the observations are currently arranged; or lets user specify.)

#### 🔪⌨️  Trait types

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
abstract type ObsArrangement end

struct IteratorOfObs <: ObsArrangement end
struct MatrixColsOfObs <: ObsArrangement end
struct MatrixRowsOfObs <: ObsArrangement end
{% endhighlight %}
</div>

#### 🔪➡️ Trait functions

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
obs_arrangement(::typeof(get_true_classes)) = IteratorOfObs()

obs_arrangement(::typeof(svmtrain)) = MatrixColsOfObs()
obs_arrangement(::typeof(svmpredict)) = MatrixColsOfObs()

obs_arrangement(::typeof(fit!)) = MatrixRowsOfObs()
obs_arrangement(::typeof(predict)) = MatrixRowsOfObs()
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
isobs(::AbstractMatrix) = Val{true}()

# If an iterator, then must be iterator of vectors,
# else it doesn't contain observations
isobs(::T) where T = Val{eltype(T) isa AbstractVector}()
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

#### 🏁 Trait dispatch

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
function model_call(func, args...; kwargs...)
    return func(maybe_organise_obs.(func, args)...; kwargs...)
end

# trait re-dispatch
maybe_organise_obs(func, arg) = maybe_organise_obs(func, arg, isobs(arg))
maybe_organise_obs(func, arg, ::Val{false}) = arg
function maybe_organise_obs(func, arg, ::Val{true})
    organise_obs(obs_arrangement(func), arg)
end
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
organise_obs(::IteratorOfObs, obs_iter) = obs_iter
organise_obs(::MatrixColsOfObs, obsmat::AbstractMatrix) = obsmat

organise_obs(::IteratorOfObs, obsmat::AbstractMatrix) = eachcol(obsmat)
function organise_obs(::MatrixColsOfObs, obs_iter)
    reduce(hcat, obs_iter)
end

function organise_obs(::MatrixRowsOfObs, obs)
    permutedims(organise_obs(MatrixColsOfObs(), obs))
end
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

#### 🎢🧾 Demo
now rather than calling things directly we use `model_call`
which takes care of rearranging things.

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
inputs = rand(100, 1_000);  # 100 features, 1000 observations
labels = model_call(get_true_classes, inputs);
{% endhighlight %}
</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
using LIBSVM
svm = model_call(svmtrain, inputs, labels)

estimated_classes_svm, probs = model_call(svmpredict, svm, inputs)
mean(estimated_classes_svm .== labels)
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
using DecisionTree
tree = DecisionTreeClassifier(max_depth=10)
model_call(fit!, tree, inputs, labels)

estimated_classes_tree = model_call(predict, tree, inputs)
mean(estimated_classes_tree .== labels)
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

# 🔪😀 Traits: Useful

### 🏛 Closures, they give you "Classic OO"

Like classes, with member functions
that can see private fields, and have public fields.

This is a classic functional programming trick,
but I first saw it applied to julia in one of   
Jeff Bezanson's Stack Overflow posts. 
https://stackoverflow.com/a/39150509/179081

And it is pretty elegant in julia.



**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
function newDuck(name)
    age=0
    get_age() = age
    inc_age() = age+=1
    
    quack() = println("Quack!")
    speak() = quack()
    
    #Declare public:
    ()->(get_age, inc_age, speak, name)
end

{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

#### 🏗 Can construct an object and can call public methods

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
duck1 = newDuck("Bill")
duck1.get_age()
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

#### 👛 Public methods can change state

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
duck1.inc_age()
duck1.get_age()
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

#### 🛑 Can't access private fields

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
duck1.age
{% endhighlight %}
</div>

**Output:**


None

None

None

None

None


##### ⛔️ Can't access private methods

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
duck1.speak()
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-stream jupyter-cell">
{% highlight plaintext %}
None
{% endhighlight %}
</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
duck1.quack()
{% endhighlight %}
</div>

**Output:**


None

None

None

None

None


### ❔👩‍🏫 How does this work?
 - Closures return singleton objects, with the directly referenced closed variables as fields.
 - All our public fields/methods are directly referenced.
 - Our private fields (e.g `age`, `quack`) is not directly referenced, but are closed over other methods that use them.

#### 💡 So we can actually see those private fields via accessing the public method closures

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
duck1.inc_age.age
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
duck1.speak.quack
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-cell">


{% highlight plaintext %}
None
{% endhighlight %}


</div>

**Input:**

<div class="jupyter-input jupyter-cell">
{% highlight julia %}
duck1.speak.quack()
{% endhighlight %}
</div>

**Output:**

<div class="jupyter-stream jupyter-cell">
{% highlight plaintext %}
None
{% endhighlight %}
</div>

# How Cassette etc. works

 <img src="https://raw.githubusercontent.com/jrevels/Cassette.jl/master/docs/img/cassette-logo.png" width="256" style="display: inline"/>

## 📼 🎓 The compiler knows nothing of these "custom compiler passes"

 - Cassette / IRTools (Zygote) is a notable julia feature.
 - Sometimes called:
    - Custom Compiler Passes
    - Contextual Dispatch
    - Dynamicaly-scoped Macros
    - Dynamic-source rewritting
 - It does not work the way you might think it does.
 - The compiler knows nothing of Cassette.

Issue [#21146](https://github.com/JuliaLang/julia/issues/21146)
<img src="./figs/cassette-issue.png"/>

PR [#22440](https://github.com/JuliaLang/julia/pull/22440)
<img src="./figs/cassette-pr.png"/>

## 🏋️‍♂️ Super-powerful

This capacity allows one to build:
 - AutoDiff tools (ForwardDiff2, Zygote, Yota)
 - Mocking tools (Mocking.jl)
 - Debuggers (MagneticReadHead)
 - Code-proof related tools (ConcolicFuzzer)
 - Generally rewriting all the code (GPUifyLoops)
 
and more

## 🔆⚡️ Generated Functions

Generated functions take types as inputs and 
return the AST for what code should run.

It is a kind of metaprogramming.


### 💤 Normal Function

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

### ⚡️ Generated function

So as a computation of the types we can workout exactly what to return.
And can generate code that only accesses the fields we want.

```julia
@generated function merge(a::NamedTuple{an}, b::NamedTuple{bn}) where {an, bn}
    names = merge_names(an, bn)
    types = merge_types(names, a, b)
    vals = Any[ :(getfield($(sym_in(n, bn) ? :b : :a), $(QuoteNode(n)))) for n in names ]
    :( NamedTuple{$names,$types}(($(vals...),)) )
end
```

## 🍰 Julia: layers of representenstation:
 - Source code (`CodeTracking.definition(String,....)`)
 - AST: (`CodeTracking.definition(Expr, ...`)
 - **Untyped IR**: `@code_lowered`
 - Typed IR: `@code_typed`
 - LLVM: `@code_llvm`
 - ASM: `@code_native`
 
You can retrieve the different representations using the functions/macros.

## 📼🚧 How Does Cassette Work?
 - It is not magic, Cassette is not specially baked into the compiler.
 - `@generated` function can return a `Expr` **or** a `CodeInfo`
 - We return a `CodeInfo` based on a modified version of one for a function argument.
 - We can use `@code_lowered` to look up what the original `CondeInfo` would have been


### 🚫⌨️ Untyped IR: this is what we are working with
 - basically a linearization of the AST.
 - Only 1 operation per statement (Nested expressions get broken up) 
 - the return values for each statement is accessed as `SSAValue(index)`
 - Variables ↦ Slots
 - Control-flow ↦ Goto based expressions
 - function names ↦ `GlobalRef(mod, func)`

### ⚙️🤝 Manual pass
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

### ⚙️🤝 Result of our manual pass:
```julia
julia> foo() = 2*(1+1);
julia> rewritten(foo)
+ (1, 1)
* (2, 2)
4
```

### 🔨📼 Rather than doing `call_and_print`:
```julia
function overdub(f, args...)
    println(f, " ", args)
    rewritten(f, args...)
end
```

This is how Cassette and IRTools work.   
🌳 Arborist works similarly, but using AST level.

### 🚫 🧙‍♂️ JuliaLang is not magic
 - Features give rise to other features
 - Types turn out to be very powerful
 - Especially with multiple dispatch

All just fall out of the combination of other features.
 - Unit syntax
 - Traits
 - Closure-based Objects
 - Contextual Compiler Passes

## Bonus Q: When is a JIT not a JIT?
## A: When it doesn't use tracing decide what to specialize

 - In julia by default all functions are specialized on all input types
 - This gives you a thing that looks a lot like a tracing JIT on a dynamic language. But without the tracing.
 - In theory: specialization is semi-orthogonal to your type system.
 - One can specialize on value (c.f. constant-folding)

Also: Multiple dispatch is just adding a human into the loop for how specialization is done.


### Julia just has a very late ahead-of-time compiler
