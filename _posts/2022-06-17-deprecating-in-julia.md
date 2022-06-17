---
layout: post
title: "Deprecating in Julia"
author: "Rory Finnegan, Frames White, Nick Robinson, Arnaud Henry, Miha Zgubič"
tags:
    - open source
    - tutorial
comments: false
---

In software development, the term “deprecated” refers to functionality that is still usable, but is obsolete and about to be replaced or removed in a future version of the software.
This blog post will explain why deprecating functionality is a good idea, and will discuss available mechanisms for deprecating functionality in the Julia language.

## Why deprecate?

An important aspect of software development is minimising the pain of users.
Making few breaking changes to the API goes a long way towards pain minimisation, but sometimes breaking changes are necessary.
Although following the semantic versioning rules [SemVer](https://semver.org) will _alert_ the users of a breaking change by incrementing the major version, the users still need to figure out the changes that need to be made in their code in order to support the new version of your software.

They can do so by going through the release notes, if such notes are provided, or by running the tests and hoping they have good coverage.

Deprecations are a way to make this easier, by releasing a non-breaking version of the software before releasing the breaking version.
In the non-breaking version, the functionality to be removed emits warnings when used, and the warnings describe how to update the code to work in the breaking version.

## How to deprecate in Julia

The following workflow is recommended:

1. Move the deprecated functionality to a file named `src/deprecated.jl`, and deprecate it (see section below for details on how to do this).
2. Move the tests for deprecated functionality to `test/deprecated.jl`.
3. Release a non-breaking version of your package.

Using this non-breaking release will emit deprecation warnings when deprecated functionality is used, guiding users to make the right changes in their code.

Once you are ready to make a breaking release:

1. Delete the src/deprecated.jl file
2. Delete the test/deprecated.jl file
3. Release a breaking version of your package

Note that for more complex breaking changes it may be more practical to not spend too much time on complicated deprecation setups, and instead just produce a written upgrading guide.

The rest of the post describes the private methods available in Julia Base to help with the deprecations.

### `Base.@deprecate`

If you've simply renamed a function or changed the signature without modifying the underlying behaviour then you should use the `Base.@deprecate` macro:

```julia
Base.@deprecate myfunc(x, y, p=1.0) myfunc(x, y; new_p=1.0)
```

The example above will inform the users that a keyword argument was renamed, and advises how to change the call signature.

**NOTE:** `Base.@deprecate` will export the symbol being deprecated by default, but you can disable that behaviour by adding false to the end of you statement:

```julia
Base.@deprecate myfunc(x, y, p=1.0) myfunc(x, y; new_p=1.0) false
```

### `Base.@deprecate_binding`

If you're deprecating a symbol from your package (e.g. renaming a type) then you should use the `@deprecate_binding` macro:

```julia
Base.@deprecate_binding MyOldType MyNewType
```

If you are removing a function without methods (like function bar end) that is typically overloaded, then you can use:

```julia
Base.@deprecate_binding bar nothing
```

### `Base.@deprecate_moved`

If you're deprecating a symbol from your package because it has simply been moved to another package, then you can use the `@deprecate_moved` macro to give an informative error:

```julia
Base.@deprecate_moved MyTrustyType "OtherPackage"
```


### `Base.depwarn`

If you're introducing other breaking changes that do not fall in any of the above categories, you can use `Base.depwarn` and write a custom deprecation warning.
An example would be deprecating a type parameter:

```julia
function MyType{G}(args...) where G
    Base.depwarn(
        "The `G` type parameter will be deprecated in a future release. " *
        "Please use `MyType(args...)` instead of `MyType{$G}(args...)`.",
        :MyType,
    )
    return MyType(args...)
end
```

Unlike other methods of deprecation, `Base.depwarn` has a `force` keyword argument that will cause it to be shown even if Julia is run with (the default) `--depwarn=false`.
Tempting as it is, we advise against using it in most circumstances, because there is a good reason deprecations are not shown.
Printing a warning in the module `__init__()` is a similar option that should almost never be used.

### Warnings in module `__init__()`

If you're making broad changes at the module level (e.g., removing exports, changes to defaults/constants) you may simply want to throw a warning when the package loads.
However, this should be a last resort, as it makes a lot of noise, often in unexpected places (e.g downstream dependents).

```julia
# Example from Impute.jl
function __init__()
    sym = join(["chain", "chain!", "drop", "drop!", "interp", "interp!"], ", ", " and ")

    @warn(
        """
        The following symbols will not be exported in future releases: $sym.
        Please qualify your calls with `Impute.<method>(...)` or explicitly import the symbol.
        """
    )

    @warn(
        """
        The default limit for all impute functions will be 1.0 going forward.
        If you depend on a specific threshold please pass in an appropriate `AbstractContext`.
        """
    )

    @warn(
        """
        All matrix imputation methods will be switching to the JuliaStats column-major convention
        (e.g., each column corresponds to an observation, and each row corresponds to a variable).
        """
    )
end
```

## When are deprecations shown in Julia?

Since Julia 1.5 (PR [#35362](https://github.com/JuliaLang/julia/pull/35362)) deprecations have been muted by default.
They are only shown if Julia is started with the `--depwarn=yes` flag set.
The reason for this has two main parts.
The practical reason is the deprecation logging itself is quite slow and so by calling a deprecated method in an inner loop some packages were becoming unusably slow, thus making the deprecation effectively breaking.
The logical reason is the user of a package often can’t fix them (unless called directly), they often need to be fixed in some downstream dependency of the package they are using.
Thanks to Julia packages following SemVer, calling a deprecated method is completely safe, until you relax the bounds.

When relaxing the compat bound it is good practice to check the release notes, relax the compat bounds and then run integration tests.
If everything passes then you are done, if not retighten the compat bounds and run with `--depwarn=error` (or pay careful attention to logs) and track them down.

Deprecations *are* by default shown during tests.
This is because tests are run mostly by the developers of the package.
However, they still may be unfixable by the developer of the package if they are downstream, but at least the developer is probably in a better position to open PRs or issues in their dependencies.

Internally at Invenia, we take a different approach.
Because deprecations are so spammy we also disable them in our main CI runs.
But we have an additional CI job that is set to `--depwarn=error`, with allowed-to-fail set on the job.
This keeps the logs in our main CI clean, while also quickly alerting us if any deprecated methods are hit.

## Conclusions

Deprecating package functionality is a bit of work, but it minimises the amount of pain for package users when breaking changes are released.

The recommended workflow in Julia includes moving all the deprecated functionality and its tests into dedicated deprecated.jl files, which makes it easier to remove once the breaking version is released.

Julia offers several convenience macros in Base which make common operations like changing a function signature, moving functionality, or renaming types a one-line operation.
For more complicated changes a custom message can be written.

Deprecation warnings are suppressed in Julia by default for performance reasons, but they can be shown by setting the `--depwarn=true` flag when starting a new Julia session.

Happy deprecating!
