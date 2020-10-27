---

layout: post

title: "Development with Interface Packages"

author: "Sam Morrison"

comments: false

tags: julia software engineering
---

At Invenia we have built a large, complex code base that is the centerpiece of both our operations and research.
Routinely, we need to test changes in our system from the top level.
We want to be able to replace parts of our system like puzzle pieces and see if the result we get from it is better than the last configuration.
However, those pieces are not designed to work in isolation, and thus cannot be tested in a vacuum.
They also tend to be quite large and complex and therefore justify being packages all on their own, but we need them to “fit” in the same way to avoid having to reconstruct everything from scratch.
This is where interfaces come in handy.
They are tools that allow us to isolate and test certain pieces of our system.

In order to have the option of reusing an interface without pulling in a lot of extra code, we avoid embedding the interface into any package using it, allowing it to exist in a package on its own.
It’s also easier to keep track of the changes to the interface when it is not tied to unrelated changes in other packages.
We therefore have a bunch of very slim interface packages to define what pieces need to be filled in and how they should fit.

## Interface Packages

For our purposes, an interface package is any Julia package that contains an abstract type with no concrete implementation---a sort of IOU in code.
Most functions defined for the type are either unimplemented or are based on other, unimplemented functions.

For modularity, they are separated from the code that uses the abstract type.
As a result, interface packages end up looking like a design document.
For example, below we sketch an interface package for a database.

```julia
module AbstractDB

"""
    AbstractDB

A simple API for read only access to a database.
"""
abstract type AbstractDB end

"""
    AbstractResponse

Query data needed for fetch.
"""
abstract type AbstractResponse end

"""
    query(
        db::AbstractDB,
        table::String,
        features::Vector{Feature}
    ) -> AbstractResponse

Submit an appropriate query to `table` in `db` asking for results matching
`features`.
"""
function query end


"""
    fetch(
        db::AbstractDB,
        response::Union{Response,Vector{Response}}
    ) -> DataFrame

Retrieve the results from `response` and parses them into a `DataFrame`.
"""
function fetch end

    """
    quick_query(
        db::AbstractDB,
        table::String,
        features::Vector{Feature}
    ) -> DataFrame

Submit  and fetches a query to `table` in `db` asking for results matching
`features`.
"""
function quick_query(db::AbstractDB, table::String, features::Vector{Feature})
    return fetch(db, query(db, table, features))
end

end
```

For our purposes we will assume a `Feature` type exists and corresponds to a column in a database table with some criteria for selecting rows.
We will also assume that `filter`ing a `DataFrame` by some `Feature`s works, but is too inefficient to be used in practice.

### A problem with interface packages

While simple to write, interface packages can fall victim to a common failure.
Since interface packages provide little more than function names and docstrings, it’s easy for the packages around them to “go rogue”, changing the function signatures for the interface as they see fit without updating the actual interface package.
This ends up creating a secret, undocumented interface known only to the implemented types and whichever packages manage to use them.

For example let’s say a concrete subtype of our `AbstractDB` database example above changes to let a user define a fancy transform to join up results during `fetch` in a new version.

```julia
# RogueDB.jl
fetch(db::RogueDB, response::Vector{Response}, fancy_transform::Callable)
quick_query(
    db::RogueDB,
    table::String,
    features::Vector{Feature},
    fancy_transform=donothing()
)
```

If another package `X` makes use of the `fancy_transform` in both functions and doesn’t fall back to the documented version of `fetch`, it now can’t easily switch to another `AbstractDB` that doesn't have the `fancy_transform`.
This goes unnoticed, as `X` is only tested against `RogueDB`.

Say we now want to hook up another DB to `X` so we make the fetch function when `NewDB` is created.

```julia
# NewDB.jl
fetch(db::NewDB, response::Vector{Response}, fancy_transform::Callable)
```

`X` has been used for several months and nobody questions the API, so `fetch` without the `fancy_transform` is not implemented.
`X` uses uses the default `quick_query` with `NewDB`, which does not include the `fancy_transform` argument.
Calling `quick_query` will cause a `MethodError`, but `X` only calls that function in an untested (or mocked) special case and the bug goes unseen.

`AbstractDB` has been at v0.1 for 2 years and still has the original definition of `fetch`.
There are several other DB packages being used in other packages that use the documented `fetch`, but nobody has noticed they are different.

Now the interface package doesn’t serve its purpose.
Packages using the interface as documented will fail to run the rogue implementations.
New implementations will have to guess at the secret interface set up by the rogue packages in order to fit into the existing ecosystem.
The version of the interface being used is dependent upon multiple packages.
Chaos!

How do we fix it? Robust tests!

## Test Utils

Having a robust set of test utilities gives an interface package enough muscle to enforce the contract it sets out.

A good set of test utilities for an interface package consists of 2 things:
- A test fake, a simplified implementation of the type for user packages to play with.
- A test suite codifying a minimum set of constraints an implementation must obey to be considered legitimate.

### The Test Suite

#### How?

The test suite is a function or set of functions that work as a validation test for a concrete implementation of a type.
It should codify the expectations of the abstract type (e.g., function signatures, return types) as precisely as possible.
The tests should check that a new implementation does not break the contract that an abstract type sets out.

The test suite should simply test that the interface for the type works as expected.
It does not replace unit tests for the correctness of output or any special-cased behaviour, nor can it be expected to check any implementation details or corner cases.

The test suites should be run as part of the unit tests for any implementation of the type.
If the API changes, it should give instant feedback when the expectations of the abstract type are not met.

A minimal test suite for the database example might look like this:

```julia
"""
    test_interface(db::AbstractDB, table::String, features::Vector{Feature})

Check if an `AbstractDB` follows the documented structure.
The arguments supplied to this function should be identical to those for a
successful `query`.
"""
function test_interface(db::AbstractDB, table::String, features::Vector{Feature})
    q = query(db, table, features)
    @test q isa AbstractResponse

    df = fetch(db, resp)
    @test df isa DataFrame

    @test filter(df, features) == df

    # Check quick_query gives the same results as long form
    @test quick_query(db, table, features) == df

    resp2 = fetch(db, [resp, resp])
    @test resp2 isa AbstractResponse
    @test fetch(db, resp) isa DataFrame
end
```

#### Why?

The test suite puts the power to define the API back into the hands of the interface package.
A package using these tests can be trusted to have defined the functions we need in the form that we expect, so long as the tests pass.
When the interface package changes, the failing tests will serve as a sign to update all related packages to stay consistent with the interface.

The test suite is also handy for creating new implementations.
The test suite is ready-made to be used in [test-driven development](https://en.wikipedia.org/wiki/Test-driven_development) and helps clear up any ambiguities that might exist in the docstrings.

### The Test Fake

#### How?

Test fakes are a kind of [test double](https://en.wikipedia.org/wiki/Test_double).
They have a working implementation but take some shortcuts, which makes them suitable only for testing functionality.

Ideally, a test fake should:
- Be easy to construct.
- Only enact the minimum expectations of the API for the abstract type.
- Give easily verifiable outputs.

Test fakes should be used in tests for any package using the abstract type in order to avoid depending on any "real" version.

A lazy implementation for the database example might be:

```julia
struct FakeDB <: AbstractDB
    data::Dict{String, DataFrame}
end

struct FakeResponse <: AbstractResponse
    fetched::DataFrame
end

FakeDB(data_dir::Path) = FakeDB(populate_data(data_dir))

function query(db::FakeDB, table::String, features::Vector{Feature})
    data = db[table]
    return FakeResponse(filter(data, features))
end

fetch(::FakeDB, r::FakeResponse) = r.fetched
fetch(::FakeDB, rs:Vector{FakeResponse}) = join(fetch.(rs)…)
```

To make sure things are consistent, the `test_interface` function from the section above should be run on `FakeDB` as part of the tests for `AbstractDB`.

#### Why?

Since the test fake is included as part of the interface package, it can be trusted to enact the interface as advertised.
Any code using the fake can be expected to work with any legitimate implementation of the interface that is already passing tests with the test suite.
Since they are minimal examples, they are unlikely to stray from the interface-defined API by adding extra functions that user packages may mistakenly begin to depend on.

Test fakes also help keeping user packages in line with the interface package versions.
When the interface changes, the fakes change and the user code can be fixed to stay consistent with the interface package.

Test fakes can be especially useful shortcuts if a “real” implementation would be awkward to construct or would rely on network access, as they are usually locally-hosted, simplified examples.

## Conclusion

Robust testing for interface packages helps reduce confusion about how abstract types work.
People who are new to the interface can look at the test suite to know how information will flow and can use the included tests as a sanity check while making changes to a concrete implementation.
The test fakes give users a toy example to play with and a minimum set of requirements for what a “real” implementation must do.

It also creates a clear separation of concerns between the packages implementing the code and the packages using the code.
Combined with the docstrings, the test suite should provide implementer packages with all the structure the code needs.
Likewise the test fakes should let user packages test their code without having to depend on a possibly incorrect implementer package.
Neither side should have to look at the details of the other to know how to use the interface itself.

Lastly, forcing both user and implementer packages to test against the interface package guarantees that the version number of the interface package dictates the version of the interface API both can use.
If an interface adds or changes a function and the implementer hasn’t added it, the test suite will fail.
If an interface removes or changes a function and the user package is testing against the test fake, the tests will fail.
If two packages are added using conflicting versions of the API, assuming both packages have bounded the interface package, we get a useful version conflict rather than unexpected behaviour.

Adding interface testing helps interface packages do their job, helping users be sure that when they run their code, a piece will fit where it should without the whole puzzle falling apart.

