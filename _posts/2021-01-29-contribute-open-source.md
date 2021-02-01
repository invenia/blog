---
layout: post
title: "How to Start Contributing to Open Source Software"
author: "Miha Zgubic"
tags:
    - open source
    - tutorial
    - first contribution
comments: false
---


If you are someone who feels comfortable using code to solve a problem, answer a question, or just implement something for fun, chances are you are relying on [open source software](https://opensource.com/resources/what-open-source).
If you want to contribute to open source software, but don’t know how and where to start, this guide is for you.

In this post, we will first discuss some of the possible mental and technical barriers standing between you and your first meaningful contribution.
Then, we will go through the steps involved in making the first contribution.

For the sake of brevity, we assume some basic familiarity with [git](https://guides.github.com/introduction/git-handbook/) (if you know how to commit changes and what a branch is, you’ll be fine).
Because many open source projects are hosted on GitHub, we also discuss [GitHub Issues](https://guides.github.com/features/issues), which refer to GitHub functionality that allows discussion about bugs and new features, and the term “pull/merge request” (PR/MR, they are the same thing), which is a mechanism for submitting your changes to be incorporated into the existing repository.

## Mental and Technical Barriers

It is easy to observe the world of open source and think:
“Oh look at all these smart people doing meaningful work and developing meaningful relationships, building humanity’s digital heritage that helps to run our civilisation.
I wish I could join the party, _but_ \<something I believe to be true\>.”

Some misconceptions I and other people have had are listed below, along with what we think about it now.

> “I need to be an expert in \<the language\> AND \<the package\> to even consider reporting a bug, making a fix, or implementing new functionality.”

If you are using the package and something doesn’t work as expected, you should report it, for example by opening a GitHub issue.
You don’t need to know how the package works internally.
All you need to do is take a quick look at the documentation to see if anything is written about your problem, and check if a similar issue has been opened already.
If you want to fix something yourself, that’s fantastic! Just jump in and try to figure it out.

> “I know about \<a thing\>, but other people know much more, and it’s much better if they implemented it.”

This is almost always the case, unless you are the world’s leading expert on \<a thing\>.
However, they are likely busy with other important work, and the issue is not a priority for them.
So, either you implement it, or it may not get done at all, so go ahead!
The best part, however, is that these other more knowledgeable people will usually be happy to review your solution and suggest how to make it even better, which means that a great thing gets built and you learn something in the process.

> “I don’t know anyone contributing to the package and they look like a team, isn’t it weird if I just jump in and open an issue/PR?”

No, it’s not weird.
Teams make their code and issues public because they’re looking for new contributors like you.

> “I won’t be able to create a perfect solution, and people will point out flaws and ask me to change it.”

Solutions to issues usually come in the form of pull requests.
However, opening a PR is best thought of as a conversation about a solution, rather than a finished product that is either approved or rejected.
Experienced contributors often open PRs to solicit feedback about an idea, because an open PR on GitHub offers convenient tools for discussion about code.
Even if you think your solution is complete, people will likely ask you to make changes, and that’s alright!
If it isn’t explicitly mentioned (it should be), ask why the changes are needed—these are valuable opportunities to learn.

> “I would like to make a contribution, but don’t know where to start.”

Finding the right place to start can be challenging, but see advice below.
Once you make a contribution or two they will lead you on to others, so you typically only have to overcome this barrier once.

> “I know what is broken and I think I know how to fix it, but don’t know the steps to publish these to the official repository.”

That’s fantastic! See the rest of the guide.

> “What if people ask for changes? How do I implement those?”

Somehow I thought implementing review feedback was hard and messy, but, in practice, it’s as easy as adding more commits to a branch.

## Steps to First Contribution

Now that we have gone through some of the concerns you may have, here is the step-by-step guide to your first contribution.

### 1) Learn the mechanics of a pull request

The workflow is described in this [excellent repository on GitHub](https://github.com/firstcontributions/first-contributions), built just for learning the mechanics of making a pull request.
I recommend to go through it by not just reading it, but by actually going through all the steps.
That exercise should make you comfortable with the process.

### 2) Find something you want to fix

A good first project might be solving a bug that affects you, as that means you already have a test case and you will be more motivated to find a solution.
However, if the bug is in a large and complicated library or requires a lot of code refactoring to fix, it is probably better to start somewhere else.

It may be more enjoyable to start with smaller or medium-sized packages because they can be easier to understand.
When you find a package you would like to modify, make sure that it is the original (not a fork) and it is being maintained, which you can check by looking at the issues and pull requests on its GitHub page.

Then, look through the issues and see if there is something that you find interesting.
Pay attention to `good-first-issue` labels, which indicate issues that people think are appropriate for first-time contributors.
This usually means that they are nice and not too hard to solve.
You don’t have to restrict yourself to issues `good-first-issue` labels, feel free to tackle anything you feel motivated and able to do.
Keep in mind that it might be better to start with a smaller PR and get that merged first, before tackling a bigger issue.
You don’t want to submit a week worth of work only to find out that the package has been abandoned and there is nobody willing to review and merge your PR.

When you find an interesting issue and decide you want to work on it, it is a good idea to comment on the issue first and ask whether anyone is willing to review a potential PR.
Commenting will also create a feeling of responsibility and ownership of the issue which will motivate you and help you finish the PR.

As a few ideas, here are some concrete Julia packages that Invenia is involved with, for example [AWS.jl](https://github.com/JuliaCloud/AWS.jl) for interacting with AWS, [Intervals.jl](https://github.com/invenia/Intervals.jl) for working with intervals over ordered types, [BlockDiagonals.jl](https://github.com/invenia/BlockDiagonals.jl) for working with block-diagonal matrices, and [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) for automatic differentiation.
We are happy to help you contribute to these!

### 3) Implement your solution, and open a PR

While you should be familiar with the mechanics of pull requests after step 1, there are some additional social/etiquette considerations.
Generally, the authors of open source packages are delighted when someone uses their package, opens issues about bugs or potential improvements, and especially so when someone opens a pull request with a solution to a known problem.
That said, they will appreciate it if you make things easy for them by linking the issue your PR is solving and a brief reason why you have chosen this approach.
If you are unsure about whether something was a good choice, point it out in the description.

If your background isn’t in computer science or software engineering, you might not have heard of [unit testing](https://ocw.mit.edu/ans7870/6/6.005/s16/classes/03-testing/index.html).
Testing is a way of ensuring the correctness of the code by checking the output of the code for a number of inputs.
In packages with unit tests, every new feature is typically expected to come with tests.
When fixing a bug, a test that fails using the old code and passes using the new code may also be expected.

You can make the review process more efficient and pleasant by quickly examining the PR yourself.
What needs to be included in the PR depends on the issue at hand, but there are some general questions you can think about:
- Does my code work in corner cases, did I include reasonable tests?
- Did I add or change documentation to match my changes?
- Does my code formatting follow the rest of the package? Some packages follow code style guides, such as [PEP8](https://www.python.org/dev/peps/pep-0008/) or [BlueStyle](https://github.com/invenia/BlueStyle).
- Did I include any lines or files by mistake?

Don’t worry if you can’t answer these questions.
It is perfectly fine to ask!
You can also self-review your PR and add some thoughts as comments to the code.

The [contributor’s guide on collaborative practices](http://colprac.sciml.ai) is a great resource about the best practices regarding collaboration on open source projects.
The packages that follow it display this badge: [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
Other packages typically have their own guidelines outlined in a file named `CONTRIBUTING.md`.

### 4) Address feedback, and wait for the merge

Once the PR is submitted someone will likely respond to it in a few days.
If it doesn’t happen, feel free to “bump” it by adding a comment to the PR asking for it to be reviewed.
Most maintainers do not mind if you bump the PR every ten days or so, and in fact find it useful in case it has slipped under their radar.

Ideally the feedback will be constructive, actionable, and educational.
Sometimes it isn’t, and if you are very unlucky the reviewer might come across as stern and critical.
It helps to remember that such feedback might have been unintentional and that you are in fact on the same side, both wanting the best code to be merged.
Using plural first-person pronouns (we) is a good way to convey this sentiment (and remind the reviewer about it), for example:
“What are the benefits if we implement \<a feature\> in this way?” is better than “Why do you think \<a feature\> should be implemented this way?”.

Addressing feedback is easy: simply add more commits to the branch that the PR is for.
Once you think you have addressed all the feedback, let the reviewer know explicitly, as they don’t know whether you plan to add more commits or not.
If all went well the reviewer will then merge the PR, hooray!

## Conclusions

Unless you have started programming very recently, you likely already have the technical/programming ability to contribute to open source projects.
You have valuable contributions to make, but psychological/sociological barriers may be holding you back.
Hopefully reading this post will help you overcome them: and we are looking forward to welcoming you to the community and seeing what you come up with!

