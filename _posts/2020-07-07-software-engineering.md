---
layout: post

title: "The Hitchhiker’s Guide to Research Software Engineering: From PhD to RSE"

author: "Glenn Moynihan"

comments: false

tags: research software engineering phd
---

In 2017, the twilight days of my PhD in computational physics, I found myself ready to leave academia behind.
While my research was interesting, it was not what I wanted to pursue full time.
However, I was happy with the type of work I was doing, contributing to research software, and I wanted to apply myself in a more industrial setting.

Many postgraduates face a similar decision.
A [study conducted by the Royal Society](https://royalsociety.org/-/media/Royal_Society_Content/policy/publications/2010/4294970126.pdf) in 2010 reported that only 3.5% of PhD graduates end up in permanent research positions in academia.
Leaving aside the roots of the [brain drain](https://jakevdp.github.io/blog/2013/10/26/big-data-brain-drain/) on Universities, it is a compelling statistic that the _vast_ majority of post-graduates end up leaving academia for industry at some point in their career.
It comes as no surprise that there are a growing number of bootcamps like [S2DS](https://www.s2ds.org/index.html), [faculty.ai](https://faculty.ai/), and [Insight](https://insightfellows.com/data-science) that have sprung up in response to this trend, for machine learning and data science especially.
There are also no shortage of helpful [forum discussions](https://news.ycombinator.com/item?id=17944306) and [blog posts](https://pascalbugnion.net/blog/from-academia-to-data-science.html) outlining what you should do in order to “break into the industry”, as well as many that relate the personal experiences of those who ultimately made the switch.

While the advice that follows in this blog post is directed at those looking to change careers, it would equally benefit those who opt to remain in the academic track.
Since the environment and incentives around building academic research software are very different to those of industry, the workflows around the former are, in general, not guided by the same engineering practices that are valued in the latter.

That is to say: _there is a difference between what is important in writing software for research, and for a user-focused, software product_.
Academic research software prioritises scientific correctness and flexibility to experiment above all else in pursuit of the researchers’ end product: published papers.
Industry software, on the other hand, prioritises maintainability, robustness, and testing as the software (generally speaking) _is_ the product.

However, the two tracks share many common goals as well, such as catering to “users”, emphasising performance and [reproducibility](https://codecheck.org.uk/), but most importantly both ventures are _collaborative_.
Arguably then, both sets of principles are needed to write and maintain high-quality research software.
Incidentally, the [Research Software Engineering](https://society-rse.org/about/) group at Invenia is uniquely tasked with incorporating all these incentives into the development of our research packages in order to get the best of both worlds.
But I digress.

## What I wish I knew in my PhD
Most postgrads are self-taught programmers and learn from the same resources as their peers and collaborators, which are ostensibly adequate for academia.
Many also tend to work in isolation on their part of the code base and don't require merging with other contributors’ work very frequently.
In industry, however, [continuous integration](https://en.wikipedia.org/wiki/Continuous_integration) underpins many development workflows. 
Under a continuous delivery cycle, a developer benefits from the prompt feedback and cooperation of a full team of professional engineers and can, therefore, learn to implement engineering best practices more efficiently.

As such, it feels like a missed opportunity for universities not to promote good engineering practices more and teach them to their students.
Not least because having stable and maintainable tools are, in a sense, ["public goods"](https://en.wikipedia.org/wiki/Commons#Digital_commons) in academia as much as industry.
Yet, while everyone gains from improving the tools, researchers are not generally incentivised to invest their precious time or effort on these tasks unless it is part of some well-funded, high-impact initiative.
As Jake VanderPlas [remarked](https://jakevdp.github.io/blog/2013/10/26/big-data-brain-drain/): “any time spent building and documenting software tools is time spent not writing research papers, which are the primary currency of the academic reward structure”.

Speaking personally, I learned a great deal about conducting research and scientific computing in my PhD; I could read and write code, squash bugs, and I wasn’t afraid of getting my hands dirty in monolithic code bases.
As such, I felt comfortable at the command line but I failed to learn the basic tenets of proper code maintenance, unit testing, code review, version control, etc., that underpin good software engineering.
While I had enough coding experience to have a sense of this at the time, I lacked the awareness of what I needed to know in order to improve or even where to start looking.

As is clear from the earlier statistic, this experience is likely not unique to me.
It prompted me to share what I’ve learned since joining Invenia 18 months ago, so that it might guide those looking to make a similar move.
The advice I provide is organised into three sections: the first recommends ways to learn a new programming language efficiently[^1]; the second describes some best practices you can adopt to improve the quality of the code you write; and the last commends the social aspect of community-driven software collaborations.

## Lesson 1: Hone your craft
**Practice**: While clichéd, there is no avoiding the fact that it takes consistent practice [over many many years](https://norvig.com/21-days.html) to become masterful at anything, and programming is no exception.

**Have personal projects**: Practicing is easier said than done if your job doesn’t revolve around programming.
A good way to get started either way is to undertake personal side-projects as a fun way to get to grips with a language, for instance via [Project Euler](https://projecteuler.net/), [Kaggle Competitions](https://www.kaggle.com/), etc.
These should be enough to get you off the ground and familiar with the syntax of the language.

**Read code**: Personal projects on their own are not enough to improve. 
If you really want to get better, you've got to read other people's code: a lot of it.
Check out the repositories of some of your favourite or most used packages---particularly if they are considered “high quality”[^2].
See how the package is organised, how the documentation is written, and how the code is structured.
Look at the open issues and pull requests.
Who are the main contributors? Get a sense of what is being worked on and how the open-source community operates.
This will give you an idea of the open issues facing the package and the language and the direction it is taking.
It will also show you how to write [_idiomatic code_](https://stackoverflow.com/questions/84102/what-is-idiomatic-code), that is, in a way that is natural for that language.

**Contribute**: You should actually contribute to the code base you use.
This is by far the most important advice for improving and I cannot overstate how instructive an experience this is.
By getting your code reviewed you get prompt and informative feedback on what you’re doing wrong and how you can do better.
It gives you the opportunity to try out what you’ve learned, learn something new, and improves your confidence in your ability.
Contributing to open source and seeing your features being used is also rewarding, and that starts a positive feedback loop where you feel like contributing more.
Further, when you start applying for jobs in industry people can see your work, and so know that you are good at what you do (I say this as a person who is now involved in reviewing these applications).

**Study**: Learning by experience is great but---at least for me---it takes a deliberate approach to formalise and cement new ideas.
Read well-reviewed books on your language (appropriate for your level) and reinforce what you learn by tackling more complex tasks and venturing [outside your comfort zone](https://www.geeksaresexy.net/2016/12/20/comfort-zone-comic/).
Reading blog posts and articles about the language is also a great idea.

**Ask for help:** Sometimes a bug just stumps you, or you just don’t know how to implement a feature.
In these circumstances, it’s quicker to reach out to experts who can help and maybe teach you something at the same time.
More often than not, someone has had the same problem or they’re happy to point you in the right direction.
I’m fortunate to work with Julia experts at Invenia, so when I have a problem they are always most helpful.
But posting on public fora like [Slack](https://slackinvite.julialang.org/), [Discourse](https://discourse.julialang.org/), or [StackOverflow](https://stackoverflow.com/) is an option we all have.

## Lesson 2: Software Engineering Practices
With respect to the environment and incentives in industry surrounding code maintainability, robustness, and testing, there are certain practices in place to encourage, enable, and ensure these qualities are met.
These key practices can turn a collection of scripts into a fully implemented package one can use and rely upon with high confidence.

While there are without doubt many universities and courses that teach these practices to their students, I find they are often neglected by coding novices and academics alike, to their own disadvantage.

**Take version control seriously:** [Git](https://git-scm.com/) is a programming staple for version control, and while it is tempting to disregard it when working alone, without it you soon find yourself creating [convoluted naming schemes](https://uidaholib.github.io/get-git/1why.html) for your files; frequently losing track of progress; and wasting time looking through email attachments for the older version of the code to replace the one you just messed up.

[Git](https://git-scm.com/) can be [a little intimidating](https://xkcd.com/1597/) to get started, but once you are comfortable with the basic commands (fetch, add, commit, push, pull, merge) and a few others (checkout, rebase, reset) you will never look back.
[GitHub](https://github.com/)’s utility, meanwhile, extends far beyond that of a programmatic hosting service; it provides [documentation hosting](https://guides.github.com/features/wikis/), [CI/CD pipelines](https://help.github.com/en/actions/building-and-testing-code-with-continuous-integration/about-continuous-integration), and many other features that enable efficient cross-party collaboration on an _enterprise_ scale.

It cannot be overstated how truly indispensable [Git](https://git-scm.com/) and [GitHub](https://github.com/) are when it comes to turning your code into functional packages, and the earlier you adopt these the better.
It also helps to know how [semantic versioning](https://semver.org/) works, so you will know what it means to increment a package version from 1.2.3 to 1.3 and why.

**Organise your code**: In terms of packaging your code, get to know the typical package folder structure.
Packages often contain src, docs, and test directories, as well as standard artefacts like a README, to explain what the package is about, and a list of dependencies, e.g.
Project and Manifest files in Julia, or requirements.txt in Python.
Implementing the familiar package structure keeps things organised and enables yourself and other users to navigate the contents more easily.

**Practice code hygiene**: This relates to the readability and maintainability of the code itself.
It’s important to practice [good hygiene](https://medium.com/@anishmahapatra/code-hygiene-dont-laugh-it-off-2a5aebcdd84b) if you want your code to be used, extended, and maintained by others.
Bad code hygiene will turn off other contributors---and eventually yourself---leaving the package unused and unmaintained.
Here are some tips for ensuring good hygiene:

* Take a **design-first** approach when creating your package.
Think about the intended user(s) and what their requirements are---this may be others in your research group or your future self.
Sometimes this can be difficult to know in advance but working iteratively is better than trying to capture all possible use cases at once.
* Think about how the [API](https://en.wikipedia.org/wiki/Application_programming_interface) should work and how it integrates with other packages or applications.
Are you building on something that already exists or is your package creating something entirely new?
* There should be a style guide for writing in the language, for example, [BlueStyle](https://github.com/invenia/BlueStyle/) in Julia and [PEP 8](https://www.python.org/dev/peps/pep-0008/) in Python.
You should adhere to it so that your code follows the same standard as everyone else.
* Give your variables and functions meaningful, and memorable names.
There is no advantage to obfuscating your code for the sake of brevity.
* Furthermore, read up on the language’s [Design Patterns](https://en.wikipedia.org/wiki/Software_design_pattern).
These are the common approaches or techniques used in the language, which you will recognise from reading the code.
These will help you write better, more idiomatic code.

**Write good documentation**: The greatest package ever written would never be used if nobody knew how it worked.
At the very least your code should be commented and a README accompanying the package explaining to your users (and your future self) what it does and how to install and use it.
You should also attach docstrings to all user-facing (aka public) functions to explain what they do, what inputs they take, what data types they return, etc.
This also applies to some internal functions, to remind maintainers (including you) what they do and how they are used.
Some minimum working examples of how to use the package features are also a welcome addition.

Lastly, documentation should evolve with the package; when the API changes or new use-cases get added these should be reflected in the latest documentation.

**Write good tests**: Researchers in computational fields might find familiar the practice of running “canonical experiments” or “reproducibility tests” that check if the code produces the correct result for some pipeline and is therefore “calibrated”.
But these don’t necessarily provide good or meaningful [test coverage](https://en.wikipedia.org/wiki/Code_coverage).
For instance, canonical experiments, by definition, test the software within the limits of its intended use.
This will not reveal latent bugs that only manifest under certain conditions, e.g.
when encountering corner cases.

To capture these you need to write adequate _Unit and Integration Tests_ that cover all expected corner cases to be reasonably sure your code is doing what it should.
Even then you can’t guarantee there isn’t a corner case you haven’t considered, but testing certainly helps.

If you do catch a bug it’s not enough to fix it and call it a day; you need to write a new test to replicate it and you will only have fixed the bug only when that new test passes.
This new test prevents [regressions](https://stackoverflow.com/questions/3464629/what-does-regression-test-mean) in behaviour if the bug ever returns.

## Lesson 3: Take Part in the Community
Undertaking a fraction of the points above would be more than enough to boost your ability to develop software.
But the return on investment is compounded by taking part in the community forums on [Slack](https://slackinvite.julialang.org/) and [Discourse](https://discourse.julialang.org/); joining organizations on [GitHub](https://github.com/); and attending [Meetups](https://www.meetup.com/London-Julia-User-Group/) and [conferences](https://juliacon.org/).
Taking part in a collaboration (and meeting your co-developers) fosters a strong sense of community that supports continual learning and encouragement to go and do great things.
In smaller communities related to a particular tool or niche language, you may even become well-known such that your potential future employer (or some of their engineers) are already familiar with who you are before you apply.

## Takeaway
Personal experience has taught me that the incentives in academic research can be qualitatively different from those in industry, despite the overlap they share.
However, the practices that are instilled in one track don’t necessarily translate off-the-shelf to the other, and switching gears between these (often competing) frameworks can initially induce an all-too-familiar sense of [imposter syndrome](https://ardalis.com/the-more-you-know-the-more-you-realize-you-dont-know).

It’s important to remember that what you learn and internalise in a PhD is, in a sense, “selected for” according to the incentives of that environment, as outlined above.
However, under the auspices of a supportive community and the proper guidelines, it’s possible to become more well-rounded in your skillset, as I have.
And while I still have much more to learn, it’s encouraging to reflect on what I have learned during my time at Invenia and share it with others.

Although this post could not possibly relay everything there is to know about software engineering, my hope is that simply being exposed to the lexicon will serve as a springboard to further learning.
To those looking down such a path, I say: you will make many many mistakes, as one always does at the outset of a new venture, but that’s all part of learning.


## Notes
[^1]:
     While these tips are language-agnostic, they would be particularly helpful for anyone interested in learning or improving with [Julia](https://julialang.org/).

[^2]:
    Examples of high quality packages include the [Requests](https://github.com/psf/requests) in Python, and [NamedDims.jl](https://github.com/invenia/NamedDims.jl) in Julia.
