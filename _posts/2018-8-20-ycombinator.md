---
layout: post

title: "A Short Note on The Y Combinator"

tags: python, programming

author: Wessel Bruinsma

---

_This is a cross-post with [https://wesselb.github.io/2018/08/16/y-combinator.html](https://wesselb.github.io/2018/08/16/y-combinator.html)._

## Introduction
This post is a short note on the notorious _Y combinator_.
No, not [that company](https://ycombinator.com), but the computer sciency objects that looks like this:

$$ \label{eq:Y-combinator}
    Y = \lambda\, f : (\lambda\, x : f\,(x\, x))\, (\lambda\, x : f\,(x\, x)).
$$

Don't worry if that looks complicated; we'll get down to some examples and the nitty gritty details in just a second.
But first, _what_ even is this Y combinator thing?
Simply put, the Y combinator is a higher-order function $$Y$$ that can be used to define recursive functions in languages that don't support recursion.
Cool!

For readers unfamiliar with the above notation, the right-hand side of Equation \eqref{eq:Y-combinator} is a _lambda term_, which is a valid expression in [_lambda calculus_](https://en.wikipedia.org/wiki/Lambda_calculus):

1. $$x$$, a variable, is a lambda term;
2. if $$t$$ is a lambda term, then the anonymous function $$\lambda\, x : t$$ is a lambda term; 
3. if $$s$$ and $$t$$ are lambda terms, then $$s\, t$$ is a lambda term, which should be interpreted as $$s$$ applied with argument $$t$$; and
4. nothing else is a lambda term.

For example, if we apply $$\lambda\, x : y\,x$$ to $$z$$, we find

$$ \label{eq:example}
    (\lambda\, x : y\,x)\, z = y\,z.
$$

Although the notation in Equation \eqref{eq:example} suggests multiplication, note that everything is function application, because really that's all there is in lambda calculus.

Consider the factorial function $$\code{fact}$$:

$$ \label{eq:fact-recursive}
    \code{fact} =
        \lambda\, n :
            (\code{if}\,
                (\code{iszero}\, n) \,
                1 \,
                (\code{multiply}\,
                    n\,
                    (\code{fact}\,
                        (\code{subtract}\, n\, 1)))).
$$

In words, if $$n$$ is zero, return $$1$$; otherwise, multiply $$n$$ with $$\code{fact}(n-1)$$.
Equation \eqref{eq:fact-recursive} would be a valid expression if lambda calculus would allow us to use $$\code{fact}$$ in the definition of $$\code{fact}$$.
Unfortunately, it doesn't.
Tricky.
Let's replace the inner $$\code{fact}$$ by a variable $$f$$:

$$ 
    \code{fact}' =
        \lambda\, f: \lambda\, n :
            (\code{if}\,
                (\code{iszero}\, n) \,
                1 \,
                (\code{multiply}\,
                    n\,
                    (f\,
                        (\code{subtract}\, n\, 1)))).
$$

Now, crucially, the Y combinator $$Y$$ is precisely designed to construct $$\code{fact}$$ from $$\code{fact}'$$:

$$
    Y\, \code{fact}' = \code{fact}.
$$

To see this, let's denote $$\code{fact2}=Y\,\code{fact}'$$ and verify that $$\code{fact2}$$ indeed equals $$\code{fact}$$:

\begin{align} 
    \code{fact2}
    &= Y\, \code{fact}' \\\
    &= (\lambda\, f : (\lambda\, x : f\,(x\, x))\, (\lambda\, x : f\,(x\, x)))\, \code{fact}' \\\ 
    &= (\lambda\, x : \code{fact}'\,(x\, x) )\, (\lambda\, x : \code{fact}'\,(x\, x)) \label{eq:step-1} \\\ 
    &= \code{fact}'\, ((\lambda\, x : \code{fact}'\, (x\, x))\,(\lambda\, x : \code{fact}'\, (x\, x))) \label{eq:step-2} \\\ 
    &= \code{fact}'\, (Y\, \code{fact}') \\\
    &= \code{fact}'\, \code{fact2},
\end{align}

which is _exactly_ what we're looking for, because the first argument to $$\code{fact}'$$ should be the actual factorial function, $$\code{fact2}$$ in this case.
Neat!

We hence see that $$Y$$ can indeed be used to define recursive functions in languages that don't support recursion.
Where does this magic come from, you say?
Sit tight, because that's up next!

## Deriving the Y Combinator

This section introduces a simple trick that can be used to derive Equation \eqref{eq:Y-combinator}.
We also show how this trick can be used to derive analogues of the Y combinator that implement _mutual recursion_ in languages that don't even support simple recursion.

Again, let's start out by considering a recursive function:

$$ 
    f = \lambda\, x:g[f, x]
$$

where $$g$$ is some lambda term that depends on $$f$$ and $$x$$.
As we discussed before, such a definition is not allowed.
However, pulling out $$f$$,

$$ \label{eq:fixed-point}
    f = \underbrace{(\lambda \, f' :\lambda\, x:g[f', x])}_{h}\,\, f = h\, f.
$$

we do find that $$f$$ is a _fixed point_ of $$h$$: $$f$$ is invariant under applications of $$h$$.
Now---and this is the trick---suppose that $$f$$ is the result of a function $$\hat{f}$$ applied to itself: $$f=\hat{f}\,\hat{f}$$.
Then Equation \eqref{eq:fixed-point} becomes

$$ 
    \color{red}{\hat{f}} \,\hat{f}
    = h\,(\hat{f}\, \hat{f})
    = (\color{red}{\lambda\,x:h(x\,x)})\,\,\hat{f},
$$

from which we, by inspection, infer that

$$ 
    \hat{f} = \lambda\,x:h(x\,x).
$$

Therefore,

$$ 
    f
    = \hat{f}\hat{f}
    = (\lambda\,x:h(x\,x))\,(\lambda\,x:h(x\,x)).
$$

Pulling out $$h$$,

$$ 
    f
    = (\lambda\, h': (\lambda\,x:h'\,(x\,x))\,(\lambda\,x:h'\,(x\,x)))\, h
    = Y\, h,
$$

where suddenly a wild Y combinator has appeared.

The above derivation shows that $$Y$$ is a _fixed-point combinator_.
Passed some function $$h$$, $$Y\,h$$ gives a fixed point of $$h$$:
$$f = Y\,h$$ satisfies $$f = h\,f$$.

Pushing it even further, consider two functions that depend on each other:

\begin{align} 
    f &= \lambda\,x:k_f[x, f, g], &
    g &= \lambda\,x:k_g[x, f, g]
\end{align}

where $$k_f$$ and $$k_g$$ are lambda terms that depend on $$x$$, $$f$$, and $$g$$.
This is foul play, as we know.
We proceed as before and pull out $$f$$ and $$g$$:

\begin{align} 
    f
    = \underbrace{
        (\lambda\,f':\lambda\,g':\lambda\,x:k_f[x, f', g'])
    }_{h_f} \,\, f\, g
    = h_f\, f\, g
\end{align}

\begin{align}    
    g
    = \underbrace{
        (\lambda\,f':\lambda\,g':\lambda\,x:k_g[x, f', g'])
    }_{h_g} \,\, f\, g
    = h_g\, f\, g.
\end{align}

Now---here's that trick again---let $$f = \hat{f}\,\hat{f}\,\hat{g}$$ and $$g = \hat{g}\,\hat{f}\,\hat{g}$$.[^1]
Then

\begin{align} 
    \hat{f}\,\hat{f}\,\hat{g}
    &= h_f\,(\hat{f}\,\hat{f}\,\hat{g})\,(\hat{g}\,\hat{f}\,\hat{g})
    = (\lambda\,x:\lambda\,y:h_f\,(x\,x\,y)\,(y\,x\,y))\,\,\hat{f}\,\hat{g},\\\ 
    \hat{g}\,\hat{f}\,\hat{g}
    &= h_g\,(\hat{f}\,\hat{f}\,\hat{g})\,(\hat{g}\,\hat{f}\,\hat{g})
    = (\lambda\,x:\lambda\,y:h_g\,(x\,x\,y)\,(y\,x\,y))\,\,\hat{f}\,\hat{g},
\end{align}

which suggests that

\begin{align} 
    \hat{f} &= \lambda\,x:\lambda\,y:h_f\,(x\,x\,y)\,(y\,x\,y), \\\ 
    \hat{g} &= \lambda\,x:\lambda\,y:h_g\,(x\,x\,y)\,(y\,x\,y).
\end{align}

Therefore

\begin{align} 
    f
    &= \hat{f}\,\hat{f}\,\hat{g} \\\ 
    &=
        (\lambda\,x:\lambda\,y:h_f\,(x\,x\,y)\,(y\,x\,y))\,
        (\lambda\,x:\lambda\,y:h_f\,(x\,x\,y)\,(y\,x\,y))\,
        (\lambda\,x:\lambda\,y:h_g\,(x\,x\,y)\,(y\,x\,y)) \\\ 
    &= Y_f\, h_f\, h_g
\end{align}

where

$$
    Y_f = (\lambda\, h_f':
         \lambda\, h_g':
        (\lambda\,x:\lambda\,y:h_f'\,(x\,x\,y)\,(y\,x\,y))\,
        (\lambda\,x:\lambda\,y:h_f'\,(x\,x\,y)\,(y\,x\,y))\,
        (\lambda\,x:\lambda\,y:h_g'\,(x\,x\,y)\,(y\,x\,y))).
$$

Similarly,

$$ 
    g = Y_g\, h_f\, h_g.
$$

_Dang_, laborious, but that worked.
And thus we have derived two analogues $$Y_f$$ and $$Y_g$$ of the Y combinator that implement mutual recursion in languages that don't even support simple recursion.


## Implementing the Y Combinator in Python

Well, that's cool and all, but let's see whether this Y combinator thing actually works.
Consider the following nearly 1-to-1 translation of $$Y$$ and $$\code{fact}'$$ to Python:

```python
Y = lambda f: (lambda x: f(x(x)))(lambda x: f(x(x)))
fact = lambda f: lambda n: 1 if n == 0 else n * f(n - 1)
```

If we try to run this, we run into some weird recursion:

```python
>>> Y(fact)(4)
RecursionError: maximum recursion depth exceeded
```

Eh?
What's going?
Let's, for closer inspection, once more write down $$Y$$:

$$
    Y = \lambda\, f: (\lambda\, x : f\,(x\, x))\, (\lambda\, x : f\,(x\, x)).
$$

After $$f$$ is passed to $$Y$$, $$(\lambda\, x : f\,(x\, x))$$ is passed to $$(\lambda\, x : f\,(x\, x))$$; which then evaluates $$x\, x$$, which passes $$(\lambda\, x : f\,(x\, x))$$ to $$(\lambda\, x : f\,(x\, x))$$; which then again evaluates $$x\, x$$, which again passes $$(\lambda\, x : f\,(x\, x))$$ to $$(\lambda\, x : f\,(x\, x))$$; _ad infinitum_.
Written down differently, evaluation of $$Y\, f\, x$$ yields

$$
    Y\, f\, x
    = (Y\, f)\, x
    = (Y\, (Y\, f))\, x
    = (Y\, (Y\, (Y\, f)))\, x
    = (Y\, (Y\, (Y\, (Y\, f))))\, x 
    = \ldots,
$$

which goes on indefinitely.
Consequently, $$Y\, f$$ will not evaluate in finite time, and this is the cause of the `RecursionError`.
But we can fix this, and quite simply so: only allow the recursion---the $$x\,x$$ bit---to happen when it's passed an argument; in other words, replace

$$ \label{eq:strict-evaluation}
    x\,x \to \lambda\,y:x\,x\,y.
$$

Subsituting Equation \eqref{eq:strict-evaluation} in Equation \eqref{eq:Y-combinator}, we find

$$ \label{eq:strict-Y-combinator}
    Y = \lambda\, f : (\lambda\, x : f(\lambda\, y: x\, x\,y))\, (\lambda\, x : f(\lambda\, y:x\, x\, y)).
$$

Translating to Python,

```python
Y = lambda f: (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))
```

And then we try again:

```python
>>> Y(fact)(4)
24

>>> Y(fact)(3)
6

>>> Y(fact)(2)
2

>>> Y(fact)(1)
1
```

Sweet success!

## Summary
To recapitulate, the Y combinator is a higher-order function that can be used to define recursion---and even mutual recursion---in languages that don't support recursion.
One way of deriving $$Y$$ is to assume that the recursive function under consideration $$f$$ is the result of some other function $$\hat{f}$$ applied to itself: 
$$f = \hat{f}\,\hat{f}$$;
after some simple manipulation, the result can then be determined by inspection.
Although $$Y$$ can indeed be used to define recursive functions, it cannot be applied literally in a contemporary programming language; recursion errors might then occur.
Fortunately, this can be fixed simply by letting the recursion in $$Y$$ happen when needed---that is, _lazily_.

[^1]: Do you see why this is the appropriate generalisation of letting $$f=\hat{f}\,\hat{f}$$?
