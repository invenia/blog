---

layout: post

title: "Memristive dynamics as a heuristic optimization tool and as an interesting problem in graph theory"

author: "Ana Zegarac and Francesco Caravelli"

comments: false

tags: optimisation graph



---

$$ \newcommand{\Ron}{R_{\text{on}}} $$
$$ \newcommand{\Roff}{R_{\text{off}}} $$
$$ \newcommand{\dif}{\mathop{}\!\mathrm{d}} $$
$$ \newcommand{\coloneqq}{\mathrel{\vcenter{:}}=} $$

Introduction
------------

Before the advent of LED technology, our light bulbs were mainly composed of a tungsten filaments: as the currents flew in the circuit, the heat produced by the high current in the wires lighted the filament via the Joule effect. The Joule effect is present whenever a current flows in a resistor and tungsten has been employed because of its high melting point. Tungsten, among other metals such as Titanium, are currently living a second life at the nanoscale for the application in memristor technology. The purpose of this blog post is to enlighten how an exotic property of certain nanoscale components can be used for the most common of all computer science applications: optimization. This might seem as a huge logic jump: how can the Physics of nanoscale light bulbs be of any use in computer science?

Let us first provide a simple example based on resistors. Let us imagine a very complex network of resistances, where for simplicity we will consider all resistances alike.

<img src="{{ site.baseurl }}/public/images/tikz_cubecircuit.png" alt="circuit" style="width:300px;margin:0 auto 0 auto;" class="img-responsive">

If we connect points A and B of the network above to a battery, current will quickly flow through the conductors (let’s say, from A to B). The rules which the current and the voltages have to follow are Kirchhoff laws. The current will "explore” the possible space of flows and settle upon a value. At first it might seem hard to see which way the current should flow, but because of the symmetry of the problem clearly the current will flow equally in all directions to reach B. The reason is that the current splits at each intersection according to the inverse resistance rule: this implies that if we follow the maximal current at each intersection, we can use this protocol to actually solve the minimum distance (or cost) problem on a graph, which is a polynomial type of problem. This one example of a physical system performing analog computation. The first thing to notice is that there is a deep connection between graph theory and resistor networks, and in fact between resistor network and Markov chains (random walks on certain graph structures). Can this connection between resistor networks and computation be further explored?

In 2008, HP “discovered” the nanoscale memristor: as the the dimension of a resistance reaches the nanoscale, certain metal oxides such as tungsten or Titanium have an interesting property: the resistance changes as a function of time. For the math enthusiasts, the HP researchers noticed that for the case of Titanium Dioxide the resistance seemed to change between two limiting values according to a seemingly simple convex law depending on an internal parameter $$w$$ constrained between $$0$$ and $$1$$:

$$ R(w)=\Ron (1-w) +w \Roff,$$

Where they noticed:

$$ \frac{\dif}{\dif t} w(t)=\alpha w- \Ron \frac{I}{\beta}, $$

where $$I$$ is the current in the device. Albeit this model has been revised several times, it still serves as a prototypical model of a memory-resistor: the memristor. What about a network of memristors? In this case things get more complicated, and the differential equation becomes vectorial and more complicated [1,2]:

$$ \frac{\dif}{\dif t}\vec{w}(t)=\alpha\vec{w}(t)-\frac{1}{\beta} \left(I+\frac{\Roff-\Ron}{\Ron} \Omega W(t)\right)^{-1} \Omega \vec S(t), $$

where $$\Omega$$ is a matrix which contains the circuit topology, $$\vec S(t)$$ are the applied voltages and $$W$$ a diagonal matrix which contains the values of $$w$$ for each memristor in the network. We do not go into the details, but it is interesting to note that the equation above, similarly to the case of resistors, also tries to solve another optimization problem, but this time more complicated: the QUBO, or Quadratically Unconstrained Binary Optimization, heuristically. The solution(s) of the QUBO are all those values of a set of binary parameters $$w_i$$’s that minimize a function of the type:

$$F(\vec w)= -\frac p 2 \sum_{ij} w_i J_{ij} w_j + \sum_j h_j w_j $$

This is due to the fact that as the dynamics of certain type of memristors which are described by the equation above, tries to minimize a QUBO functional (in the language of dynamical systems, they possess a Lyapunov functional). For the case of realistic circuits, $$\Omega$$ has to be a very specific matrix which we will discuss later, but from the point of view of optimization purposes, the system of differential equations can be simulated for arbitrary $$\Omega$$ in principle. Thus, the memristive differential equation can serve as a heuristic method for a NP-Complete problem such as QUBO [3]. These problems are NP-Complete because there is no known algorithm that is better than exhaustive search: because of the binary nature of the variables, we necessarily have to explore all the $$2^N$$ possible values of the variables $$w$$’s to decide which extremum (or extrema) are better. In a certain sense, the memristive differential equation is a relaxation of the QUBO problem to continuous variables. For the purpose of this audience, one problem of interest might be the following: given the returns and covariance matrix between prices in a stock market, in which assets shall we invest and which shall we not? This is a bit different than the typical question of the “portion” of our capital we should invest in which asset, but if we should invest or not, which is a binary variable. Thus we use a modification of the Markowitz approach, but for binary variables, we aim to maximize (rather than minimize):

$$M(W)=\sum_i \left(r_i-\frac{p}{2}\Sigma_{ii} \right)W_i-\frac{p}{2} \sum_{i\neq j} W_i \Sigma_{ij} W_j$$

where $$r_i$$ are the returns and $$\Sigma$$ the covariance matrix. The mapping between the optimization of the returns above and the equivalent memristive equation is:

$$
\begin{align}
\Sigma &= \Omega, \\
\frac{\alpha}{2} +\frac{\alpha\xi}{3}\Omega_{ii} -\frac{1}{\beta} \sum_j \Omega_{ij} S_j &= r_i-\frac{p}{2}\Sigma_{ii},\\
\frac{p}{2} &= \alpha \xi.
\end{align}
$$

From which we can obtain the vector $$S$$ through inversion of the matrix $$\Sigma$$ if it is invertible. We still have the freedom of choosing $$\xi$$ and $$\alpha$$ freely given the constraint, but the two are slightly different in nature: $$\xi\gg 1$$ is the deep nonlinear regime, while $$\alpha\gg 1$$ is the deep diffusive regime. There are some conditions for this method to be suitable for heuristic optimization, which we do not go into the detail here (but are related to the spectral properties of the matrix $$J$$), but as a heuristic method this is rather easier than the exhaustive search: it requires a one time matrix inversion which scales as $$N^3$$, and the simulation of a first order differential equation which also requires a matrix inversion step by step, and thus scales as $$T \cdot N^3$$ where $$T$$ is the number of time steps (heuristically of the order of the hundreds for Euler integration methods). As a comparison $$2^{100}$$ is circa $$10^{30}$$, while $$100\times(100)^3$$ is of the order of $$100$$ millions, which a regular computer can handle. We propose two approaches to this problem. One is based on the most common Metropolis-Hasting algorithm and the other one on the heuristic “memristive” equation.

### MATLAB code

Below we included the MATLAB code for both algorithms.

Firstly, an example of input parameters and calls to functions:

{% highlight MATLAB %}
clc
clear

n = 20;

returns = rand(n,1) - 0.5;
Sigma = 1/n*rand(n);
p = 1;
W0 = rand(n,1);

[WfinalMH, EtotalMH] = MetropolisHastings(returns, Sigma, p, W0);
[WfinalMO, EtotalMO] = MemristiveOptimization(returns, Sigma, p, W0);
{% endhighlight %}

Then we have the Metropolis-Hastings algorithm:

 {% highlight MATLAB %}
 % The Metropolis-Hasting Monte Carlo annealing optimization code

 % Input:
 % returns = vector of returns for each variable (n x 1)
 % Sigma   = covariance matrix (n x n)
 % p       = parameter (scalar)
 % W0      = vector of initial conditions for variables Wi (n x 1)

 % Output:
 % Wfinal = values of Ws in the last iteration (n x 1)
 % Etotal = energy at every iteration (1 x total_time)


 % auxiliary function Energy evaluates the function we are trying to
 % minimize

 function [Wfinal, Etotal] = MetropolisHastings(returns, Sigma, p, W0)

 n = length(W0);                 % number of variables

 reg = 0.0001;                   % regularizer
 Sigma = reg*eye(n) + Sigma;     % regularised covariance matrix

 lambda = 0.999;                 % annealing factor

 eff_timesteps=1000;             % number of effective time steps
 temperature=100;                % temperature

 total_time = eff_timesteps * n; % number of iterations
 Etotal = zeros(1,total_time);   % energy at every time step

 Emin = Energy(returns,Sigma,p,W0); % initial energy
 W = W0;

 for t = 1:total_time
     temperature = temperature * lambda;
     rand_index = floor(rand*n) + 1;

     Wtemp = W;
     Wtemp(rand_index) = 1 - Wtemp(rand_index);
     Etemp = Energy(returns,Sigma,p,W);

     if (rand < 1/(1+exp(-(Emin-Etemp)/temperature)))
         Emin = Etemp;
         W = Wtemp;
     end

     Etotal(t) = Emin;

 end

 Wfinal = W;

 end


 function E = Energy(returns,Sigma,p,W)
 n = length(W);
 E = 0;

 for i = 1:n
     E = E + (returns(i) - p/2*Sigma(i,i))*W(i);

     for j=1:n
         if (j ~= i)
             E = E - p/2*Sigma(i,j)*W(j)*W(i);
         end
     end
 end

 end
{% endhighlight %}


And finally, the memristive optimisation code:

{% highlight MATLAB %}

% Memristive Optimization code

% Input:
% returns = vector of returns for each variable (n x 1)
% Sigma   = covariance matrix (n x n)
% p       = parameter (scalar)
% W0      = vector of initial conditions for variables Wi (n x 1)

% Output:
% Wfinal = values of Ws in the last iteration (n x 1)
% Etotal = energy at every iteration (1 x total_time)

% auxiliary function Energy evaluates the function we are trying to
% minimize

function [Wfinal,Etotal] = MemristiveOptimization(returns, Sigma, p, W0)

n = size(Sigma, 1);              % number of variables

alpha = 0.1;                    % nonlinearity parameter
xi = p/2 / alpha;
beta = 10;

reg = 0.0001;                   % regularizer
Sigma = reg * eye(n) + Sigma;     % regularising the covariance matrix

S = beta * inv(Sigma) * ...
    (alpha/2 * ones(n,1) + (p/2 + alpha*xi/3) * diag(Sigma) - returns);


% Euler integration

dt = 0.1;                       % time step
total_time = 3000;              % total time steps

Wtotal = zeros(n, total_time);
Wtotal(:,1) = W0;

Etotal = zeros(1, total_time);
Etotal(1) = Energy(returns, Sigma, p, Wtotal(:, 1));

for t = 1:total_time-1

    Wtotal(:,t + 1) = Wtotal(:, t) + ...
        dt*(alpha*Wtotal(:, t) - 1/beta * ...
        inv(eye(n) + xi * Sigma * diag(Wtotal(:, t))) * Sigma * S);

    % making sure W are never greater than 1 nor less than 0:
    Wtotal(Wtotal(:,t+1)>1, t+1)=1;
    Wtotal(Wtotal(:,t+1)<0, t+1)=0;

    Etotal(t + 1) = Energy(returns, Sigma, p, Wtotal(:, t + 1));

end

Wfinal = Wtotal(:, end);

end


function E = Energy(returns, Sigma, p, W)

E = -p/2 * W' * Sigma*W + (returns - p/2 * diag(Sigma))' * W;

end


{% endhighlight %}
The equation for memristors is also interesting per se!

Graph theory
------------

In the second half of this post, we will see how the geometry of a memristive circuit affects interactions between memristors.

For purely memristive circuits, we saw we get the following system of equations:

$$\frac{\dif\vec{W}(t)}{\dif{t}} = \alpha \vec W(t) - \frac 1 \beta (I + \xi \Omega W(t))^{-1} \Omega \vec S(t),$$

where $$\vec W$$ is the vector containing internal memory values of memristors in the circuit, $$\alpha, \beta, \xi$$ are constants, and $$\vec S$$ is the vector of voltages. We will first try to explain what $$\Omega$$ is in the next few sections. We start by introducing some concepts from graph theory.

### Graph theoretic introduction

A (directed) graph consists of two objects - vertices and edges. Vertices can be thought of as points and edges as lines that connect some of those points. We will label vertices as $$v_1, \dots, v_n$$ and edges as $$e_1, \dots, e_m$$. Mathematically, we represent an edge starting at vertex $$v_i$$ and ending at vertex $$v_j$$ as an ordered pair $$(v_i, v_j)$$.

We call a graph *planar* if it can be drawn in a plane without any of its edges intersecting. The picture below shows an example of a directed planar graph.

<img src="{{ site.baseurl }}/public/images/tikz-egplanargraph.png" alt="planar" style="width:300px;margin:0 auto 0 auto;" class="img-responsive">

### Incidence matrix; $$\Omega_{B^T}$$

Let $$G$$ be a directed graph. One way of representing $$G$$ is by specifying where each of its edges starts and where it ends. It is convenient to do this using a matrix. We call such a matrix an *incidence matrix*.

As an example, consider the graph labelled as in the picture below.

<img src="{{ site.baseurl }}/public/images/tikz-egplanargraph_labelled.png" alt="planarlabelled" style="width:300px;margin:0 auto 0 auto;" class="img-responsive">
<!--\label{fig:tikz-egplanargraph_labelled.png}-->

Each column of its incidence matrix represents an edge: the first edge starts at vertex $$1$$ and ends at vertex $$2$$, so the first column of the matrix has entry $$1$$ in the first row and entry $$-1$$ in the second row. All the other entries in the first column are $$0$$ because none of the other vertices are a part of that edge.
By continuing this process for every edge, we get the incidence matrix $$B$$ of the given graph:

$$
B =
\begin{pmatrix}
 1 &-1 &-1 & 0 & 0 & 0 & 0 & 0 \\  
-1 & 0 & 0 & 1 & 1 & 0 & 0 & 0 \\
 0 & 1 & 0 &-1 & 0 &-1 &-1 & 0 \\
 0 & 0 & 0 & 0 & 0 & 1 & 0 &-1 \\
 0 & 0 & 1 & 0 &-1 & 0 & 1 & 1 \\
\end{pmatrix}.
$$

More formally, if a graph $$G$$ has $$n$$ vertices and $$m$$ edges, then the incidence matrix $$B$$ of $$G$$ is an $$n \times m$$ matrix (i.e. a matrix with $$n$$ rows and $$m$$ columns), whose entry $$(i,j)$$ is defined as

$$
B_{ij} \coloneqq \begin{cases}
1 & \text{if $v_i$ is the initial vertex of the edge $e_j$,} \\
-1 & \text{if $v_i$ is the terminal vertex of the edge $e_j$,} \\
0 & \text{otherwise.}
\end{cases}
$$

If $$B$$ is an incidence matrix, we can define the projector operator $$\Omega_{B^T}$$:

$$
\label{eq:Omega_def}
\Omega_{B^T} = B^T\left(B B^T\right)^{-1} B.
$$

This is a *projector* operator because $$ \Omega_{B^T}^2 = \Omega_{B^T}. $$

If we try to compute $$\Omega_{B^T}$$ from the definition, we will find that the inverse $$\left(B B^T\right)^{-1}$$ does not exist in general. This can be solved by either considering the reduced incidence matrix (obtained by removing a row from the original incidence matrix) or by taking the pseudoinverse of the expression instead of the "regular" inverse.

### Cycle matrix; $$\Omega$$

Before we define the projector operator $$\Omega$$, let us discuss a few more objects from graph theory.

We define a *walk* on a directed graph $$G$$ to be a sequence of vertices, say $$v_1, v_2, \dots, v_n$$, such that for every pair of consecutive vertices in the sequence there exists an edge that connects them, i.e. for every $$i$$ such that $$1<i\leq n$$ either $$(v_{i-1}, v_i) \in E(G)$$ or $$(v_i, v_{i-1}) \in E(G)$$, where $$E(G)$$ is the edge set of the graph $$G$$.

An example of a walk on a graph is shown in orange in picture below.

<img src="{{ site.baseurl }}/public/images/tikz-egplanargraph_walk.png" alt="graphwalk" style="width:300px;margin:0 auto 0 auto;" class="img-responsive">

A *cycle* is a walk $$W = v_1 v_2 \dots v_n$$ such that $$l\geq 3$$, $$v_0 = v_n$$ and the vertices $$v_i$$, $$0<i<n$$ are distinct from each other and from $$v_0$$. An example of a cycle is shown in the image below.

<img src="{{ site.baseurl }}/public/images/tikz-egplanargraph_cycle.png" alt="egplanarcycle" style="width:300px;margin:0 auto 0 auto;" class="img-responsive">

Edge space has a structure of a vector space. A *cycle space* is the subset of edge space that is spanned by all cycles of a graph.

---

**On vector spaces**

The property of being a *vector space* is an abstract mathematical notion, but here we will try to explain it in an intuitive way. We can think of a vector space as a collection of arrows. The two key properties of a vector space (in terms of our arrows) are:

- if we add two arrows, we get an arrow:

<img src="{{ site.baseurl }}/public/images/tikz-vs_add.png" alt="vsadd" style="width:400px;margin:0 auto 0 auto;" class="img-responsive">

 - if we multiply an arrow by some number, we still end up with an arrow (possibly of different length):

<img src="{{ site.baseurl }}/public/images/tikz-vs_multiply.png" alt="vsmult" style="width:400px;margin:0 auto 0 auto;" class="img-responsive">

It is the same with cycles -- just that we think of cycles on a graph instead of arrows in Euclidean space. For example, if we add two cycles, we will get a cycle (figure below).

<img src="{{ site.baseurl }}/public/images/tikz-egplanargraph_cycle_add.png" alt="cyclesadd" style="width:600px;margin:0 auto 0 auto;" class="img-responsive">
<!--\label{fig:tikz-egplanargraph_cycle_add.png}-->

To make this even more clear, let us see what happens when we represent these vectors as column vectors.

In the case of "arrows", if the first row corresponds to the $$x$$-axis and the second row to the $$y$$-axis, the arrow addition example above written out in coordinates becomes:

$$
\begin{pmatrix}3 \\ 0\end{pmatrix} +
\begin{pmatrix}1 \\ 2\end{pmatrix} =
\begin{pmatrix}5 \\ 2\end{pmatrix}.
$$

In the case of cycles, if the $$i$$-th row corresponds the $$i$$-th edge of a graph and we label edges as in the figure above, the graph addition example becomes:<!-- In the case of cycles (shown in Figure \ref{fig:tikz-egplanargraph_cycle_add.png}), if the $$i$-th row corresponds the $$i$-th edge of a graph and we label edges as in Figure \ref{fig:tikz-egplanargraph_labelled.png}, we have: -->

$$
\begin{pmatrix}1\\0\\1\\0\\1\\0\\0\\0\end{pmatrix} +
\begin{pmatrix}0\\0\\0\\1\\-1\\0\\-1\\0\end{pmatrix} =
\begin{pmatrix}1\\0\\1\\1\\0\\0\\-1\\0\end{pmatrix}.
$$

One of the properties of a vector space is that there exists a set of vectors that can be used to express any other vector. We call that set a "basis". It is not necessarily a unique set.

---

A cycle matrix $$A$$ is a matrix whose columns form a basis of the cycle space. For example, if $$\{\vec c_1, \dots, \vec c_n\}$$ is a set of column vectors that form a basis of the cycle space, then the cycle matrix is

$$ A = (\vec c_1, \dots, \vec c_n). $$

Finally, the *projector operator $$\Omega$$ on the cycle space of the graph* is defined to be $$\Omega = A(A^T A)^{-1}A^T $$.

Locality
--------

### Locality in planar graphs

In [5] the following bound on locality of interactions in planar graphs was proved:

$$ |\Omega_{i,j}| \leq e^{-z \text{d}(i,j) + \tilde \rho}. $$

For the purpose of this blog post, we can think of $$z$$, $$\tilde \rho$$ as constants and of $$\text{d}(i,j)$$ as the distance between edges $$i$$ and $$j$$.

Full derivation can be found in [5]. Here we will focus on one of the key parts of the calculation.

Finding an analytic expression for a quantity which involves an inverse of a potentially large matrix is a non-trivial problem. To overcome this in the case of $$\Omega = A(A^T A)^{-1}A^T$$, the author noticed that the expression for $$\Omega$$ simplifies if we orthonormalise $$A$$ first. If we denote by $$\tilde A$$ the orthonormalised matrix, we get $$\Omega = \tilde A \tilde A^{-1}$$.

To obtain the orthonormalised matrix $$\tilde A$$ from $$A$$, a variation of Gram-Schmidt process was used, where the $$p$$-th column, $$A_p$$, of the orthonormalised matrix $$\tilde A$$ is given by

$$
\tilde A_p =
\det \begin{pmatrix}
\langle \vec A_1, \vec A_1 \rangle & \langle \vec A_1, \vec A_2 \rangle & \dots & \langle \vec A_n, \vec A_n \rangle & \vec A_1\\
\vdots & \vdots & \ddots & \vdots & \vdots\\
\langle \vec A_n, \vec A_1 \rangle & \langle \vec A_n, \vec A_2 \rangle & \dots & \langle \vec A_n, \vec A_n \rangle & \vec A_n\\
\end{pmatrix}.
$$

We will now show how the inner products $$\langle A_i, A_j \rangle$$ can be expressed in terms of an adjacency matrix.

Consider the following grid graph $$G$$:

<img src="{{ site.baseurl }}/public/images/tikz-gridGraph3.png" alt="gridgraph" style="width:400px;margin:0 auto 0 auto;" class="img-responsive">

Pick a basis of the cycle space of $$G$$ as shown below.

<img src="{{ site.baseurl }}/public/images/tikz-gridGraph3allcycles.png" alt="gridgraphbasis" style="width:400px;margin:0 auto 0 auto;" class="img-responsive">

Now let us denote by $$G'$$ the graph that has a vertex for each basis cycle of $$G$$ and an edge between two vertices if the corresponding cycles in $$G$$ are adjacent.

We can then express the inner products as

$$
\langle A_i, A_j \rangle =
\begin{cases}
M_{ij} \qquad \text{if $i = j$} \\
|C_{i}| \qquad \text{if $i = j$}
\end{cases},
$$

where $$M_{ij}$$ is the $$(i,j)$$-th entry of the adjacency matrix of $$G'$$ and
$$|{C_{i}}|$$
is the length of the cycle corresponding to the $$i$$-th vertex of $$G'$$.

This was the key step that allowed manipulations which ultimately lead to the expression for the bound in [5].

### Locality in nonplanar graphs

We found that the same method cannot be applied to the nonplanar graphs.

In the case of planar graphs, it was possible to choose the basis of the cycle space in such a way that the basis cycles bound the faces of the graph.

A generalisation of this construction to nonplanar graphs is not obvious due to the fact that in nonplanar graphs the notion of *faces* is not well-defined. In an attempt to overcome this, we defined the faces of a nonplanar graph $$G$$ to be the faces of an embedding of $$G$$ in some closed orientable surface (we can think of an embedding of $$G$$ in a surface as a "drawing of $$G$$ on that surface" such that no two edges of $$G$$ intersect). For a given graph $$G$$, it is always possible to find such a surface [7].

We illustrate this on the simplest nonplanar graph, $$K_{3,3}$$:

<img src="{{ site.baseurl }}/public/images/tikz-K33switched.png" alt="K33switched" style="width:300px;margin:0 auto 0 auto;" class="img-responsive">

An embedding of $$K_{3,3}$$ in a torus is shown below. We thus define the faces of $$K_{3,3}$$ to be the connected components of $$T^2 \setminus \bar K_{3,3}$$, where $$\bar K_{3,3}$$ denotes the embedding of $$K_{3,3}$$ inside the torus $$T^2$$ (here $$T^2 \setminus \bar K_{3,3}$$ means that we 'subtract' the embedding of $$K_{3,3}$$ from the torus).

<img src="{{ site.baseurl }}/public/images/K33torus_ppt.png" alt="K33torus" style="width:300px;margin:0 auto 0 auto;" class="img-responsive">

To be able to use the method from [5], we would have to form a basis of the cycle space using only cycles that bound faces.

However, the number of elements in the basis of a cycle space of $$K_{3,3}$$ is four (see [6] for a way of calculating the dimension of a cycle space). As can be seen in the figure above, there are only three faces in the embedding of $$K_{3,3}$$ in a torus. This is a problem.

### Formal discussion in a more general setting

Setting aside the question of whether the faces are well defined in the way described above, it might seem that if we embed $$K_{3,3}$$ in a different way or in some other closed orientable surface (for example a triple torus shown below), we could get sufficient number of faces. We prove below that there does not exist an embedding with sufficient number of faces.

<img src="{{ site.baseurl }}/public/images/tripletorus.png" alt="tripletorus" style="width:300px;margin:0 auto 0 auto;" class="img-responsive">

Let us define the *Euler characteristic* of a graph $$G$$ embedded in a surface $$S$$ as

$$\chi_{G, S} = |V| - |E| + |F|, $$

where
$$|V|$$,
$$|E|$$ and
$$|F|$$
denote the number of vertices, edges and faces, respectively.

To ensure we have enough basis cycles which bound faces, we want the number of faces to be greater than or equal to the dimension of the cycle space, that is:

$$ |F| \geq \text{dim} \ \mathcal{C} = |E| - (|V| - 1), $$

where $$\mathcal C$$ denotes the cycle space (the equality above follows from the calculation of the dimension of a cycle space described in [6]).

Substituting into the expression for Euler characteristic, we get

$$
\begin{align}
\chi_{G, S} &= |V| - |E| + |F| \\
&\geq |V| - |E| + \left(|E| - (|V| - 1)\right).
\end{align}
$$

That is, we need $$\chi_{G, S} \geq 1$$.

A graph embedded in a surface has Euler characteristic equal to the Euler characteristic of that surface.
If we denote by $$g$$ the genus of a surface (roughly speaking, a number of holes the surface has), we can also express the Euler characteristic in the following way:

$$\chi_{G,S} = 2 - 2g. $$

Hence, using the previously obtained inequality we get $$g \leq 1/2$$. This implies we can only find sufficiently many basis cycles that bound faces in surfaces with less that $$1/2$$ holes. Only such closed orientable surface is a sphere (with $$0$$ holes) and sphere is equivalent to a plane for all our purposes.

Therefore the approach described in [5] only applies to the planar case.

---

-	[1] https://arxiv.org/abs/1608.08651
-	[2] https://arxiv.org/abs/1611.02104
-	[3] https://arxiv.org/abs/1712.07046
-	[4] For a recent review, see https://engrxiv.org/c4qr9
-	[5] https://arxiv.org/abs/1705.00244
-	[6] Bollobás, B., 2012. Graph theory: an introductory course (Vol. 63). Springer Science & Business Media.
-	[7] http://www.gipsa-lab.fr/~francis.lazarus/Enseignement/compuTopo3.pdf
