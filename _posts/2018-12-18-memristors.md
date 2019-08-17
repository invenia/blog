---

layout: post

title: "Memristive dynamics: a heuristic optimization tool and problem in graph theory"

author: "Ana Zegarac and Francesco Caravelli"

comments: false

tags: optimization, graph theory, memristors



---

$$ \newcommand{\Ron}{R_{\text{on}}} $$
$$ \newcommand{\Roff}{R_{\text{off}}} $$
$$ \newcommand{\dif}{\mathop{}\!\mathrm{d}} $$
$$ \newcommand{\coloneqq}{\mathrel{\vcenter{:}}=} $$

Introduction
------------

In this post we will explore an interesting area of modern circuit theory and computing: how exotic properties of certain nanoscale circuit components (analogous to tungsten filament lightblulbs) can be used for solving certain optimization problems. Let's start with a simple example: imagine a circuit made up of resistors, which all have the same resitance.

<img src="{{ site.baseurl }}/public/images/tikz_cubecircuit.png" alt="circuit" style="width:300px;margin:0 auto 0 auto;" class="img-responsive">

If we connect points A and B of the circuit above to a battery, current will flow through the conductors (let’s say from A to B), according to Kirchhoff's laws. Because of the symmetry of the problem, the current splits at each intersection according to the inverse resistance rule, and so the currents will flow equally in all directions to reach B. This implies that if we follow the maximal current at each intersection, we can use this circuit to solve the minimum distance (or cost) problem on a graph. This can be seen as an example of a physical system performing analog computation. More generally, there is a deep connection between graph theory and resistor networks.

In 2008, HP discovered what is now called the "nanoscale memristor". This is a type of resistor made of certain metal oxides such as tungsten or titanium whose resistance, as its physical dimensions reach the nanoscale, changes as a function of time. For the case of titanium dioxide, the resistance changes between two limiting values according to a seemingly simple convex law depending on an internal parameter $$w$$ constrained between $$0$$ and $$1$$:

$$ R(w)=\Ron (1-w) +w \Roff,$$

where,

$$ \frac{\dif}{\dif t} w(t)=\alpha w- \Ron \frac{I}{\beta}, $$

and $$I$$ is the current in the device. Even though this simple model has been revised several times, it still serves as a prototypical model of a memory-resistor, or \textit{memristor}. 

In the general case of a network of memristors, the differential equation becomes [1,2]:

$$ \frac{\dif}{\dif t}\vec{w}(t)=\alpha\vec{w}(t)-\frac{1}{\beta} \left(I+\frac{\Roff-\Ron}{\Ron} \Omega W(t)\right)^{-1} \Omega \vec S(t), $$

where $$\Omega$$ is a matrix representing the circuit topology, $$\vec S(t)$$ are the applied voltages and $$W$$ a diagonal matrix of the values of $$w$$ for each memristor in the network. 

It is interesting to note that the equation above, similarly to the case of a circuit of simple resistors, is related to an optimization problem: quadratically unconstrained binary optimization (QUBO). The solution(s) of a QUBO problem are the set of binary parameters, $$w_i$$, that minimize a function of the form:

$$F(\vec w)= -\frac p 2 \sum_{ij} w_i J_{ij} w_j + \sum_j h_j w_j. $$

This is because the dynamics of memristors which are described by the equation above are such that a QUBO functional is minimized (in the language of dynamical systems, there exists a Lyapunov functional). For the case of realistic circuits, $$\Omega$$ has to be a very specific matrix which we will discuss later, but from the point of view of optimization theory, the system of differential equations can be simulated for arbitrary $$\Omega$$ in principle. Therefore, the memristive differential equation can serve as a \textit{heuristic} solution method for an NP-Complete problem such as QUBO [3]. These problems are NP-Complete because there is no known algorithm that is better than exhaustive search: because of the binary nature of the variables, in the worst case we have to explore all $$2^N$$ possible values of the variables $$w$$’s to determine the extreme of the problem. In a sense, the memristive differential equation is a relaxation of the QUBO problem to continuous variables. 

Next let's look at an application to a standard problem: given the expected returns and covariance matrix between prices of some financial assets, which assets should an investor allocate capital to? This setup, with binary decision variables, is different than the typical question of portfolio allocation, where the decision variables are real valued (fractions of wealth to allocate to assets). More formally, the objective is to maximize:

$$M(W)=\sum_i \left(r_i-\frac{p}{2}\Sigma_{ii} \right)W_i-\frac{p}{2} \sum_{i\neq j} W_i \Sigma_{ij} W_j,$$

where $$r_i$$ are the expected returns and $$\Sigma$$ the covariance matrix. The mapping between the above and the equivalent memristive equation is given by:

$$
\begin{align}
\Sigma &= \Omega, \\
\frac{\alpha}{2} +\frac{\alpha\xi}{3}\Omega_{ii} -\frac{1}{\beta} \sum_j \Omega_{ij} S_j &= r_i-\frac{p}{2}\Sigma_{ii}.\\
\frac{p}{2} &= \alpha \xi.
\end{align}
$$

The solution vector $$S$$ is obtained through inversion of the matrix $$\Sigma$$ (if it is invertible). We still have the freedom of choosing $$\xi$$ and $$\alpha$$ freely given the constraint, but the two are slightly different in nature: $$\xi\gg 1$$ is the deep nonlinear regime, while $$\alpha\gg 1$$ is the deep diffusive regime. There are some conditions for this method to be suitable for heuristic optimization (related to the spectral properties of the matrix $$J$$), but as a heuristic method, this is much cheaper than exhaustive search: it requires a one time matrix inversion which scales as $$N^3$$, and the simulation of a first order differential equation which also requires a matrix inversion step by step, and thus scales as $$T \cdot N^3$$ where $$T$$ is the number of time steps. As a comparison $$2^{100}$$ is $$O(10^{30})$$, while $$100\times(100)^3$$ is $$O(10^{8})$$. We propose two approaches to this problem. One is based on the most common Metropolis-Hasting algorithm and the other on the heuristic memristive equation.


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


---

-	[1] F. Caravelli, F. L. Traversa, M. Di Ventra, Phys. Rev. E 95, 022140 (2017) - https://arxiv.org/abs/1608.08651
-	[2] F. Caravelli, 	International Journal of Parallel, Emergent and Distributed Systems, 1-17 (2017) - https://arxiv.org/abs/1611.02104
-	[3] F. Caravelli, Entropy 2019, 21(8), 789 - https://www.mdpi.com/1099-4300/21/8/789, https://arxiv.org/abs/1712.07046
-	[4] For a recent review, see F. Caravelli and J. P. Carbajal, Technologies 2018, 6(4), 118 - https://engrxiv.org/c4qr9
-	[5] F. Caravelli, Phys. Rev. E 96, 052206 (2017) - https://arxiv.org/abs/1705.00244
-	[6] Bollobás, B., 2012. Graph theory: an introductory course (Vol. 63). Springer Science & Business Media.
-	[7] http://www.gipsa-lab.fr/~francis.lazarus/Enseignement/compuTopo3.pdf
-	[8] A. Zegarac, F. Caravelli, EPL 125 10001, 2019 -  https://iopscience.iop.org/article/10.1209/0295-5075/125/10001/pdf
