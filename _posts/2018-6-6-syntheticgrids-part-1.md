---

layout: post

title: "SyntheticGrids.jl: Part 1"
----------------------------------

Background
----------

It should come as no surprise that electricity plays a vital role in many aspects of modern life. From reading this article, to running essential hospital equipment, or powering your brand-new Tesla, many things that we take for granted would not be possible without the generation and transmission of electrical power. This is only possible due to extensive power grids, which connect power producers with consumers through a very complex network of towers, transmission lines, transformers etc. Needless to say, it is important to understand the peculiarities of these systems in order to avoid large scale blackouts, or your toaster burning out due to a fluctuation in the current.

Power grid research requires testing in realistic, large-scale, electric networks. Real power grids may have tens of thousands of nodes (also called buses), interconnected by multiple power lines each, spanning hundreds of thousands of square kilometers. In light of security concerns, most information on these power grids is considered sensitive and is not available to the general public or to most researchers. This has led to most power transmission studies being done using only a few publicly available test grids [^1], [^2]. These test grids tend to be too small to capture the complexity of real grids, severely limiting the practical applications of such research. With this in mind, there has recently been an effort in developing methods for building realistic synthetic grids, based only on publicly available information. These synthetic grids are based on real power grids and present analogous statistical properties---such as the geographic distribution of load and generation, total load, and generator types---while not actually exposing potentially sensitive information about a real grid.

The pioneers in treating power grids as networks were Watts and Strogatz[^3], when they pointed out that electric grids share similarities with *small-world networks*: networks that are highly clustered, but exhibit small characteristic path lengths due to a few individual nodes being directly connected to distant nodes (see Figure 1). This type of network is very useful in explaining social networks--- [see six degrees of Kevin Bacon](https://en.wikipedia.org/wiki/Six_Degrees_of_Kevin_Bacon)---but, despite similarities, power grids differ from small-world networks [^4],[^5]. If you are looking for an extensive list of studies on power grids, Pagani and Aiello [^6] is a good place to start.

![Figure 1: Some examples of different network topologies containing 20 nodes and 40 edges. (a) Small-world; (b) random; (c) scale-free (exponent 2). For more information, see Watts and Strogatz[^3].](Networks.png)

In order to study the dynamic properties of electric grids, some research has adopted simplified topologies, such as tree structures [^7] or ring structures [^8], which may fail to capture relevant aspects of the system. Efforts to build complete and realistic synthetic grids are a much more recent phenomenon. The effort of two teams is particularly relevant for this post, namely, Overbye's team [^9],[^10],[^11] and Soltan and Zussman [^12].

Considering the potential impact of synthetic grids in the study of power grids and the recency of these approaches, we at Invenia Labs have developed [SyntheticGrids.jl](https://github.com/invenia/SyntheticGrids.jl), an open source [Julia](https://julialang.org/) package. The central idea of SyntheticGrids.jl is to provide a standalone and easily expandable framework for generating synthetic grids, adopting assumptions based on research by Overbye's and Zussman's teams. Currently, it only works for grids within the territory of the US, but it should be easily extendable to other regions, provided there is similar data available.

There are two key sources of data for the placement of loads and generators: [USA census data](https://www.census.gov/geo/maps-data/data/gazetteer2010.html) and [EIA generator survey data](https://www.eia.gov/electricity/data/eia860/index.html). The former is used to locate and size loads, while the latter is used for generators. Since there is no sufficiently granular location-based consumption data available, loads are built based on population patterns. Load has a nearly linear correlation with population size [^12], so we adopt census population as a proxy for load. Further, loads are sited at each zip code location available in the census data. When placing generators, the EIA data provides us with all the necessary information, including geographic location, nameplate capacity, technology type, etc. This procedure is completely deterministic, since we want be as true as possible to the real grid structure, i.e. we want to use an unaltered version of the real data.

Our package treats power grids as a collection of buses connected by transmission lines. Buses can be either *load* or *generation* buses. Each generation bus represents a different power plant, so it may group several distinct generators together. Further, buses can be combined into *substations*, providing a coarse-grained description of the grid.

The coarse-graining of the buses into substations, if desired, is done via a simple hierarchical clustering procedure, as proposed by Birchfield *et al.* [^9]. This stochastic approach starts with each bus being its own cluster. At each step, the two most similar clusters (determined by the similarity measure of choice) are fused into one, and these steps continue until a stopping criterion has been reached. This allows the grouping of multiple load and generator units, similarly to what is actually done by Independent System Operators ([ISOs](https://en.wikipedia.org/wiki/Regional_transmission_organization_(North_America))).

In contrast to loads and generators, there is no publicly available data on transmission lines, so we have to adopt heuristics. The procedure implemented in the package is based on that proposed by Soltan and Zussman [^12]. It adopts several realistic considerations in order to stochastically generate the whole transmission network, which are summarised in the following three main principles:

> The degree distributions of power grids are very similar to those of scale-free networks [see: https://en.wikipedia.org/wiki/Scale-free_network], but grids have less degree 1 and 2 nodes and do not have very high degree nodes.
>
> It is inefficient and unsafe for the power grids to include very long lines.
>
> Nodes in denser areas are more likely to have higher degree.

Currently, SyntheticGrids.jl allows its generated grids to be directly exported to pandapower, a Python-based powerflow package. Soon, an interface with PowerModels.jl, a Julia-based powerflow package, will also be provided.

In the [second part](SynGrids_p2.md) we will go over how to use the main features of the package.

References
----------

[^1]: Power Systems Test Case Archive (UWEE) - http://www2.ee.washington.edu/research/pstca/

[^2]: Power Cases} - Illinois Center for a Smarter Electric Grid (ICSEG) - http://icseg.iti.illinois.edu/power-cases

[^3]: Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of ‘small-world’ networks. nature, 393(6684), 440-442. Chicago

[^4]: Hines, P., Blumsack, S., Sanchez, E. C., & Barrows, C. (2010, January). The topological and electrical structure of power grids. In System Sciences (HICSS), 2010 43rd Hawaii International Conference on (pp. 1-10). IEEE.

[^5]: Cotilla-Sanchez, E., Hines, P. D., Barrows, C., & Blumsack, S. (2012). Comparing the topological and electrical structure of the North American electric power infrastructure. IEEE Systems Journal, 6(4), 616-626.

[^6]: Pagani, G. A., & Aiello, M. (2013). The power grid as a complex network: a survey. Physica A: Statistical Mechanics and its Applications, 392(11), 2688-2700.

[^7]: Carreras, B. A., Lynch, V. E., Dobson, I., & Newman, D. E. (2002). Critical points and transitions in an electric power transmission model for cascading failure blackouts. Chaos: An interdisciplinary journal of nonlinear science, 12(4), 985-994.

[^8]: Parashar, M., Thorp, J. S., & Seyler, C. E. (2004). Continuum modeling of electromechanical dynamics in large-scale power systems. IEEE Transactions on Circuits and Systems I: Regular Papers, 51(9), 1848-1858.

[^9]: Birchfield, A. B., Xu, T., Gegner, K. M., Shetye, K. S., & Overbye, T. J. (2017). Grid structural characteristics as validation criteria for synthetic networks. IEEE Transactions on power systems, 32(4), 3258-3265. Chicago

[^10]: Birchfield, A. B., Gegner, K. M., Xu, T., Shetye, K. S., & Overbye, T. J. (2017). Statistical considerations in the creation of realistic synthetic power grids for geomagnetic disturbance studies. IEEE Transactions on Power Systems, 32(2), 1502-1510. Chicago

[^11]: Gegner, K. M., Birchfield, A. B., Xu, T., Shetye, K. S., & Overbye, T. J. (2016, February). A methodology for the creation of geographically realistic synthetic power flow models. In Power and Energy Conference at Illinois (PECI), 2016 IEEE (pp. 1-6). IEEE.

[^12]: Soltan, Saleh, and Gil Zussman. "Generation of synthetic spatially embedded power grid networks." arXiv:1508.04447 [cs.SY], Aug. 2015.
