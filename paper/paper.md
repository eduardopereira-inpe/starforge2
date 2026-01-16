---
title: 'STARFORGE - STellar And black hole Rate FORmation & Gravitational Emission'
tags:
  - Python
  - astronomy
  - black holes
  - inverse problem
  - Active Galact Nuclei

authors:
  - name: Eduardo S. Pereira
    orcid: 0000-0002-1564-2933
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: COPDT, INPE -Instituto Nacional de Pesquisas Espaciais, Av. dos Astronautas, 1758 {S\~ao Jos\'e dos Campos}, SP, Brazil
   index: 1

date: 16 Janurary 2026
bibliography: paper.bib

# Summary 

Supermassive black holes (SMBHs) are ubiquitous in the nuclei of massive galaxies, yet their formation pathways and early growth mechanisms remain uncertain. Proposed seeding scenarios range from the remnants of massive Population III stars to the direct collapse of supermassive stars in atomically cooled dark-matter halos. Inferring the properties of these primordial seeds from present-day observations is intrinsically difficult, as the observable SMBH population has undergone complex, nonlinear evolution driven by accretion, mergers, and cosmological structure formation.

Conventional studies typically adopt a forward-modelling strategy, in which parametrized dynamical or semi-analytical models are evolved numerically and tuned to reproduce observational data. While physically motivated, this approach is computationally expensive, sensitive to modeling assumptions, and often suffers from parameter degeneracies that limit its ability to uniquely constrain the initial SMBH mass function.

In this work, we introduce an inverse-problem-based framework aimed at reconstructing the primordial SMBH mass function while consistently accounting for cosmological evolution. Rather than focusing on precise astrophysical constraints, the primary objective is to assess the stability, robustness, and reliability of the inversion methodology itself.

To support this analysis, we employ STARFORGE, a scientific software platform developed to investigate cosmic star formation and the formation and evolution of SMBHs across different cosmological scenarios, including their associated gravitational-wave emission.

# Statement of need

Active Galactic Nuclei (AGN) are powered by supermassive black holes (SMBHs) surrounded by accretion disks `[@mortlock2011luminous,@chen2025origins]`. Despite their ubiquity, the origin of the initial black hole seeds remains an open problem. Remnants of the first generation of metal-free stars, known as Population III (Pop III) stars, have long been proposed as viable progenitors of these seeds `[@woods2019titans; @chen2025origins]`. However, the discovery of luminous quasars at redshifts $$z \gtrsim 6$$ poses a significant challenge to this scenario, as the limited time available for growth makes it difficult for light Pop III remnants to reach the observed masses `[@woods2019titans]`. This tension has motivated alternative models in which SMBHs originate from more massive seeds $$\gtrsim 10^{5}\,M_{\odot}$$, formed through the collapse of supermassive stars in atomically cooled halos, as well as other proposed seeding channels `[@woods2019titans]`.

Most current studies of SMBH formation and growth adopt forward-modelling approaches, in which seed populations are evolved under prescribed models for accretion, mergers, and cosmological expansion `[@yu2002observational,@shankar2013accretion; @tucci2017constraining,@lai2024supermassive]`. While physically well motivated, these methods are computationally demanding, highly sensitive to modelling assumptions and priors, and poorly suited to inverse problems, such as reconstructing the primordial SMBH population from present-day observables. Moreover, large-scale cosmological simulations are not designed for rapid experimentation with alternative evolutionary operators or inversion techniques.

STARFORGE addresses this gap by providing a modular and transparent software framework for modeling the coupled evolution of cosmic star formation and SMBH seed formation and growth under different cosmological assumptions. Its architecture is specifically designed to support inverse-problem methodologies, enabling systematic testing of the stability and robustness of reconstruction techniques while maintaining computational efficiency and reproducibility.
