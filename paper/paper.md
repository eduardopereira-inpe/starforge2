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

Active Galactic Nuclei are constituted by supermassive black holes (SMBH) surrounded by a accreation disk `[@mortlock2011luminous; @chen2025origins]`. The source of original black holes seeds is a challenger. The remanents of the first metal-free stellar generation, knowing as Population III (Pop III), have been suggested as possible seeds `[@woods2019titans,@chen2025origins]'. Howerver, the discovery of massive quasar at redshift $$z \gtrsim 6$$ indicate a dificult for this scenario `[@woods2019titans]`. This fact motivates models in which SMBHs originate from more massive seed black holes (\(\gtrsim 10^5\,M_{\odot}\)) formed from supermassive stars in atomically cooled halos, as well as alternative seeding channels \cite{woods2019titans}. 

Current studies of SMBH formation and growth predominantly rely on forward- modelling approaches, in wich seed models are evolved under assumed presciption for accretion, merger and cosmology. Theses models are computationally expensive, sensitive to model assumptions and priors as well as poorly suited for inver problems, such as recosntructing the primordial SMBH population from the present data observables.