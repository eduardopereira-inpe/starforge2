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

## Software Design

STARFORGE was designed as a research-oriented framework to study the coupled evolution of cosmic star formation and supermassive black hole (SMBH) populations, with particular emphasis on methodological transparency and inverse-problem applications. The software architecture reflects a deliberate decomposition of the underlying physical processes, rather than a purely technical organization. Core functionality is structured hierarchically into packages corresponding to cosmological background evolution, structure formation (e.g., dark-matter halo collapse), star formation rate modeling, SMBH seed formation and growth, and observational data handling. This separation mirrors the causal structure of the astrophysical system and enables independent development, validation, and substitution of physical prescriptions.

A key design trade-off in STARFORGE is the prioritization of modularity and interpretability over maximal computational optimization. Evolutionary processes such as accretion, mergers, and star formation laws are implemented as interchangeable components, allowing users to systematically explore alternative models and assess their impact on inferred SMBH populations. While this modular design introduces some computational overhead relative to monolithic implementations, it significantly enhances flexibility and is well suited to studies focused on uncertainty quantification, sensitivity analysis, and inversion stability rather than large-scale forward simulations.

An object-oriented programming (OOP) approach, guided by SOLID design principles, underpins the software architecture. Core physical entities—such as black holes, halos, and cosmological models—are represented as objects with clearly defined responsibilities and interfaces. This design minimizes coupling between high-level scientific workflows and low-level numerical implementations, facilitating extensibility and reducing the risk that changes in one component propagate unintentionally through the codebase. In particular, this separation is critical for inverse-problem formulations, where evolution operators, priors, and observational mappings must be modified or replaced independently.

Inverse modeling considerations constitute a central design constraint of STARFORGE. Unlike most existing astrophysical frameworks, which are optimized for forward evolution, STARFORGE explicitly separates evolutionary operators from observable-space mappings. This architectural choice enables the reconstruction of primordial SMBH populations from present-day observables and supports systematic testing of inversion robustness under different regularization schemes and cosmological assumptions. The resulting design requires more explicit bookkeeping of intermediate states but provides a level of methodological control that is difficult to achieve in forward-only simulation codes.

To promote reproducibility and transparency, observational datasets are integrated into dedicated data modules rather than treated as external, ad hoc inputs. This approach ensures consistent preprocessing, unit handling, and versioning of data used in numerical experiments, while facilitating direct comparison between model predictions and observations. Although bundling data increases the complexity of the codebase, it substantially lowers barriers to reproducible analysis and independent validation.

Overall, STARFORGE is intentionally designed as a flexible research framework rather than a fixed end-to-end pipeline. Users are encouraged to assemble and inspect workflows programmatically, modify components at runtime, and analyze intermediate results. This design philosophy aligns with the needs of exploratory theoretical research and enables controlled comparisons of SMBH seeding scenarios, cosmological models, and inversion techniques within a unified and reproducible software environment.


# Research Impact Statement

This software consolidates and operationalizes more than a decade of research in extragalactic astrophysics. The underlying  mathematical framework for modeling the cosmic star formation rate was first introduced in `@pereira2010stochastic`. Subsequent studies extended this approach to the reconstruction of the formation and growth history of supermassive black holes (SMBHs) under cosmological evolution, with key results presented in `@pereira2010massive`, `@pereira2011supermassive`, `@pereira2014accretion`, and `@pereira2019towards`.

STARFORGE provides a unified and reproducible software implementation of these previously published methodologies. By translating established theoretical and numerical models into a modular and extensible codebase, the software enables independent verification of earlier results and facilitates their application to new datasets and inverse-problem formulations. The inclusion of documented workflows and reproducible numerical experiments constitutes concrete evidence of research impact by lowering barriers to reuse and supporting methodological transparency.


# AI usage disclosure

Generative AI tools were used in a limited and assistive capacity during the development of this work. GitHub Copilot was employed for code completion and the generation of initial docstring templates. All code produced with AI assistance was reviewed, modified where necessary, and validated by the authors through testing and comparison with expected physical and numerical behavior.

In addition, a large language model (ChatGPT) was used to assist with language editing and clarity improvements in the manuscript text. The scientific content, structure, and conclusions were defined by the authors, and all AI-suggested revisions were critically reviewed to ensure accuracy, consistency, and compliance with scholarly standards.


# Acknowledgements

The authors acknowledge the National Institute for Space Research.



# References