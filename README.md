# STARFORGE  
## STellar And black hole Rate FORmation & Gravitational Emission

**STARFORGE** is a scientific software designed to study the cosmic star formation history and the formation and evolution of supermassive black holes (SMBHs) within different cosmological models, including their associated gravitational emission.

---

## Scientific Scope

The original model underlying STARFORGE was first presented in:

**Pereira, E. S., & Miranda, O. D. (2010)**  
*MNRAS, 401, 1924*

The framework is based on three fundamental layers:

1. Cosmological background  
2. Hierarchical structure formation based on Press–Schechter-like formalisms  
3. Cosmic star formation rate (CSFR) evolution  

---

## Extended Framework

The code has been updated to include an inverse-problem-based framework aimed at reconstructing the primordial SMBH mass function under cosmological evolution.

This framework focuses on methodological validation rather than on deriving definitive astrophysical constraints. Using synthetic present-day data with controlled noise levels, the robustness and stability of the reconstruction are assessed.

Key results include:

- Reliable recovery of the initial mass function and physical parameters for noise levels up to ~5%  
- Reconstruction errors at the few-percent level for the present-day state  
- Significant degradation at higher noise levels due to the ill-posed nature of the inverse problem and parameter degeneracies  

The framework emphasizes the importance of regularization and uncertainty quantification when inferring cosmological initial conditions from final-state observables.

---

## Main Features

- Cosmic star formation history modeling for different cosmologies  
- Hierarchical structure formation  
- Cosmological evolution of SMBH populations  
- Inverse reconstruction of primordial SMBH mass functions  
- Controlled treatment of observational noise  

---

## Author

**Eduardo S. Pereira**  
Email: pereira.somoza@gmail.com

---

## License
Copyright (c) 2026, Eduardo S. Pereira
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



