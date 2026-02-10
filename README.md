# hbvpy

A Python implementation of the HBV conceptual lumped hydrologic model for educational and research use. There is no routing implemented yet. 

## overview

This repository provides a simple and readable Python implementation of the HBV rainfall–runoff model. The code is designed primarily for teaching hydrologic processes, demonstrating conceptual model structure, and supporting small-scale experiments and reproducible examples. It is not intended to be a fully featured operational hydrologic modeling system. 

It also includes uncertainty analysis and sensitivity analysis for HBV. 

## repository contents

- hbv.py  
  Core HBV conceptual hydrologic model implementation.

- metrics.py  
  Helper functions for commonly used model performance metrics.

- HBV_demo.ipynb
  Demonstration notebook illustrating model setup, execution, and evaluation.

- HBV_uncertainty_propagation.ipynb
  Uncertainty propagation using Latin Hypercube Sampling (LHS) to explore parameter uncertainty effects on simulated discharge.

- HBV_sensitivity_analysis.ipynb
  Sensitivity analysis comparing four methods: One-at-a-Time (OAT), Morris, Sobol (first/second/total order), and Distance-based GSA (DGSA).

## installation

This repository is not distributed as a Python package. To use it, clone the repository:

git clone https://github.com/lijingwang/hbvpy.git

cd hbvpy

You may then run the demonstration notebook or import the model directly in Python.

## usage

The recommended starting point is the `HBV_demo.ipynb` notebook. It demonstrates:

- Model structure and parameterization
- Forcing data input
- Simulation of streamflow
- Evaluation using standard performance metrics

Additional notebooks:

- `HBV_uncertainty_propagation.ipynb`: Propagates parameter uncertainty through the model using LHS ensembles and visualizes the spread in simulated discharge.
- `HBV_sensitivity_analysis.ipynb`: Compares OAT, Morris, Sobol, and DGSA sensitivity methods across multiple flow metrics (Mean Q, Q10, Q90) and full time series response.

All notebooks include a Colab badge for one-click execution in Google Colab.

Users are encouraged to modify parameters and forcings to explore hydrologic process sensitivity, including snow accumulation and melt, soil moisture accounting, and runoff generation.

## citation

If you use this repository or derived materials for teaching, presentations, or publications, please cite:

AghaKouchak, A., and Habib, E., 2010. Application of a conceptual hydrologic model in teaching hydrologic processes. International Journal of Engineering Education, 26(4), 963–973.

## acknowledgements

This implementation was modified from the following sources:

- HRL (2026). HBV-EDU Hydrologic Model, MATLAB Central File Exchange  
  https://www.mathworks.com/matlabcentral/fileexchange/41395-hbv-edu-hydrologic-model

- https://github.com/johnrobertcraven/hbv_hydromodel.git

- Perzan, Z. pyDGSA: Distance-based Generalized Sensitivity Analysis in Python
  https://github.com/zperzan/pyDGSA

Portions of the repository documentation were drafted with assistance from Claude AI, ChatGPT and subsequently reviewed and edited by the author.

## license

This repository is released under the MIT License. See the LICENSE file for details.

## contact

Author: Lijing Wang  
Email: lijing.wang@uconn.edu

