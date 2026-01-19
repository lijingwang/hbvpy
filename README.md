# hbvpy

A Python implementation of the HBV conceptual lumped hydrologic model for educational and research use. There is no routing implemented yet. 

## overview

This repository provides a simple and readable Python implementation of the HBV rainfall–runoff model. The code is designed primarily for teaching hydrologic processes, demonstrating conceptual model structure, and supporting small-scale experiments and reproducible examples. It is not intended to be a fully featured operational hydrologic modeling system. 

## repository contents

- hbv.py  
  Core HBV conceptual hydrologic model implementation.

- metrics.py  
  Helper functions for commonly used model performance metrics.

- HBV_demo.ipynb  
  Demonstration notebook illustrating model setup, execution, and evaluation.


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

Users are encouraged to modify parameters and forcings to explore hydrologic process sensitivity, including snow accumulation and melt, soil moisture accounting, and runoff generation.

## citation

If you use this repository or derived materials for teaching, presentations, or publications, please cite:

AghaKouchak, A., and Habib, E., 2010. Application of a conceptual hydrologic model in teaching hydrologic processes. International Journal of Engineering Education, 26(4), 963–973.

## acknowledgements

This implementation was modified from the following sources:

- HRL (2026). HBV-EDU Hydrologic Model, MATLAB Central File Exchange  
  https://www.mathworks.com/matlabcentral/fileexchange/41395-hbv-edu-hydrologic-model

- https://github.com/johnrobertcraven/hbv_hydromodel.git

Portions of the repository documentation were drafted with assistance from OpenAI ChatGPT and subsequently reviewed and edited by the author.

## license

This repository is released under the MIT License. See the LICENSE file for details.

## contact

Author: Lijing Wang  
Email: lijing.wang@uconn.edu

