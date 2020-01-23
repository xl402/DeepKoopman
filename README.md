# DeepKoopman
*CUED IIB Master's Thesis Project*

## Introduction: 
The Koopman operator framework is becoming increasingly popular for obtaining linear representations of nonlinear systems from data. This project aims to optimally input non-affine nonlinear systems, utilizing Deep Learning (DL) to discover the Koopman invariant subspace, bridging the gap between DL based Koopman eigenfunction discovery and optimal predictive control.

## Networks Overview:
Script `networks.py` contains all networks discussed in the thesis, including:
- **LREN**: Linearly Recurrent Encoder Network
- **DENIS**: Deep Encoder with Initial State parameterisation
- **DEINA**: Deep Encoder for Input Non-Affine systems

## Pendulum Example
Left: Predicted trajectories overlaying ground truth. Right: Top two Koopman eigenfunctions (which together, convey the Hamiltonian energy of the system)

![Pendulum](https://media.giphy.com/media/UTv9kmS9nfk0PSI05h/giphy.gif)
![Imgur](https://imgur.com/xYJKNjh)

