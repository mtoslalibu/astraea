# astraea
The main repo of Astraea implementation. 

# content
This repo includes /src: 
AstraeaController: Periodically run; fetch traces, extract span units, apply Bayesian methods, and feed sampling decisions to application. 
TraceManager: Read and parse traces. Convert trace to DAG, and extract span units with statistics.
BayesianFramework: Implementation of Astraea Bayesian online learning and mab algorithm. Initialize and update beta distributions, apply MAB algorithm.

## setup
sudo apt install libjpeg-dev zlib1g-dev
pip3 install Pillow


