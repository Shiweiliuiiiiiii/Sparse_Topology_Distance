# Sparse_Topology_Distance
This repo contains code accompaning the paper, [Topological Insights into Sparse Neural Networks (Liu et al., ECMLPKDD 2020)](https://arxiv.org/abs/2006.14085) . It includes code and topology example for measuring the sparse topology distance between two sparse neural networks. The sparse topology used in this paper refers to the topology of a sub-network (a fraction of parameters) of a dense network. And the sparse topology distance measured in this paper refers to the distance between two sub-networks. 

## Requirements

This code requires the following:

Linux

Python 3.*

Sscipy 1.4.1

matplotlib 3.1.3

## Code Structure and Usage

The repo provide an example implementation of NNSTD, a method to measure the distance between the graphs spanned over the network weights.

The main script is sparse_topology_distance.py which contains the code to compare different sparse topologies. 

The sparse networks are obtained by [Sparse Evolutionary Training (SET)](https://www.nature.com/articles/s41467-018-04316-3.pdf) with their efficient [code](https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks).

We upload the topologies used to reproduce the results of Figure 4 in the original paper in the folder topo/. 

To reproduce the results of Figure 4 from the paper, run 
```
python sparse_topology_distance.py --epoch 0 --layer 0
```
## Results from the Paper

We initialize  one sparse network with a density level of 0.6%. Then, we generate 9 networks by iteratively changing 1% of the connections from the previous generation step. By doing this, the density of these networks is the same, whereas thetopologies vary a bit. Measured by our method, we find that similar initial topologies gradually evolve to very different topologies while training with adaptive sparse connectivity shown as following:
![](Figure_4.png)

## References

If you use this code, please consider citing the following paper:

@inproceedings{liu2020topological,  
title={Topological Insights into Sparse Neural Networks},  
author={Liu, Shiwei and Van der Lee, Tim and Yaman, Anil and Atashgahi, Zahra and Ferraro, Davide and Sokar, Ghada and Pechenizkiy, Mykola and Mocanu, Decebal Constantin},  
booktitle={ECMLPKDD},  
year={2020}  
}  






