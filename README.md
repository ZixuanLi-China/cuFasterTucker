# cuFasterTucker


sh run


reference


@article{10.1145/3648094,
author = {Li, Zixuan and Qin, Yunchuan and Xiao, Qi and Yang, Wangdong and Li, Kenli},
title = {cuFasterTucker: A Stochastic Optimization Strategy for Parallel Sparse FastTucker Decomposition on GPU Platform},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {2329-4949},
url = {https://doi.org/10.1145/3648094},
doi = {10.1145/3648094},
abstract = {The amount of scientific data is currently growing at an unprecedented pace, with tensors being a common form of data that display high-order, high-dimensional, and sparse features. While tensor-based analysis methods are effective, the vast increase in data size has made processing the original tensor infeasible. Tensor decomposition offers a solution by decomposing the tensor into multiple low-rank matrices or tensors that can be efficiently utilized by tensor-based analysis methods. One such algorithm is the Tucker decomposition, which decomposes an N-order tensor into N low-rank factor matrices and a low-rank core tensor. However, many Tucker decomposition techniques generate large intermediate variables and require significant computational resources, rendering them inadequate for processing high-order and high-dimensional tensors. This paper introduces FasterTucker decomposition, a novel approach to tensor decomposition that builds on the FastTucker decomposition, a variant of the Tucker decomposition. We propose an efficient parallel FasterTucker decomposition algorithm, called cuFasterTucker, designed to run on a GPU platform. Our algorithm has low storage and computational requirements and provides an effective solution for high-order and high-dimensional sparse tensor decomposition. Compared to state-of-the-art algorithms, our approach achieves a speedup of approximately 7 to 23 times.},
note = {Just Accepted},
journal = {ACM Trans. Parallel Comput.},
month = {feb},
keywords = {GPU CUDA Parallelization, Kruskal Approximation, Sparse Tensor Decomposition, Stochastic Strategy, Tensor Computation}
}
