# DISTANA Implementation by Lars Gehrke
This is the repository for Lars Gehrke writing his master's thesis about implementing DISTANA in a parallelized fashion on the GPU to accelerate the model.
The code is based on the former implementation of Matthias Karlbauer (*code_archive/model/distana*).

## code folder



## code archive folder
+ **code_archive/model/distana:** former implementation by Karlbauer
+ __code_archive/model/distana*:__ first steps of code adaption

+ **code_archive/others/extension-cpp:** this is the code from the [original PyTorch CUSTOM C++ AND CUDA EXTENSIONS exampel](https://pytorch.org/tutorials/advanced/cpp_extension.html)

... this is the main resource to implement a custom CUDA kernel for your PyTorch program. Note that there is a plain PyTorch implementation (/python), a C++ implementation (/cpp) and a CUDA implementation (/cuda) of the same custom module! As the tutorial suggests, if you can program your model in plain PyTorch, it should be fine most of the time. PyTorch uses there own very fast CUDA kernels for tensor operations, so there is no need to write your own CUDA kernel most of the time.

+ **code_archive/others/extension-cpp-dev:** slightly updated/changed version, to analyse (deprecated) original code and make it work on my server

