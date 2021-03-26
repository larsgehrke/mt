# DISTANA Implementation by Lars Gehrke
This is the repository for Lars Gehrke writing his master's thesis about implementing DISTANA in a parallelized fashion on the GPU to accelerate the model.
The code is based on the former implementation of Matthias Karlbauer (*code_archive/model/distana*).

## code folder
The code for the DISTANA implementation. Every use case (generate data, training, testing, run unit test) has its own executable script at the highest level of this folder structure. These top-level scripts work as controller in a model-view-controller fashion by reasonably combining the scripts of the different subfolders.

+ **config** The whole parameter management and the argument parsing from the command line is done here. For the data generation and the training/testing of DISTANA there are two different scripts which are subclasses of _params.py_. All the parameters are automatically loaded from a binary file and can be edited and saved via command line arguments. By choosing the configuration file name "default", you can overwrite the default settings. These binary configuration files are unversioned (.gitignore: \*.pkl). Thus you can have different parameters for different execution environments (server, local) and easily change them via command line.

+ **diagram** An unversioned folder. By choosing to save the diagrams from test.py, this folder will be created automatically and all diagrams from the test run will be saved here.

+ **diagram_gpu** Not important, can be deleted. This folder was just used to access the diagrams from the server test run. 

+ **model** The main implementation of the different variations of DISTANA are here. 

+ **qa** Quality assurance (qa). Folder for all unit tests. The execution of these unit tests should be selected by calling unit_test.py with the specific command line argument.

+ **tools** To not overstructure the code, all specific code sections that are worth outsourcing are collected here. Here are code sections responsible for persistence, pytorch specific or view specific issues.


## code archive folder
+ **code_archive/model/distana** former implementation by Karlbauer
+ __code_archive/model/distana*__ first steps of code adaption

+ **code_archive/others/extension-cpp** this is the code from the [original PyTorch CUSTOM C++ AND CUDA EXTENSIONS exampel](https://pytorch.org/tutorials/advanced/cpp_extension.html)

  + this is the main resource to implement a custom CUDA kernel for your PyTorch program. Note that there is a plain PyTorch implementation (/python), a C++ implementation (/cpp) and a CUDA implementation (/cuda) of the same custom module! As the tutorial suggests, if you can program your model in plain PyTorch, it should be fine most of the time. PyTorch uses there own very fast CUDA kernels for tensor operations, so there is no need to write your own CUDA kernel most of the time.

+ **code_archive/others/extension-cpp-dev** slightly updated/changed version, to analyse (deprecated) original code and make it work on my server

