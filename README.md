# DISTANA Implementation by Lars Gehrke
This is the repository for Lars Gehrke writing his master's thesis about implementing DISTANA in a parallelized fashion on the GPU to accelerate the training and the testing.
The code is based on the former implementation of Matthias Karlbauer (*code_archive/model/distana*).

## code folder
The code for the DISTANA implementation. Every use case (generate data, training, testing, run unit test) has its own executable script at the highest level of this folder structure. These top-level scripts work as controller in a model-view-controller fashion by reasonably combining the scripts of the different subfolders.

+ **config** The whole parameter management and the argument parsing from the command line is done here. For the data generation and the training/testing of DISTANA there are two different scripts which are subclasses of _params.py_. All the parameters are automatically loaded from a binary file and can be edited and saved via command line arguments. By choosing the configuration file name "default", you can overwrite the default settings. These binary configuration files are unversioned (.gitignore: \*.pkl). Thus you can have different parameters for different execution environments (server, local) and easily change them via command line.

+ **diagram** An unversioned folder. By choosing to save the diagrams from test.py, this folder will be created automatically and all diagrams from the test run will be saved here.

+ **diagram_gpu** Not important, can be deleted. This folder was just used to access the diagrams from the server. 

+ **model** The different implementations of DISTANA are collected here. Each variation has its own subfolder. The facade defines the function signatures, every variation must implement and it selects the variation specified by the parameters at runtime. 
*abstract_evaluator* is a super class for different variations. 
[Object-oriented programming (OOP) can be a curse or a blessing: OOP has of course many benefits, e.g. better readability, better maintainability, better expandability etc. but Python is a slow programming language and has no strong support of oop approaches. The outsourcing of code segments that are called per batch/sample or even per time step should be done as little as possible. For the management and analysis of different implementation variation, a super class can be a good choice. But if you want to optimize the speed of code execution, the superclass *abstract_evaluator* costs unnecessary Python overhead.]
  + **old** [No batch; stacked lateral output] 
  Basically Karlbauer's implementation embedded in this code framework. This implementation can only process the data files per sample  (batch size = 1) and each PK produce an individual lateral output for every outgoing connection. 
  + **old2** [No batch; single lateral output] 
  As *old*, but each PK produce only one lateral output for all outgoing connections.
  + **v1** [batch processing; stacked lateral output] 
  Based on *old*, but this implementation can process the samples of one batch (with arbitrary batch size) in parallel. The Prediction Kernel is implemented as a custom PyTorch class in Python with the usage of tensor operations realising the nn layers (fc, lstm, fc). In the last iteration per epoch the batch size will probably not fit the rest of the data samples. In this case the weight tensors are automatically adapted for this last iteration with a special batch size (amount of remaining samples).
  + **v2** [batch processing; single lateral output; CUDA hard-coded lateral connections] 
  As *v1*, but with a custom CUDA kernel that is implementing the lateral flow between the PKs at the beginning of each time step. However, in this CUDA kernel the grid structure where each PK is laterally connected with up to 8 surrounding neighbors (cf. https://arxiv.org/pdf/1912.11141.pdf) is hard coded. 
  + **v3** [batch processing; single lateral output; flexible lateral connections in CUDA] 

+ **qa** Quality assurance (qa). Folder for all unit tests. The execution of these unit tests should be selected by calling unit_test.py with the specific command line argument.

+ **tools** To not overstructure the code, all specific code sections that are worth outsourcing are collected here, e.g. code sections responsible for persistence, pytorch specific or view specific issues.


## code archive folder
+ **code_archive/model/distana** former implementation by Karlbauer
+ __code_archive/model/distana*__ first steps of code adaption

+ **code_archive/others/extension-cpp** this is the code from the [original PyTorch CUSTOM C++ AND CUDA EXTENSIONS exampel](https://pytorch.org/tutorials/advanced/cpp_extension.html)

  + this is the main resource to implement a custom CUDA kernel for your PyTorch program. Note that there is a plain PyTorch implementation (/python), a C++ implementation (/cpp) and a CUDA implementation (/cuda) of the same custom module! As the tutorial suggests, if you can program your model in plain PyTorch, it should be fine most of the time. PyTorch uses there own very fast CUDA kernels for tensor operations, so there is no need to write your own CUDA kernel most of the time.

+ **code_archive/others/extension-cpp-dev** slightly updated/changed version, to analyse (deprecated) original code and make it work on the server

