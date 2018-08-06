# Tensorflow Nearest Neighbour ops
Memory-efficient tensorflow implementation of nearest neighbour algorithm. C++/Cuda code is from [Shapenet2017](https://shapenet.cs.stanford.edu/iccv17/) [Chamfer evaluation code](https://shapenet.cs.stanford.edu/iccv17/recon3d/Chamfer.zip). I include the source files for convenience. These files were provided without author/license information, but I am happy to ammend them with such if anyone can provide them.

I had issues with the compile/python script provided so created this repo with the result of my changes. Most changes came from information in the `Compile the op using your system compiler` of the [adding an op](https://www.tensorflow.org/extend/adding_an_op) page.

# Installation
1. Clone this repository
```
cd path/to/parent/dir
git clone https://github.com/jackd/tf_nearest_neighbour.git
```
2. Compile the operation
```
cd src
./compile.sh
cd ..
```
3. Add the parent directory to your python path
```
export PYTHONPATH=path/to/parent/dir:$PYTHONPATH
```
Consider adding this to your `.bashrc`.
4. Test by running the example
```
python example.py
```

# If you installed Tensorflow from source
As the [adding an op](https://www.tensorflow.org/extend/adding_an_op) page states, if you use a `gcc` version `>= 5` you'll likely need to add `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` as bazel command to compile the Python package.

# Alternatives
The nearest neighbour is computed by taking the pair-wise distance between each point in each cloud. While this algorithm is trivial to implement, and this implementation is memory-efficient, it does not scale well to large point clouds. For clouds of size `N` and `M`, it scales like `O(N*M)`.

Different algorithms exist that scale better with large `N` and `M`. For example, to find the one-directional nearest neighbours, KDTrees can be built in `O(M log M)` time and queries in `O(N log M)` time. `benchmark.py` contains a simple wrapping of `scipy`'s `cKDTree` for tensorflow. These are not compiled/run on the GPU, so this leaves plenty of improvement potential. Note the timings do not include the time taken to build the tree, since this can be done on the CPU during preprocessing.

# Requirements
* tensorflow v1.4 or later
* cuda/cudnn
