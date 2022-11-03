# cufft examples

Examples to reproduce the problem that upsets me when implementing fft in 
paddle with cufft as a backend.

To be concise, I tried to follow the convention of reusing cufft plans via 
wrapping cufftHandles in a RAII-style class. Then use an LRUCache to hold 
instances of the wrapped cufftHandles.

But when using cufft in [cuda 10.1.105](https://developer.nvidia.com/cuda-10.1-download-archive-base) via dynamic loading(`dlopen` the shared library and 
`dlsym` the functions I need) instead of dynamic linking, I almost always get a 
segmentfault when cufftHandles wrer being destroyed.

However, using cufft in [cuda 10.1.105](https://developer.nvidia.com/cuda-10.1-download-archive-base) via dynamic linking is okay. Also, using cufft from 
higher versions of cuda via dynamic loading is okay, too.

I have not found the reasons for this, but this example reproduces what is 
described above.

## to reproduce

1. install several versions of cuda toolkit, cuda 10.1, and higher versions of 
  cuda cuda 10.1 update 1, cuda 10.1 update 2 or cuda 11.4+.
2. git clone this repo, checkout the dynload branch, build it with different 
  versions of cuda and run ./cufft_plans in the build directory.
3. check if it fails with a segmentfault.

## how to build it 

1. a c++ compiler with c++14 support.
2. cuda toolkit. To be able to use a wide range of cuda toolkit, you may need 
  to install cuda driver of high enough version.

```bash
# build it
mkdir build
cd build
cmake ..
make 

# run it 
./cufft_plans
```

NOTE: make sure to clean the build directory after switching to another cuda 
toolkit version.

## Results

`Prime factor <= 127` means none of fft transformation dimension sizes have a 
prime factor larger than 127;
`prime factor > 127` means at least one of fft transformation dimension sizes 
has a prime factor larger than 127;

results dynamic load

|               | Prime factor <= 127 | prime factor > 127 |
| ------------- | ------------------- | ------------------ |
| 10.1 origin   | segmentfault        | segmentfault       |
| 10.1 update 1 | ok                  | segmentfault       |
| 10.1 update2  | ok                  | segmentfault       |
| 10.2          | ok                  | segmentfault       |
| 11.0.3        | segmenfault         | segmentfault       |
| 11.1.1        | segmenfault         | segmenfault        |
| 11.2.2        | ok                  | ok                 |
| 11.3          | ok                  | ok                 |
| 11.4.1        | ok                  | ok                 |


dynamic link

|               | Prime factor <= 127 | prime factor > 127 |
| ------------- | ------------------- | ------------------ |
| 10.1 origin   | ok                  | ok                 |
| 10.1 update 1 | ok                  | ok                 |
| 10.1 update2  | ok                  | ok                 |
| 10.2          | ok                  | ok                 |
| 11.0.3        | ok                  | ok                 |
| 11.1.1        | ok                  | ok                 |
| 11.2          | ok                  | ok                 |
| 11.3.1        | ok                  | ok                 |
| 11.4.1        | ok                  | ok                 |

