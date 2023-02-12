# GPU-image-processing-with-OpenCL
GPU kernel is created for bluring an imaging using OpenCL

In this repository, the blur function written for CPU, MPI, and CUDA, is implemented for OpenCL. For running the main.cpp, the user must pass the path of an image and blur level as the first and second command line argument, respectively. The application will return the original and blurred image using both CPU and GPU as well as the timing of CPU and GPU processing and speedup. A detailed explanation of the OpenCL kernel and the a2.cpp file and performance comparison with CUDA GPU processing has been provided in the following sections.

## OpenCL vs CUDA
The most obvious difference between OpenCL and CUDA is the simplicity of CUDA programming. In the main function of the CUDA application, there is no need to define platform, device, context, command queue, or program. In CUDA, Only data needs to be transferred to the device (GPU) from the host (CPU) and calling the kernel with desired block and thread configuration, and finally copying the answer from the device to the host. However, since OpenCL is designed to work with all kinds of GPUs (NVIDIA, AMD, Intel etc.) there is a need to clearly configure the application by defining platform, device, context, command queue, and program, as can be seen in main.cpp which is attached to this report.

In contrast to the main function of CUDA and OpenCL, there is a minor difference between CUDA and OpenCL kernels. The figures below show the OpenCL kernel for blurring an Image. Similar to the CUDA application, a 2-D thread configuration is implemented with OpenCL in this repo. The main difference here is how to get the thread indexes in x and y dimensions, as can be seen in lines 5 and 6 of the kernel.

## Comparison

The table below shows the timing of the CPU, OpenCL, and CUDA on the same image but with different blur levels. It should be noted that the CUDA was run with the optimum threads configurations that were found in the repo related to CUDA image processing.

|     Blur   level    |     CPU   time (s)    |     OpenCL   time (s)    |     CUDA   time    |     OpenCL   Speedup    |     CUDA   speedup    |
|---------------------|-----------------------|--------------------------|--------------------|-------------------------|-----------------------|
|     2               |     0.403             |     0.016                |     0.033          |     25.188              |     12.212            |
|     3               |     0.785             |     0.032                |     0.063          |     24.531              |     12.460            |
|     5               |     2.204             |     0.078                |     0.159          |     28.256              |     13.862            |
|     7               |     4.139             |     0.125                |     0.297          |     33.112              |     13.936            |
|     10              |     8.057             |     0.250                |     0.576          |     32.228              |     13.988            |
