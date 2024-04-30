# slang-3DGS
3D Gaussian Splatting Implemented in Slang
Zhouhua Xie - xzh015

## Project Goal
Implement a basic version of 3DGS in Slang
For compatibility and easiness of debugging, start with slang-py
The focus is on implementing the Gaussian kernel optimization part, not on the splitting and merging logic of initializing the Gaussian kernel.

Input: Unoptimized Gaussian kernel, reference images, and camera parameters.
Output: Optimized Gaussian kernel, novel view generation.

- Mile Stones
  - Single Gaussian Optimization with slang-py in a synthesized scene
  - Multi-Gaussian Optimization with slang-py in a synthesized scene
  - Re-optimization of a real scene from 3DGS paper with slang-py
  - Export and test the slang code compiled in a rendering language (HLSL)
  - 3D scene reconstruction using slang-py code as a plug-in for the original 3DGS code

Using the proposed matrix in the 3DGS paper, with additional comparison on performance.
