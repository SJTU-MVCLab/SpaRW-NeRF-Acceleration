#include <iostream>
// #include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t> 
__global__ void compute_xyz_cuda_kernel(
    scalar_t* __restrict__ depth,
    scalar_t* __restrict__ K,
    scalar_t* __restrict__ pcd,
    const int H,
    const int W
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = idx / H;
  const int col = idx % W;
  const int offset = 3*(row + W*col);

  const scalar_t fx = K[0]; // [0][0]
  const scalar_t cx = K[2]; // [0][2]
  const scalar_t fy = K[4]; // [1][1]
  const scalar_t cy = K[5]; // [1][2]

  if(row < H) {
    pcd[offset] = (static_cast<scalar_t>(col) + 0.5 - cx) * depth[offset] / fx; // x
    pcd[offset + 1] = -(static_cast<scalar_t>(row) + 0.5 - cy) * depth[offset] / fy; // y
    pcd[offset + 2] = -depth[offset]; // z
  }
}

std::vector<at::Tensor> compose_pcd(
    at::Tensor img, 
    at::Tensor depth, 
    at::Tensor K, 
    const int H, 
    const int W
){
    // Eigen::Tensor<float, 3> pcd(H, W, 3);
    
    // std::vector<std::vector<std::vector<int>>> pcd(H, std::vector<std::vector<float>>(W,std::vector<int>(3,0)));
    
    // double *pcd;
	  // cudaMalloc((void**)(&pcd), H * W * 3 * sizeof(double));
    auto pcd = at::zeros({H, W, 3}, depth.options());

    const int threads = 512;
    const int blocks = (H * W + threads - 1) / threads;

    at::AT_DISPATCH_FLOATING_TYPES(depth.type(), "compose_pcd", ([&] {
    compute_xyz_cuda_kernel<scalar_t><<<blocks, threads>>>(
      depth.data<scalar_t>(),
      K.data<scalar_t>(),
      pcd.data<scalar_t>(),
      H, W);
    }));
    
    return {pcd};
}

int main() {
    at::Device device(at::kCUDA);
    auto img = at::rand({800, 800, 3}, device);
    auto depth = at::rand({800, 800, 3}, device);
    auto K = at::rand({3, 3}, device);
    
    auto res = compose_pcd(img, depth, K, 800, 800);
    std::cout<<res[0]<<std::endl;
    return 0;
}