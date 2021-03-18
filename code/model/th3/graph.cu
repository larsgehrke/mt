#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/* Custom config file written by Python code*/
#include <config.h>

#define DIMS 3
#define NEIGHBORS 8

namespace 
{

    template <typename scalar_t>
    __global__ void graph_forward_kernel(
        const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> dyn_input,
        const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> lat_input,
        const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> connections,
        torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> out) {

      /*

      Number of blocks in the grid:
      gridDim.x — number of blocks in the x dimension of the grid 
      gridDim.y — number of blocks in the y dimension of the grid

      Number of threads in a block:
      blockDim.x — number of threads in the x dimension if the grid 
      blockDim.y — number of threads in the y dimension if the grid

      Block Index:
      blockIdx.x — block’s index in x dimension
      blockIdx.y — block’s index in y dimension

      Thread Index:
      threadIdx.x — thread’s index in x dimension
      threadIdx.y — thread’s index in y dimension

      */

      /*
        Calculating block index: 
        row no (blockIdx.y) * length of row (gridDim.x) + row position (blockIdx.x)
      */
      const int batch_block_id = blockIdx.y * gridDim.x + blockIdx.x;

      /*
        Calculating thread index:
        like block_id, see above
      */
      const int pk_thread_id = threadIdx.y * blockDim.x + threadIdx.x;


      for (int dyn = 0; dyn < DYN_SIZE; dyn++)
      {
        out[batch_block_id][pk_thread_id][dyn] = dyn_input[batch_block_id][pk_thread_id][dyn];
      } 



      for (int lat = 0; lat < LAT_SIZE; lat++)
      {
        
        
        /* end of for loop for lateral connections*/
      }
      /* end of forward pass*/
    }

    template <typename scalar_t>
    __global__ void graph_backward_kernel(
      const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_out,
      const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> connections,
        torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_dyn_input,
        torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_lat_input) 
    {
      
      /* end of backward pass */
    }

/* end of namespace */
} 

std::vector<torch::Tensor> graph_cuda_forward(
    torch::Tensor dyn_input,
    torch::Tensor lat_input,
    torch::Tensor connections) {

  auto options = torch::TensorOptions().device(torch::kCUDA).requires_grad(true);

  const auto batch_size = dyn_input.size(0);

  auto out = torch::zeros({batch_size, PK_ROWS * PK_COLS, 
    DYN_SIZE + NEIGHBORS * LAT_SIZE}, options);


  const dim3 threads(PK_COLS, PK_ROWS);
  const dim3 blocks(batch_size);

  AT_DISPATCH_FLOATING_TYPES(out.type(), "graph_forward_kernel", ([&] {
    graph_forward_kernel<scalar_t><<<blocks, threads>>>(
        dyn_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        lat_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        connections.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        out.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>()        
        );
  }));

  return {out};

}

std::vector<torch::Tensor> graph_cuda_backward(
    torch::Tensor d_out,
    torch::Tensor connections) 
{
  const auto batch_size = d_out.size(0);
  auto options = torch::TensorOptions().device(torch::kCUDA).requires_grad(true);

  auto d_dyn_input = torch::zeros({batch_size, PK_ROWS * PK_COLS, DYN_SIZE}, options);
  auto d_lat_input = torch::zeros({batch_size, PK_ROWS * PK_COLS, LAT_SIZE}, options);

  const dim3 threads(PK_ROWS, PK_COLS);
  const dim3 blocks(batch_size);

  AT_DISPATCH_FLOATING_TYPES(d_dyn_input.type(), "graph_backward_kernel", ([&] {
    graph_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_out.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        connections.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        d_dyn_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        d_lat_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        );
  }));


  return {d_dyn_input, d_lat_input};
}
