#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*#include <config.h>*/

#define BATCH_SIZE 10
#define PK_ROWS 16
#define PK_COLS 16
#define DIMS 3

namespace 
{

    template <typename scalar_t>
    __global__ void graph_forward_kernel(
        const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> in,
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
      ThreadIdx.x — thread’s index in x dimension
      ThreadIdx.y — thread’s index in y dimension

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
        
        /* TODO: Forward Pass
        out[batch_block_id][pk_thread_id][0] = in;*/


    }

    template <typename scalar_t>
    __global__ void graph_backward_kernel(
      torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> out,
        torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> in) 
    {

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
        
        /* TODO: Backward Pass
        out[batch_block_id][pk_thread_id][0] = in;*/

    }

} // namespace

std::vector<torch::Tensor> graph_cuda_forward(
    torch::Tensor input) {

  auto out = torch::zeros_like(input);

  const dim3 threads(PK_ROWS, PK_COLS);
  const dim3 blocks(BATCH_SIZE);

  AT_DISPATCH_FLOATING_TYPES(out.type(), "graph_forward_kernel", ([&] {
    graph_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        out.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>()
        );
  }));

  return {out};

}

std::vector<torch::Tensor> graph_cuda_backward(
    torch::Tensor d_out) 
{

  auto d_in = torch::zeros_like(d_out);

  const dim3 threads(PK_ROWS, PK_COLS);
  const dim3 blocks(BATCH_SIZE);

  AT_DISPATCH_FLOATING_TYPES(d_in.type(), "graph_backward_kernel", ([&] {
    graph_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_out.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        d_in.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>());
  }));


  return {d_in};
}
