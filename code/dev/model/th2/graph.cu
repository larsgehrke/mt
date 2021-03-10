#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*#include <config.h>*/

#define BATCH_SIZE 8
#define PK_ROWS 16
#define PK_COLS 16
#define DIMS 3
#define NEIGHBORS 8
#define LAT_SIZE 1
#define DYN_SIZE 1


namespace 
{

    template <typename scalar_t>
    __global__ void graph_forward_kernel(
        const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> dyn_input,
        const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> lat_input,
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

      const int top = pk_thread_id - PK_COLS;
      const int bottom = pk_thread_id + PK_COLS;
      
      if (threadIdx.y > 0)
      {
        /* TOP CENTER */
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][top][DYN_SIZE + lat] = lat_input[batch_block_id][top][lat];
        }

      }

      if (threadIdx.y < PK_ROWS - 1)
      {
        /* BOTTOM CENTER */
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][bottom][DYN_SIZE + LAT_SIZE + lat] = lat_input[batch_block_id][bottom][lat];
        }
      }

      if(threadIdx.x > 0)
      {
        /* LEFT */
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id-1][DYN_SIZE + LAT_SIZE*2 + lat] = lat_input[batch_block_id][pk_thread_id-1][lat];
        }
      }

      if(threadIdx.x < PK_COLS -1)
      {
        /* RIGHT */
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id+1][DYN_SIZE + LAT_SIZE* 3 + lat] = lat_input[batch_block_id][pk_thread_id+1][lat];
        }
      }
      

      /* TOP CENTER */
      /* TOP RIGHT */
      /* LEFT */
      /* RIGHT */
      /* BOTTOM LEFT */
      /* BOTTOM CENTER */
      /* BOTTOM RIGHT */
      
        
      

    }

    template <typename scalar_t>
    __global__ void graph_backward_kernel(
      torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_out,
        torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_dyn_input,
        torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_lat_input) 
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
    torch::Tensor dyn_input,
    torch::Tensor lat_input) {

  auto out = torch::zeros_like(dyn_input, {BATCH_SIZE, PK_ROWS * PK_COLS, 
    DYN_SIZE + NEIGHBORS * LAT_SIZE});


  const dim3 threads(PK_COLS, PK_ROWS);
  const dim3 blocks(BATCH_SIZE);

  AT_DISPATCH_FLOATING_TYPES(out.type(), "graph_forward_kernel", ([&] {
    graph_forward_kernel<scalar_t><<<blocks, threads>>>(
        dyn_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        lat_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        out.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>()
        );
  }));

  return {out};

}

std::vector<torch::Tensor> graph_cuda_backward(
    torch::Tensor d_out) 
{

  auto d_dyn_input = torch::zeros({BATCH_SIZE, PK_ROWS * PK_COLS, DYN_SIZE});
  auto d_lat_input = torch::zeros({BATCH_SIZE, PK_ROWS * PK_COLS, LAT_SIZE});

  const dim3 threads(PK_ROWS, PK_COLS);
  const dim3 blocks(BATCH_SIZE);

  AT_DISPATCH_FLOATING_TYPES(d_dyn_input.type(), "graph_backward_kernel", ([&] {
    graph_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_out.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        d_dyn_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        d_lat_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>());
  }));


  return {d_dyn_input, d_lat_input};
}
