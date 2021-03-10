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
      

      /* TOP LEFT */
      if (threadIdx.y > 0 && threadIdx.x > 0)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + lat] = lat_input[batch_block_id][top-1][lat];
        }

      }
      
      /* TOP CENTER */
      if (threadIdx.y > 0)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE + lat] = lat_input[batch_block_id][top][lat];
        }

      }

      /* TOP RIGHT */
      if (threadIdx.y > 0 && threadIdx.x < PK_COLS - 1)
      {        
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 2 + lat] = lat_input[batch_block_id][top+1][lat];
        }

      }

      /* LEFT */
      if(threadIdx.x > 0)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 3 + lat] = lat_input[batch_block_id][pk_thread_id-1][lat];
        }
      }

      /* RIGHT */
      if(threadIdx.x < PK_COLS -1)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 4 + lat] = lat_input[batch_block_id][pk_thread_id+1][lat];
        }
      }

      /* BOTTOM LEFT */
      if (threadIdx.y < PK_ROWS - 1 && threadIdx.x > 0)
      {        
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 5 + lat] = lat_input[batch_block_id][bottom-1][lat];
        }
      }

      /* BOTTOM CENTER */
      if (threadIdx.y < PK_ROWS - 1)
      {        
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 6 + lat] = lat_input[batch_block_id][bottom][lat];
        }
      }

      /* BOTTOM RIGHT */
      if (threadIdx.y < PK_ROWS - 1 && threadIdx.x < PK_COLS - 1)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 7 + lat] = lat_input[batch_block_id][bottom+1][lat];
        }
      }


    }

    template <typename scalar_t>
    __global__ void graph_backward_kernel(
      const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_out,
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


      for (int dyn = 0; dyn < DYN_SIZE; dyn++)
      {
        d_dyn_input[batch_block_id][pk_thread_id][dyn] = d_out[batch_block_id][pk_thread_id][dyn];
      } 

      const int top = pk_thread_id - PK_COLS;
      const int bottom = pk_thread_id + PK_COLS;

      /* TOP LEFT */
      if (threadIdx.y > 0 && threadIdx.x > 0)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          lat_input[batch_block_id][top-1][lat] = out[batch_block_id][pk_thread_id][DYN_SIZE + lat];
        }

      }
      
      /* TOP CENTER */
      if (threadIdx.y > 0)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          lat_input[batch_block_id][top][lat] = out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE + lat];
        }

      }

      /* TOP RIGHT */
      if (threadIdx.y > 0 && threadIdx.x < PK_COLS - 1)
      {        
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          lat_input[batch_block_id][top+1][lat] = out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 2 + lat];
        }

      }

      /* LEFT */
      if(threadIdx.x > 0)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          lat_input[batch_block_id][pk_thread_id-1][lat] = out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 3 + lat];
        }
      }

      /* RIGHT */
      if(threadIdx.x < PK_COLS -1)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          lat_input[batch_block_id][pk_thread_id+1][lat] = out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 4 + lat];
        }
      }

      /* BOTTOM LEFT */
      if (threadIdx.y < PK_ROWS - 1 && threadIdx.x > 0)
      {        
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          lat_input[batch_block_id][bottom-1][lat] = out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 5 + lat];
        }
      }

      /* BOTTOM CENTER */
      if (threadIdx.y < PK_ROWS - 1)
      {        
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          lat_input[batch_block_id][bottom][lat] = out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 6 + lat];
        }
      }

      /* BOTTOM RIGHT */
      if (threadIdx.y < PK_ROWS - 1 && threadIdx.x < PK_COLS - 1)
      {
        for (int lat = 0; lat < LAT_SIZE; lat++)
        {
          lat_input[batch_block_id][bottom+1][lat] = out[batch_block_id][pk_thread_id][DYN_SIZE + LAT_SIZE * 7 + lat];
        }
      }

    }

} // namespace

std::vector<torch::Tensor> graph_cuda_forward(
    torch::Tensor dyn_input,
    torch::Tensor lat_input) {

  auto options = torch::TensorOptions().device(torch::kCUDA).requires_grad(true);

  auto out = torch::zeros({BATCH_SIZE, PK_ROWS * PK_COLS, 
    DYN_SIZE + NEIGHBORS * LAT_SIZE}, options);


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
