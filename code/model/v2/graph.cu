#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/* Custom config file written by Python code*/
#include <config.h>


namespace 
{

    template <typename scalar_t>
    __global__ void graph_forward_kernel(
        const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> dyn_input,
        const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> lat_input,
        torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> out,
        const int lat_size) {

      /*
      [Note that the out variable is the output of this function, 
      but the content will be used later as the input of the network!]

      This kernel implements the concatination of the dynamical and lateral input of the network
      and it implements the graph connections based on the grid of PKs
      (creation of the lateral input from the lateral output).
      Every PK is connected with its surrounding neighbors. 
      There can be up to 8 neighbors (top left, top, top right, left, right, 
      bottom left, bottom, bottom right).
      The corners of the grid have only 3 neighbors, 
      the nodes at the edge of the grid have only 5 neighbors.

      :param dyn_input: Is the dynamical input of the current time step
      :paran lat_input: Is actually the lateral output of the former time step, 
          which now became the input of this kernel
      :param out: Output of this function 
      
      CUDA specific information:
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

      /* Setting the dynamical input of the network */
      for (int dyn = 0; dyn < DYN_SIZE; dyn++)
      {
        out[batch_block_id][pk_thread_id][dyn] = dyn_input[batch_block_id][pk_thread_id][dyn];
      } 

      /* Variables to access the correct neighbors */
      const int top = pk_thread_id - PK_COLS;
      const int bottom = pk_thread_id + PK_COLS;
      const bool y_gt_0 = threadIdx.y > 0;
      const bool x_gt_0 = threadIdx.x > 0;
      const bool y_lt_max = threadIdx.y < PK_ROWS - 1;
      const bool x_lt_max = threadIdx.x < PK_COLS - 1;

      /* Setting the lateral input of the network */
      for (int lat = 0; lat < lat_size; lat++)
      {
        /* TOP GROUP */
        if (y_gt_0)
        {
           /* TOP LEFT */
          if (x_gt_0)
          {
            out[batch_block_id][pk_thread_id][DYN_SIZE + lat] = lat_input[batch_block_id][top-1][lat];
          }

          /* TOP CENTER */
          out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size + lat] = lat_input[batch_block_id][top][lat];
      
          /* TOP RIGHT */
          if (x_lt_max)
          {
            out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 2 + lat] = lat_input[batch_block_id][top+1][lat];
          }
        }
  
        /* LEFT */
        if(x_gt_0)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 3 + lat] = lat_input[batch_block_id][pk_thread_id-1][lat];
        }

        /* RIGHT */
        if(x_lt_max)
        {
          out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 4 + lat] = lat_input[batch_block_id][pk_thread_id+1][lat];
        }

        /* BOTTOM GROUP */
        if (y_lt_max)
        {
          /* BOTTOM LEFT */
          if (x_gt_0)
          { 
            out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 5 + lat] = lat_input[batch_block_id][bottom-1][lat];
          }
          /* BOTTOM CENTER */
          out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 6 + lat] = lat_input[batch_block_id][bottom][lat];
          
          /* BOTTOM RIGHT */
          if (x_lt_max)
          {
            out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 7 + lat] = lat_input[batch_block_id][bottom+1][lat];
          }
        } 
        /* end of for loop for lateral connections*/
      }
      /* end of forward pass*/
    }

    template <typename scalar_t>
    __global__ void graph_backward_kernel(
      const torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_out,
        torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_dyn_input,
        torch::PackedTensorAccessor32<scalar_t,DIMS,torch::RestrictPtrTraits> d_lat_input,
        const int lat_size) 
    {

      /*
      *
      *
      The same as the forward pass, only the other way round.
      *
      *
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
        d_dyn_input[batch_block_id][pk_thread_id][dyn] = d_out[batch_block_id][pk_thread_id][dyn];
      } 

      const int top = pk_thread_id - PK_COLS;
      const int bottom = pk_thread_id + PK_COLS;
      const bool y_gt_0 = threadIdx.y > 0;
      const bool x_gt_0 = threadIdx.x > 0;
      const bool y_lt_max = threadIdx.y < PK_ROWS - 1;
      const bool x_lt_max = threadIdx.x < PK_COLS - 1;

      for (int lat = 0; lat < lat_size; lat++)
      {
        /* TOP GROUP */
        if (y_gt_0)
        {
           /* TOP LEFT */
          if (x_gt_0)
          {
            d_lat_input[batch_block_id][top-1][lat] += d_out[batch_block_id][pk_thread_id][DYN_SIZE + lat];
          }

          /* TOP CENTER */
          d_lat_input[batch_block_id][top][lat] += d_out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size + lat];
      
          /* TOP RIGHT */
          if (x_lt_max)
          {
            d_lat_input[batch_block_id][top+1][lat] += d_out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 2 + lat];
          }
        }
  
        /* LEFT */
        if(x_gt_0)
        {
          d_lat_input[batch_block_id][pk_thread_id-1][lat] += d_out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 3 + lat];
        }

        /* RIGHT */
        if(x_lt_max)
        {
          d_lat_input[batch_block_id][pk_thread_id+1][lat] += d_out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 4 + lat];
        }

        /* BOTTOM GROUP */
        if (y_lt_max)
        {
          /* BOTTOM LEFT */
          if (x_gt_0)
          { 
            d_lat_input[batch_block_id][bottom-1][lat] += d_out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 5 + lat];
          }
          /* BOTTOM CENTER */
          d_lat_input[batch_block_id][bottom][lat] += d_out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 6 + lat];
          
          /* BOTTOM RIGHT */
          if (x_lt_max)
          {
            d_lat_input[batch_block_id][bottom+1][lat] += d_out[batch_block_id][pk_thread_id][DYN_SIZE + lat_size * 7 + lat];
          }
        } 
        /* end of for loop for lateral connections*/
      }
      
      /* end of backward pass */
    }

/* end of namespace */
} 

std::vector<torch::Tensor> graph_cuda_forward(
    torch::Tensor dyn_input,
    torch::Tensor lat_input) {

  /* get the torch tensor options to specify the gpu usage later */
  auto options = torch::TensorOptions().device(torch::kCUDA).requires_grad(true);

  /* set the batch size dynamically by means of the input shape*/
  const auto batch_size = dyn_input.size(0);
  const auto amount_pks = dyn_input.size(1);

  const int lat_size = lat_input.size(2);

  /* allocate enough memory space for the output of the kernel function */
  auto out = torch::zeros({batch_size, amount_pks, 
    DYN_SIZE + NEIGHBORS * lat_size}, options);

  /* map the grid of PKs to the grid of threads per block*/
  const dim3 threads(PK_COLS, PK_ROWS);
  /* map batches to blocks*/
  const dim3 blocks(batch_size);

  /* call the forward kernel function */
  AT_DISPATCH_FLOATING_TYPES(out.type(), "graph_forward_kernel", ([&] {
    graph_forward_kernel<scalar_t><<<blocks, threads>>>(
        dyn_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        lat_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        out.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        lat_size
        );
  }));

  return {out};

}

std::vector<torch::Tensor> graph_cuda_backward(
    torch::Tensor d_out) 
{
   /* set the batch size dynamically by means of the input shape*/
  const auto batch_size = d_out.size(0);
  const auto amount_pks = d_out.size(1);
  const auto total = d_out.size(2);
  const int lat_size = (total - DYN_SIZE)/NEIGHBORS;

  /* get the torch tensor options to specify the gpu usage later */
  auto options = torch::TensorOptions().device(torch::kCUDA).requires_grad(true);

  /* allocate enough memory space for the output of the kernel function */
  auto d_dyn_input = torch::zeros({batch_size, amount_pks, DYN_SIZE}, options);
  auto d_lat_input = torch::zeros({batch_size, amount_pks, lat_size}, options);

  /* map the grid of PKs to the grid of threads per block*/
  const dim3 threads(PK_ROWS, PK_COLS);
  /* map batches to blocks*/
  const dim3 blocks(batch_size);


  /* call the backward kernel function */
  AT_DISPATCH_FLOATING_TYPES(d_dyn_input.type(), "graph_backward_kernel", ([&] {
    graph_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_out.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        d_dyn_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        d_lat_input.packed_accessor32<scalar_t,DIMS,torch::RestrictPtrTraits>(),
        lat_size);
  }));


  return {d_dyn_input, d_lat_input};
}
