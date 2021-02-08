#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <config.h>

namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}

template <typename scalar_t>
__global__ void distana_cuda_forward_kernel(
    /*const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell*/
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input) {
  
    // PKs ?
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    // Batch ?
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    input_gate[y][x][0][0] = -7;
}

template <typename scalar_t>
__global__ void distana_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_old_cell,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_cell,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gate_weights) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < d_gates.size(2)){
    const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
    const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
    const auto d_new_cell =
        d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


    d_old_cell[n][c] = d_new_cell;
    const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
    const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

    d_gates[n][0][c] =
        d_input_gate * d_sigmoid(gate_weights[n][0][c]);
    d_gates[n][1][c] =
        d_output_gate * d_sigmoid(gate_weights[n][1][c]);
    d_gates[n][2][c] =
        d_candidate_cell * d_elu(gate_weights[n][2][c]);
  }
}
} // namespace

std::vector<torch::Tensor> distana_cuda_forward(
    torch::Tensor input,
    torch::Tensor pre_weights,
    torch::Tensor lstm_weights,
    torch::Tensor post_weights,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  // Concatinates the tensors in the given dimensionality
  //auto X = torch::cat({old_h, input}, /*dim=*/1);

  // Performs a matrix multiplication of the matrices X and weights.transpose. 
  // The matrix bias is added to the final result.
  //auto gate_weights = torch::addmm(bias, X, pre_weights.transpose(0, 1));

  //const auto batch_size = input.size(0);
  //const auto state_size = old_cell.size(1);

  //auto gates = gate_weights.reshape({batch_size, 3, state_size});
  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);
  auto X = torch::zeros_like(input);
  auto new_pre_weights = torch::zeros_like(pre_weights);
  auto new_lstm_weights = torch::zeros_like(lstm_weights);
  auto new_post_weights = torch::zeros_like(post_weights);


  const int threads = BATCH_SIZE;
  const dim3 blocks(PK_ROWS, PK_COLS);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "distana_forward_cuda", ([&] {
    distana_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
      input.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
       /* gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()*/
        );
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, new_pre_weights, lstm_weights, post_weights};
}

std::vector<torch::Tensor> distana_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gates,
    torch::Tensor weights) {
  auto d_old_cell = torch::zeros_like(new_cell);
  auto d_gates = torch::zeros_like(gates);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "distana_forward_cuda", ([&] {
    distana_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        d_gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        grad_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        input_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        candidate_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));

  auto d_gate_weights = d_gates.flatten(1, 2);
  auto d_weights = d_gate_weights.t().mm(X);
  auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gate_weights.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
