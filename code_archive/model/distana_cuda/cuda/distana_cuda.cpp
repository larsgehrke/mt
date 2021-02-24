#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> distana_cuda_forward(
    torch::Tensor input,
    torch::Tensor pre_weights,
    torch::Tensor lstm_weights,
    torch::Tensor post_weights,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> distana_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> distana_forward(
    torch::Tensor input,
    torch::Tensor pre_weights,
    torch::Tensor lstm_weights,
    torch::Tensor post_weights,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  CHECK_INPUT(input);
  CHECK_INPUT(pre_weights);
  CHECK_INPUT(lstm_weights);
  CHECK_INPUT(post_weights);
  CHECK_INPUT(old_h);
  CHECK_INPUT(old_cell);

  return distana_cuda_forward(input, pre_weights, lstm_weights, post_weights, old_h, old_cell);
}

std::vector<torch::Tensor> distana_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);

  return distana_cuda_backward(
      grad_h,
      grad_cell,
      new_cell,
      input_gate,
      output_gate,
      candidate_cell,
      X,
      gate_weights,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &distana_forward, "DISTANA forward (CUDA)");
  m.def("backward", &distana_backward, "DISTANA backward (CUDA)");
}
