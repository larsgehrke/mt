#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> graph_cuda_forward(
    torch::Tensor dyn_input,
    torch::Tensor lat_input);

std::vector<torch::Tensor> graph_cuda_backward(
    torch::Tensor d_out);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> graph_forward(
    torch::Tensor dyn_input,
    torch::Tensor lat_input) {
  CHECK_INPUT(dyn_input);
  CHECK_INPUT(lat_input);


  return graph_cuda_forward(dyn_input, lat_input);
}

std::vector<torch::Tensor> graph_backward(
    torch::Tensor d_out) {
  CHECK_INPUT(d_out);

  return graph_cuda_backward(
      d_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &graph_forward, "Graph forward (CUDA)");
  m.def("backward", &graph_backward, "Graph backward (CUDA)");
}
