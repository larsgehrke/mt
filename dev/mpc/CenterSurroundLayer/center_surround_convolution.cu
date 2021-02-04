#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>


/** forward kernel
 * c) Implement the forward pass of the Center Surround Convolution
 *  - Write a CUDA-kernel that computes the forward pass form Equation 1. The
 *    kernel should take I, w c , w s, w b and O in form of
 *    torch::PackedTensorAccessor32<scalar t> objects and write into O
 */
template <typename scalar_t>
__global__ void center_surround_convolution_forward_kernel (
        const torch::PackedTensorAccessor32<scalar_t, 4,
        torch::RestrictPtrTraits> I,
        const torch::PackedTensorAccessor32<scalar_t, 2,
        torch::RestrictPtrTraits> w_c,
        const torch::PackedTensorAccessor32<scalar_t, 2,
        torch::RestrictPtrTraits> w_s,
        const torch::PackedTensorAccessor32<scalar_t, 1,
        torch::RestrictPtrTraits> w_b,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> O) {
    // TODO implement the forward kernel
}

/** backward kernels
 * d) Implement the backward pass of the Center Surround Convolution.
 * - Write CUDA kernels to compute the partial derivatives dL_dI, dL_dw_c,
 *   dL_dw_s and dL_dw_b by implementing Equations 3 4 5 and 6.
 */
// dL_dw_c and d_L_dw_s
template <typename scalar_t>
__global__ void dL_dw_kernel (
        const torch::PackedTensorAccessor32<scalar_t, 4,
        torch::RestrictPtrTraits> dL_dO,
        const torch::PackedTensorAccessor32<scalar_t, 4,
        torch::RestrictPtrTraits> I,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        dL_dw_c,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        dL_dw_s) {
    // TODO compute dL_dw_c and dL_dw_s here
}

// dL_dw_b
template <typename scalar_t>
__global__ void dL_dw_b_kernel (
        const torch::PackedTensorAccessor32<scalar_t, 4,
        torch::RestrictPtrTraits> dL_dO,
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>
        dL_dw_b) {
    // TODO compute dL_dw_b here
}

// dL_dI
template <typename scalar_t>
__global__ void dL_dI_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 4,
        torch::RestrictPtrTraits> dL_dO_padded,
        const torch::PackedTensorAccessor32<scalar_t, 2,
        torch::RestrictPtrTraits> w_c,
        const torch::PackedTensorAccessor32<scalar_t, 2,
        torch::RestrictPtrTraits> w_s,
        torch::PackedTensorAccessor32<scalar_t, 4,
        torch::RestrictPtrTraits> dL_dI) {
    // TODO your kernel for dL_dI here
}

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/** c)
 * - Write a c++ function that allocates memory for O and calls your kernel
 *   with appropriate block- and grid dimensions. This function should take I,
 *   w_c, w_s and w_b as torch::Tensor objects and returns a
 *   std::vector<torch::Tensor> the computed O tensor.
 */
std::vector<torch::Tensor> center_surrond_convolution_forward (
        torch::Tensor I,
        torch::Tensor w_c,
        torch::Tensor w_s,
        torch::Tensor w_b) {
    // TODO Use the forward kernel to compute O
    // - Check inputs
    // - Allocate Memory for O
    // - Call the kernel (only for floating types)
    // - return {O}
    return {I.slice(2, 1, -1, 1).slice(3, 1, -1, 1)};
}

/** d)
 *  - Write a c++ function that allocates tensors for the derivatives and calls
 *  the kernels to compute
 *  their content.
 */
std::vector<torch::Tensor> center_surround_convolution_backward (
        torch::Tensor dL_dO,
        torch::Tensor I,
        torch::Tensor w_c,
        torch::Tensor w_s,
        torch::Tensor w_b) {
    // TODO Use the backward kernels to compute the derivatives
    // - Check inputs
    // - Allocate memory for dL_dI, dL_dw_c, dL_dw_s and dL_dw_b
    // - Call the kernels with correct grid and block sizes
    // - return {dL_dI, dL_dw_c, dL_dw_s, dL_dw_b};

	// XXX: Use this padded version of dL_dO to compute dL_dI
	auto dL_dO_padded = torch::constant_pad_nd(dL_dO, torch::IntList({2, 2, 2, 2}), 0);

	return {I, w_c, w_s, w_b};
}

/** c) & d)
 * Export your c++ function to a python module. Call the exported function
 * forward / backward.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &center_surrond_convolution_forward,
            "Center-Surround-Convolution TODO documentation string");
    m.def("backward", &center_surround_convolution_backward,
            "Center-Surround-Convolution TODO documentation string");
}
