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

    const int o = blockIdx.x * blockDim.x + threadIdx.x;// Calculate this threads output index
    const int n = blockIdx.y * blockDim.y + threadIdx.y;// Calculate this threads batch index

    // 28 x 28 Images

    scalar_t result = 0.0f;

    idx_o = 29 + (o/27)*28 + o%27;

    for (int i = -1; i< 2; ++i)
    {
        for (int j = -1; j< 2; ++j)
        {
            idx = idx_o + i + 28*j;

            if(idx >= 0 && idx < 784)
            {
                if(idx == o)
                {
                    result += I[n][o] * w_c;
                }
                else
                {
                    result += I[n][o] * w_s;
                }
            }
        }
    }

    O[n][o] = result;
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

    const int i= blockIdx.x* blockDim.x+ threadIdx.x;// Calculate this thread's input index
    const int n = blockIdx.y* blockDim.y+ threadIdx.y;// Calculate this thread's batch index

    

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
std::vector<torch::Tensor> center_surround_convolution_forward (
        torch::Tensor I,
        torch::Tensor w_c,
        torch::Tensor w_s,
        torch::Tensor w_b) {
    // TODO Use the forward kernel to compute O
    // - Check inputs
    // - Allocate Memory for O
    // - Call the kernel (only for floating types)
    // - return {O}

    CHECK_INPUT(I);
    CHECK_INPUT(w_c);
    CHECK_INPUT(w_s);
    CHECK_INPUT(w_b);

    // 27x27 =729
    // batch size 100

    const auto batch_size = I.size(0);// Obtain the batch size
    const auto input_size = I.size(1);// Obtain the I size

    auto O = torch::empty({batch_size, 729}, input_size.options());// Create an uninitialized output tensor

    const dim3 block_dim(32, 32);// Use 1024 element blocks
    const dim3 grid_dim((729 + 31) / 32, (100 + 31) / 32);// Map output elements to x and batch elements to y

    // -> One thread per calculated output element
    AT_DISPATCH_FLOATING_TYPES(// Executes the kernel only if I.type() is a floating point type
    I.type(), "center_surround_convolution", ([&] { // and sets the kernel's template parameter accordingly.
    center_surround_convolution_forward_kernel<scalar_t><<<grid_dim, block_dim>>>(
    I.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
    w_c.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    w_s.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    w_b.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
    O.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    return {O};    
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
