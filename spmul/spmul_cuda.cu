#include <torch/extension.h>
#include <vector>

namespace {
template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> F,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> V,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> Z,
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> offsets,
    const long n_data,
    const long n_vec,
    const long n_link_all,
    const long n_dim) {

    const long data_start = blockIdx.x;
    const long vec_start = threadIdx.x;
    const long dim_start = threadIdx.y;

    for (long i=data_start; i<n_data; i+=gridDim.x) {
        for (long p=vec_start; p<n_vec; p+=blockDim.x) {
            for (long d=dim_start; d<n_dim; d+=blockDim.y) {
                for (long k=0; k<n_link_all; k++) {
                    Z[i][p][d] += F[i][p][k] * V[i][(p+offsets[k]) % n_vec][d];
                }
            }
        }
    }
}

torch::Tensor forward_host(
    torch::Tensor F,
    torch::Tensor V,
    torch::Tensor offsets,
    const long n_block,
    const long n_thread_vec,
    const long n_thread_dim) {

  auto Z = torch::zeros_like(V);
  const long n_data = F.size(0);
  const long n_vec = F.size(1);
  const long n_link_all = F.size(2);
  const long n_dim = V.size(2);
  dim3 n_thread(n_thread_vec, n_thread_dim);

  AT_DISPATCH_FLOATING_TYPES(F.type(), "spmul_forward", ([&] {
    forward_kernel<scalar_t><<<n_block, n_thread>>>(
        F.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        V.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        Z.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        offsets.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        n_data,
        n_vec,
        n_link_all,
        n_dim);
  }));

  return Z;
}

template <typename scalar_t>
__global__ void backward_dJdV_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dJdZ,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> F,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dJdV,
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> offsets,
    const long n_data,
    const long n_vec,
    const long n_link_all,
    const long n_dim) {

    const long data_start = blockIdx.x;
    const long vec_start = threadIdx.x;
    const long dim_start = threadIdx.y;
    for (long i=data_start; i<n_data; i+=gridDim.x) {
        for (long p=vec_start; p<n_vec; p+=blockDim.x) {
            for (long d=dim_start; d<n_dim; d+=blockDim.y) {
                for (long k=0; k<n_link_all; k++) {
                    long j = ( p - offsets[k] + n_vec) % n_vec;
                    dJdV[i][p][d] += F[i][j][k] * dJdZ[i][j][d];
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void backward_dJdF_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dJdZ,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> V,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> dJdF,
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> offsets,
    const long n_data,
    const long n_vec,
    const long n_link_all,
    const long n_dim) {

    const long data_start = blockIdx.x;
    const long vec_start = threadIdx.x;
    const long link_start = threadIdx.y;

    for (long i=data_start; i<n_data; i+=gridDim.x) {
        for (long p=vec_start; p<n_vec; p+=blockDim.x) {
            for (long k=link_start; k<n_link_all; k+=blockDim.y) {
                long j = ( p + offsets[k] ) % n_vec;
                for (long d=0; d<n_dim; d++) {
                    dJdF[i][p][k] += dJdZ[i][p][d] * V[i][j][d];
                }
            }
        }
    }
}

std::vector<torch::Tensor> backward_host(
    torch::Tensor dJdZ,
    torch::Tensor F,
    torch::Tensor V,
    torch::Tensor offsets,
    const long n_block,
    const long n_thread_vec,
    const long n_thread_dim,
    const long n_thread_link) {

  auto dJdV = torch::zeros_like(V);
  auto dJdF = torch::zeros_like(F);
  const long n_data = F.size(0);
  const long n_vec = F.size(1);
  const long n_link_all = F.size(2);
  const long n_dim = V.size(2);

  dim3 n_thread_dJdV(n_thread_vec, n_thread_dim);
  AT_DISPATCH_FLOATING_TYPES(F.type(), "spmul_backward_dJdV", ([&] {
    backward_dJdV_kernel<scalar_t><<<n_block, n_thread_dJdV>>>(
        dJdZ.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        F.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        dJdV.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        offsets.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        n_data,
        n_vec,
        n_link_all,
        n_dim);
  }));


  dim3 n_thread_dJdF(n_thread_vec, n_thread_link);
  AT_DISPATCH_FLOATING_TYPES(F.type(), "spmul_backward_dJdF", ([&] {
    backward_dJdF_kernel<scalar_t><<<n_block, n_thread_dJdF>>>(
        dJdZ.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        V.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        dJdF.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        offsets.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
        n_data,
        n_vec,
        n_link_all,
        n_dim);
  }));

  return {dJdF, dJdV};
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_host", &forward_host, "Sparse Tensor Multiplicatiion Forward (CUDA)");
  m.def("backward_host", &backward_host, "Sparse Tensor Multiplicatiion Backward (CUDA)");
}
