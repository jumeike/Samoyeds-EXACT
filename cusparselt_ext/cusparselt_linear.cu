#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cusparseLt.h>
#include <cusparse.h>

#include <vector>

#define CHECK_CUSPARSELT(func)                                                        \
  {                                                                                    \
    cusparseStatus_t status = (func);                                                  \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                           \
      throw std::runtime_error("cuSPARSELt error at " + std::to_string(__LINE__));     \
    }                                                                                  \
  }

torch::Tensor cusparselt_spmm(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be FP16");
  TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be FP16");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

  int64_t m = A.size(0);
  int64_t k = A.size(1);
  int64_t k_b = B.size(0);
  int64_t n = B.size(1);
  TORCH_CHECK(k == k_b, "A.cols must match B.rows");

  auto C = torch::zeros({m, n}, A.options());

  auto order = CUSPARSE_ORDER_ROW;
  auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto type = CUDA_R_16F;
  auto compute_type = CUSPARSE_COMPUTE_32F;
  unsigned alignment = 16;

  int64_t lda = (order == CUSPARSE_ORDER_ROW) ? k : m;
  int64_t ldb = (order == CUSPARSE_ORDER_ROW) ? n : k;
  int64_t ldc = (order == CUSPARSE_ORDER_ROW) ? n : m;

  cusparseLtHandle_t handle;
  cusparseLtMatDescriptor_t matA, matB, matC;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;

  CHECK_CUSPARSELT(cusparseLtInit(&handle));
  CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
      &handle, &matA, m, k, lda, alignment, type, order, CUSPARSELT_SPARSITY_50_PERCENT));
  CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, ldb, alignment, type, order));
  CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(&handle, &matC, m, n, ldc, alignment, type, order));

  CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
      &handle, &matmul, opA, opB, &matA, &matB, &matC, &matC, compute_type));
  CHECK_CUSPARSELT(cusparseLtMatmulAlgSelectionInit(
      &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
  CHECK_CUSPARSELT(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

  size_t workspace_size = 0;
  CHECK_CUSPARSELT(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

  size_t compressed_size = 0;
  size_t compressed_buffer_size = 0;
  CHECK_CUSPARSELT(cusparseLtSpMMACompressedSize(
      &handle, &plan, &compressed_size, &compressed_buffer_size));

  void* dA_compressed = nullptr;
  void* dA_compressed_buffer = nullptr;
  void* d_workspace = nullptr;
  if (compressed_size > 0) {
    TORCH_CHECK(cudaMalloc(&dA_compressed, compressed_size) == cudaSuccess, "cudaMalloc dA_compressed failed");
  }
  if (compressed_buffer_size > 0) {
    TORCH_CHECK(cudaMalloc(&dA_compressed_buffer, compressed_buffer_size) == cudaSuccess, "cudaMalloc dA_compressed_buffer failed");
  }
  if (workspace_size > 0) {
    TORCH_CHECK(cudaMalloc(&d_workspace, workspace_size) == cudaSuccess, "cudaMalloc workspace failed");
  }

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  CHECK_CUSPARSELT(cusparseLtSpMMACompress(
      &handle, &plan, A.data_ptr(), dA_compressed, dA_compressed_buffer, stream));

  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUSPARSELT(cusparseLtMatmul(
      &handle, &plan, &alpha, dA_compressed, B.data_ptr(), &beta,
      C.data_ptr(), C.data_ptr(), d_workspace, &stream, 1));

  if (dA_compressed_buffer) cudaFree(dA_compressed_buffer);
  if (dA_compressed) cudaFree(dA_compressed);
  if (d_workspace) cudaFree(d_workspace);

  CHECK_CUSPARSELT(cusparseLtMatDescriptorDestroy(&matA));
  CHECK_CUSPARSELT(cusparseLtMatDescriptorDestroy(&matB));
  CHECK_CUSPARSELT(cusparseLtMatDescriptorDestroy(&matC));
  CHECK_CUSPARSELT(cusparseLtMatmulPlanDestroy(&plan));
  CHECK_CUSPARSELT(cusparseLtDestroy(&handle));

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cusparselt_spmm", &cusparselt_spmm, "cuSPARSELt SpMM (A sparse, B dense)");
}
