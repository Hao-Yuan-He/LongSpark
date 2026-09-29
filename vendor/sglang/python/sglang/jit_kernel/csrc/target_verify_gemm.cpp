// Bounded BF16 algorithm experiment. No weights, precision or runtime dispatch changes.
#include <cublasLt.h>
#include <vector>
#include <cstring>

struct Plan {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulPreference_t pref{};
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
  ~Plan() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (op) cublasLtMatmulDescDestroy(op);
    if (handle) cublasLtDestroy(handle);
  }
};

extern "C" void* make_plan(int m, int n, int k, int requested, size_t workspace, int* count) {
  auto* p = new Plan;
  *count = 0;
  cublasOperation_t trans = CUBLAS_OP_T;
  // Row-major X[M,K] W[N,K] -> Y[M,N], expressed as column-major Y^T=W X^T.
  if (cublasLtCreate(&p->handle) ||
      cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) ||
      cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)) ||
      cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16BF, k, n, k) ||
      cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16BF, k, m, k) ||
      cublasLtMatrixLayoutCreate(&p->c, CUDA_R_16BF, n, m, n) ||
      cublasLtMatmulPreferenceCreate(&p->pref) ||
      cublasLtMatmulPreferenceSetAttribute(p->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &workspace, sizeof(workspace))) {
    delete p; return nullptr;
  }
  p->algos.resize(requested);
  if (cublasLtMatmulAlgoGetHeuristic(p->handle, p->op, p->a, p->b, p->c, p->c,
                                    p->pref, requested, p->algos.data(), count)) {
    delete p; return nullptr;
  }
  p->algos.resize(*count);
  return p;
}

extern "C" int matmul(void* raw, int index, void* x, void* w, void* y, void* workspace,
                       size_t workspace_bytes, void* stream) {
  auto* p = static_cast<Plan*>(raw);
  if (index < 0 || index >= static_cast<int>(p->algos.size())) return -1;
  const float alpha = 1.f, beta = 0.f;
  return cublasLtMatmul(p->handle, p->op, &alpha, w, p->a, x, p->b, &beta,
                        y, p->c, y, p->c, &p->algos[index].algo,
                        workspace, workspace_bytes, static_cast<cudaStream_t>(stream));
}

extern "C" void release_plan(void* raw) { delete static_cast<Plan*>(raw); }

extern "C" size_t lt_version() { return cublasLtGetVersion(); }
extern "C" size_t export_algorithm(void* raw, int index, void* output, size_t size) {
  auto* p = static_cast<Plan*>(raw);
  if (index < 0 || index >= static_cast<int>(p->algos.size())) return 0;
  if (size < sizeof(cublasLtMatmulAlgo_t)) return sizeof(cublasLtMatmulAlgo_t);
  std::memcpy(output, &p->algos[index].algo, sizeof(cublasLtMatmulAlgo_t));
  return sizeof(cublasLtMatmulAlgo_t);
}
