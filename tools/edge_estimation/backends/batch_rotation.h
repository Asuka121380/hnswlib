#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace uq {
namespace detail {

#ifdef UQ_EDGE_ESTIMATION_HAVE_BLAS
extern "C" {
int sgemm_(const char* transa,
           const char* transb,
           const int* m,
           const int* n,
           const int* k,
           const float* alpha,
           const float* a,
           const int* lda,
           const float* b,
           const int* ldb,
           const float* beta,
           float* c,
           const int* ldc);
int sgemv_(const char* trans,
           const int* m,
           const int* n,
           const float* alpha,
           const float* a,
           const int* lda,
           const float* x,
           const int* incx,
           const float* beta,
           float* y,
           const int* incy);
}
#endif

inline const char* batchRotationEngine() {
#ifdef UQ_EDGE_ESTIMATION_HAVE_BLAS
    return "blas_sgemv_sgemm";
#else
    return "portable_scalar_fallback";
#endif
}

inline int checkedBlasSize(size_t value, const char* name) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error(name);
    return static_cast<int>(value);
}

// Input and output are row-major query matrices. Rotation is a row-major R,
// and the contract is output = input * R^T. The BLAS layout matches Faiss'
// LinearTransform::apply_noalloc: GEMV for one row and GEMM for a batch.
inline void rotateQueryBatch(const float* input,
                             size_t query_count,
                             uint32_t dimension,
                             const float* rotation,
                             float* output) {
    if (!input || !rotation || !output || query_count == 0U || dimension == 0U)
        throw std::invalid_argument("invalid batch rotation input");
#ifdef UQ_EDGE_ESTIMATION_HAVE_BLAS
    const int d = checkedBlasSize(dimension, "rotation dimension exceeds BLAS integer range");
    const float one = 1.0f;
    const float zero = 0.0f;
    if (query_count == 1U) {
        const int stride = 1;
        const char trans = 'T';
        sgemv_(&trans, &d, &d, &one, rotation, &d, input, &stride,
               &zero, output, &stride);
        return;
    }
    const int n = checkedBlasSize(query_count, "query batch exceeds BLAS integer range");
    const char transposed = 'T';
    const char not_transposed = 'N';
    sgemm_(&transposed, &not_transposed, &d, &n, &d, &one,
           rotation, &d, input, &d, &zero, output, &d);
#else
    for (size_t query = 0; query < query_count; ++query) {
        const float* source = input + query * dimension;
        float* destination = output + query * dimension;
        for (uint32_t row = 0; row < dimension; ++row) {
            const float* rotation_row = rotation + static_cast<size_t>(row) * dimension;
            float sum = 0.0f;
            for (uint32_t column = 0; column < dimension; ++column)
                sum += rotation_row[column] * source[column];
            destination[row] = sum;
        }
    }
#endif
}

// Build query-major inner-product tables. Each table has M * K entries and
// table[q, m, k] = dot(query[q, m], centroid[m, k]).
inline void buildPqInnerProductTablesBatch(const float* queries,
                                           size_t query_count,
                                           uint32_t dimension,
                                           uint32_t subquantizers,
                                           uint32_t centroid_count,
                                           uint32_t subdimension,
                                           const float* codebook,
                                           float* tables) {
    if (!queries || !codebook || !tables || query_count == 0U ||
        dimension == 0U || subquantizers == 0U || centroid_count == 0U ||
        subdimension == 0U || subquantizers * subdimension != dimension)
        throw std::invalid_argument("invalid batch LUT input");
#ifdef UQ_EDGE_ESTIMATION_HAVE_BLAS
    const int nx = checkedBlasSize(query_count, "query batch exceeds BLAS integer range");
    const int d = checkedBlasSize(dimension, "query dimension exceeds BLAS integer range");
    const int ds = checkedBlasSize(subdimension, "subdimension exceeds BLAS integer range");
    const int k = checkedBlasSize(centroid_count, "centroid count exceeds BLAS integer range");
    const size_t table_stride_value = static_cast<size_t>(subquantizers) * centroid_count;
    const int table_stride = checkedBlasSize(
        table_stride_value, "distance table stride exceeds BLAS integer range");
    const float one = 1.0f;
    const float zero = 0.0f;
    const char transposed = 'T';
    const char not_transposed = 'N';
    for (uint32_t sub = 0; sub < subquantizers; ++sub) {
        const float* centers = codebook +
            static_cast<size_t>(sub) * centroid_count * subdimension;
        const float* query_subspace = queries + static_cast<size_t>(sub) * subdimension;
        float* table_subspace = tables + static_cast<size_t>(sub) * centroid_count;
        sgemm_(&transposed, &not_transposed, &k, &nx, &ds, &one,
               centers, &ds, query_subspace, &d, &zero,
               table_subspace, &table_stride);
    }
#else
    const size_t table_stride = static_cast<size_t>(subquantizers) * centroid_count;
    for (size_t query = 0; query < query_count; ++query) {
        const float* query_row = queries + query * dimension;
        float* table = tables + query * table_stride;
        for (uint32_t sub = 0; sub < subquantizers; ++sub) {
            for (uint32_t code = 0; code < centroid_count; ++code) {
                const float* center = codebook +
                    (static_cast<size_t>(sub) * centroid_count + code) * subdimension;
                float sum = 0.0f;
                for (uint32_t column = 0; column < subdimension; ++column)
                    sum += query_row[sub * subdimension + column] * center[column];
                table[static_cast<size_t>(sub) * centroid_count + code] = sum;
            }
        }
    }
#endif
}

}  // namespace detail
}  // namespace uq
