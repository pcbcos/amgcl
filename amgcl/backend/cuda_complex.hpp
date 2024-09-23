#ifndef AMGCL_BACKEND_CUDA_COMPLEX_HPP
#define AMGCL_BACKEND_CUDA_COMPLEX_HPP

#include <memory>
#include <type_traits>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/solver/skyline_lu.hpp>
#include <amgcl/util.hpp>

#include <cusparse_v2.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/scatter.h>
#include <amgcl/backend/cuda.hpp>

namespace amgcl {
namespace backend {



namespace detail {
template<>
cudaDataType cuda_datatype<std::complex<float>>(){
    return CUDA_C_32F;
}

template<>
cudaDataType cuda_datatype<std::complex<double>>(){
    return CUDA_C_64F;
}

template<>
cusparseDnVecDescr_t cuda_vector_description(thrust::device_vector<std::complex<float>> &x) {
    cusparseDnVecDescr_t desc;
    AMGCL_CALL_CUDA(
            cusparseCreateDnVec(
                &desc,
                x.size(),
                thrust::raw_pointer_cast(&x[0]),
                cuda_datatype<std::complex<float>>()
                )
            );
    return desc;
}

template<>
cusparseDnVecDescr_t cuda_vector_description(thrust::device_vector<std::complex<double>> &x) {
    cusparseDnVecDescr_t desc;
    AMGCL_CALL_CUDA(
            cusparseCreateDnVec(
                &desc,
                x.size(),
                thrust::raw_pointer_cast(&x[0]),
                cuda_datatype<std::complex<double>>()
                )
            );
    return desc;
}

// template <>
// cusparseDnVecDescr_t cuda_vector_description(const thrust::device_vector<std::complex<float>> &&x) {
//     cusparseDnVecDescr_t desc;
//     AMGCL_CALL_CUDA(
//             cusparseCreateDnVec(
//                 &desc,
//                 x.size(),
//                 thrust::raw_pointer_cast((void*)&x[0]),
//                 cuda_datatype<std::complex<float>>()
//                 )
//             );
//     return desc;
// }

// template <>
// cusparseDnVecDescr_t cuda_vector_description(const thrust::device_vector<std::complex<double>> &&x) {
//     cusparseDnVecDescr_t desc;
//     AMGCL_CALL_CUDA(
//             cusparseCreateDnVec(
//                 &desc,
//                 x.size(),
//                 thrust::raw_pointer_cast((void*)&x[0]),
//                 cuda_datatype<std::complex<double>>()
//                 )
//             );
//     return desc;
// }

template <>
cusparseSpMatDescr_t cuda_matrix_description(
        size_t nrows,
        size_t ncols,
        size_t nnz,
        thrust::device_vector<int> &ptr,
        thrust::device_vector<int> &col,
        thrust::device_vector<std::complex<float>> &val
        )
{
    cusparseSpMatDescr_t desc;
    AMGCL_CALL_CUDA(
            cusparseCreateCsr(
                &desc,
                nrows,
                ncols,
                nnz,
                thrust::raw_pointer_cast(&ptr[0]),
                thrust::raw_pointer_cast(&col[0]),
                thrust::raw_pointer_cast(&val[0]),
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                detail::cuda_datatype<std::complex<float>>()
                )
            );
    return desc;
}

template <>
cusparseSpMatDescr_t cuda_matrix_description(
        size_t nrows,
        size_t ncols,
        size_t nnz,
        thrust::device_vector<int> &ptr,
        thrust::device_vector<int> &col,
        thrust::device_vector<std::complex<double>> &val
        )
{
    cusparseSpMatDescr_t desc;
    AMGCL_CALL_CUDA(
            cusparseCreateCsr(
                &desc,
                nrows,
                ncols,
                nnz,
                thrust::raw_pointer_cast(&ptr[0]),
                thrust::raw_pointer_cast(&col[0]),
                thrust::raw_pointer_cast(&val[0]),
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                detail::cuda_datatype<std::complex<double>>()
                )
            );
    return desc;
}




}

template<typename R>
struct complex_multiply {
    __host__ __device__
    std::complex<R> operator()(const std::complex<R> &lhs, const std::complex<R> &rhs) const {
        return lhs * conj(rhs);
    }
};


template <>
struct inner_product_impl<
    thrust::device_vector<std::complex<float>>,
    thrust::device_vector<std::complex<float>>
    >
{
    typedef thrust::device_vector<std::complex<float>> vector;
    typedef std::complex<float> V;
    typedef float R;

    static std::complex<float> get(const vector &x, const vector &y)
    {
        return thrust::inner_product(x.begin(), x.end(), y.begin(), V(), thrust::plus<V>(), complex_multiply<R>());
    }
};

template <>
struct inner_product_impl<
    thrust::device_vector<std::complex<double>>,
    thrust::device_vector<std::complex<double>>
    >
{
    typedef thrust::device_vector<std::complex<double>> vector;
    typedef std::complex<double> V;
    typedef double R;

    static std::complex<double> get(const vector &x, const vector &y)
    {
        return thrust::inner_product(x.begin(), x.end(), y.begin(), V(), thrust::plus<V>(), complex_multiply<R>());
    }
};

template <>
class cuda_matrix<std::complex<float>> {
    public:
        typedef std::complex<float> value_type;
        typedef std::complex<float> real;
        cuda_matrix(
                size_t n, size_t m,
                const ptrdiff_t *p_ptr,
                const ptrdiff_t *p_col,
                const std::complex<float>      *p_val,
                cusparseHandle_t handle
                )
            : nrows(n), ncols(m), nnz(p_ptr[n]), handle(handle),
              ptr(p_ptr, p_ptr + n + 1), col(p_col, p_col + nnz), val(p_val, p_val + nnz)
        {
              desc.reset(
                      detail::cuda_matrix_description(nrows, ncols, nnz, ptr, col, val),
                      backend::detail::cuda_deleter()
                      );
        }

        void spmv(
                real alpha, thrust::device_vector<real> const &x,
                real beta,  thrust::device_vector<real>       &y
            ) const
        {
            std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type> xdesc(
                    detail::cuda_vector_description(const_cast<thrust::device_vector<real>&>(x)),
                    backend::detail::cuda_deleter()
                    );
            std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type> ydesc(
                    detail::cuda_vector_description(y),
                    backend::detail::cuda_deleter()
                    );

            size_t buf_size;
            AMGCL_CALL_CUDA(
                    cusparseSpMV_bufferSize(
                        handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        desc.get(),
                        xdesc.get(),
                        &beta,
                        ydesc.get(),
                        detail::cuda_datatype<real>(),
                        CUSPARSE_SPMV_CSR_ALG1,
                        &buf_size
                        )
                    );

            if (buf.size() < buf_size)
                buf.resize(buf_size);

            AMGCL_CALL_CUDA(
                    cusparseSpMV(
                        handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        desc.get(),
                        xdesc.get(),
                        &beta,
                        ydesc.get(),
                        detail::cuda_datatype<real>(),
                        CUSPARSE_SPMV_CSR_ALG1,
                        thrust::raw_pointer_cast(&buf[0])
                        )
                    );
        }

        size_t rows()     const { return nrows; }
        size_t cols()     const { return ncols; }
        size_t nonzeros() const { return nnz;   }
        size_t bytes()    const {
            return
                sizeof(int)  * (nrows + 1) +
                sizeof(int)  * nnz +
                sizeof(real) * nnz;
        }
    public:
        size_t nrows, ncols, nnz;

        cusparseHandle_t handle;

        std::shared_ptr<std::remove_pointer<cusparseSpMatDescr_t>::type> desc;

        thrust::device_vector<int>  ptr;
        thrust::device_vector<int>  col;
        thrust::device_vector<real> val;

        mutable thrust::device_vector<char> buf;

};


template <>
class cuda_matrix<std::complex<double>> {
    public:
        typedef std::complex<double> value_type;
        typedef std::complex<double> real;
        cuda_matrix(
                size_t n, size_t m,
                const ptrdiff_t *p_ptr,
                const ptrdiff_t *p_col,
                const std::complex<double>      *p_val,
                cusparseHandle_t handle
                )
            : nrows(n), ncols(m), nnz(p_ptr[n]), handle(handle),
              ptr(p_ptr, p_ptr + n + 1), col(p_col, p_col + nnz), val(p_val, p_val + nnz)
        {
              desc.reset(
                      detail::cuda_matrix_description(nrows, ncols, nnz, ptr, col, val),
                      backend::detail::cuda_deleter()
                      );
        }

        void spmv(
                real alpha, thrust::device_vector<real> const &x,
                real beta,  thrust::device_vector<real>       &y
            ) const
        {
            std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type> xdesc(
                    detail::cuda_vector_description(const_cast<thrust::device_vector<real>&>(x)),
                    backend::detail::cuda_deleter()
                    );
            std::shared_ptr<std::remove_pointer<cusparseDnVecDescr_t>::type> ydesc(
                    detail::cuda_vector_description(y),
                    backend::detail::cuda_deleter()
                    );

            size_t buf_size;
            AMGCL_CALL_CUDA(
                    cusparseSpMV_bufferSize(
                        handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        desc.get(),
                        xdesc.get(),
                        &beta,
                        ydesc.get(),
                        detail::cuda_datatype<real>(),
                        CUSPARSE_SPMV_CSR_ALG1,
                        &buf_size
                        )
                    );

            if (buf.size() < buf_size)
                buf.resize(buf_size);

            AMGCL_CALL_CUDA(
                    cusparseSpMV(
                        handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        desc.get(),
                        xdesc.get(),
                        &beta,
                        ydesc.get(),
                        detail::cuda_datatype<real>(),
                        CUSPARSE_SPMV_CSR_ALG1,
                        thrust::raw_pointer_cast(&buf[0])
                        )
                    );
        }

        size_t rows()     const { return nrows; }
        size_t cols()     const { return ncols; }
        size_t nonzeros() const { return nnz;   }
        size_t bytes()    const {
            return
                sizeof(int)  * (nrows + 1) +
                sizeof(int)  * nnz +
                sizeof(real) * nnz;
        }
    public:
        size_t nrows, ncols, nnz;

        cusparseHandle_t handle;

        std::shared_ptr<std::remove_pointer<cusparseSpMatDescr_t>::type> desc;

        thrust::device_vector<int>  ptr;
        thrust::device_vector<int>  col;
        thrust::device_vector<real> val;

        mutable thrust::device_vector<char> buf;

};


}  // namespace backend
}  // namespace amgcl


namespace std {

template <typename V>
__host__ __device__ bool operator<(const std::complex<V> &a, const std::complex<V> &b) {
    return std::abs(a) < std::abs(b);
}

} // namespace std


namespace amgcl {
namespace math {

/// Specialization that extracts the scalar type of a complex type.
template <class T>
struct scalar_of< std::complex<T> > {
    typedef T type;
};

/// Replace scalar type in the complex type.
template <class T, class S>
struct replace_scalar<std::complex<T>, S> {
    typedef std::complex<S> type;
};

/// Specialization of conjugate transpose for scalar complex arguments.
template <typename T>
struct adjoint_impl< std::complex<T> >
{
    typedef std::complex<T> return_type;

    __host__ __device__ static std::complex<T> get(std::complex<T> x) {
        return std::conj(x);
    }
};

/// Default implementation for inner product
/** \note Used in adjoint() */
template <typename T>
struct inner_product_impl< std::complex<T> > {
    typedef std::complex<T> return_type;

    __host__ __device__ static return_type get(std::complex<T> x, std::complex<T> y) {
        return x * std::conj(y);
    }
};

/// Specialization of constant element for complex type.
template <typename T>
struct constant_impl< std::complex<T> >
{
    __host__ __device__ static std::complex<T> get(T c) {
        return std::complex<T>(c, c);
    }
};

}  // namespace math

}
#endif