#pragma once

#include <dlpack/dlpack.h>
#include <Python.h>

#include <cstddef>
#include <type_traits>
#if defined(MDSPAN_IMPL_HAS_CUDA) && defined(__NVCC__) && (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__ * 10 >= 1260)
#include <cuda/std/limits>
#else
#include <limits>
#endif

#include <mdspan/mdspan.hpp>

namespace interop{
namespace detail{

template <typename Type>
requires std::is_same_v<Type, DLManagedTensorVersioned> || std::is_same_v<Type, DLManagedTensor>
void NonOwningDLPackDeleter(Type* managed_tensor) {
    if (managed_tensor) {
        delete[] managed_tensor->dl_tensor.shape;
        if (managed_tensor->dl_tensor.strides) {
            delete[] managed_tensor->dl_tensor.strides;
        }
        delete managed_tensor;
    }
}

template <typename Type>
[[nodiscard]] inline ::DLDataType type_to_dlpack () {

  if constexpr (std::is_floating_point_v<Type>)
  {
    return ::DLDataType{::kDLFloat, sizeof(Type) * CHAR_BIT, 1};
  }
  else
  {
    throw(std::invalid_argument("No known conversion of type into DLDataType"));
  }
}

template <size_t Rank>
struct dlpack_tensor
{
  std::array<std::int64_t, Rank> shape{};
  std::array<std::int64_t, Rank> strides{};
  ::DLTensor tensor{};

  [[nodiscard]] ::DLTensor get() const& noexcept
  {
    auto ret    = tensor;
    ret.shape   = Rank > 0 ? const_cast<std::int64_t*>(shape.data()) : nullptr;
    ret.strides = Rank > 0 ? const_cast<std::int64_t*>(strides.data()) : nullptr;
    return ret;
  }

  ::DLTensor get() const&& = delete;
};

template <typename ElementType, typename Extents, typename Layout, typename Accessor>
[[nodiscard]] dlpack_tensor<Extents::rank()>
to_dlpack(const MDSPAN_IMPL_STANDARD_NAMESPACE :: mdspan<ElementType, Extents, Layout, Accessor>& m,
            ::DLDeviceType device_type,
            int device_id)
{
  static_assert(std::is_pointer_v<typename Accessor::data_handle_type>, "data_handle_type must be a pointer");
  using element_type = std::remove_cv_t<ElementType>;
  dlpack_tensor<Extents::rank()> wrapper{};
  auto& tensor  = wrapper.tensor;
  tensor.data   = m.size() > 0 ? const_cast<element_type*>(m.data_handle()) : nullptr;
  tensor.device = ::DLDevice{device_type, device_id};
  tensor.ndim   = static_cast<int>(m.rank());
  tensor.dtype  = detail::type_to_dlpack<std::remove_cv_t<ElementType>>();

  if constexpr (Extents::rank() > 0)
  {
    constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
    for (std::size_t i = 0; i < m.rank(); ++i)
    {
      if (!(m.extent(i) < maximum))
      {
        throw(std::invalid_argument("Extent of mdspan is too large to fit into dlpack"));
      }
      if (!(m.stride(i) < maximum))
      {
        throw(std::invalid_argument("Stride of mdspan is too large to fit into dlpack"));
      }
      wrapper.shape[i]   = static_cast<std::int64_t>(m.extent(i));
      wrapper.strides[i] = static_cast<std::int64_t>(m.stride(i));
    }
  }
  tensor.byte_offset = 0;
  return wrapper;
}
}

// Public API
template <typename ElementType, typename Extents, typename Layout, typename Accessor>
[[nodiscard]] detail::dlpack_tensor<Extents::rank()>
to_dlpack_tensor(const MDSPAN_IMPL_STANDARD_NAMESPACE :: mdspan<ElementType, Extents, Layout, Accessor>& m)
{
  using mdspan_type = MDSPAN_IMPL_STANDARD_NAMESPACE :: mdspan<ElementType, Extents, Layout, Accessor>;

#if defined MDSPAN_IMPL_HAS_CUDA
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, m.data_handle());

    MDSPAN_IMPL_PRECONDITION(error == cudaErrorInvalidValue);

    if (attributes.type == cudaMemoryTypeManaged)
    {
      return detail::to_dlpack(mdspan_type{m}, ::kDLCUDA, attirbutes.device);
    }
    else if (attributes.type == cudaMemoryTypeDevice)
    {
      return detail::to_dlpack(mdspan_type{m}, ::kDLCUDAManaged, 0);
    }
    else if (attributes.type == cudaMemoryTypeHost)
    {
      return detail::to_dlpack(mdspan_type{m}, ::kDLCUDAHost, 0);
    }
    else // cudaMemoryTypeUnregistered
    {
      return detail::to_dlpack(mdspan_type{m}, ::kDLCPU, 0);
    }

#elif defined MDSPAN_IMPL_HAS_HIP
    hipPointerAttribute_t attributes;
    hipError_t error = hipPointerGetAttributes(&attributes, m.data_handle());

    MDSPAN_IMPL_PRECONDITION(error == hipErrorInvalidValue);

    if (attributes.type == hipMemoryTypeManaged)
    {
      return detail::to_dlpack(mdspan_type{m}, ::kDLROCM, attirbutes.device);
    }
    else if (attributes.type == hipMemoryTypeDevice)
    {
      return detail::to_dlpack(mdspan_type{m}, ::kDLROCMManaged, 0);
    }
    else if (attributes.type == hipMemoryTypeHost)
    {
      return detail::to_dlpack(mdspan_type{m}, ::kDLROCMHost, 0);
    }
    else // Unregistered
    {
      return detail::to_dlpack(mdspan_type{m}, ::kDLCPU, 0);
    }

#elif defined MDSPAN_IMPL_HAS_SYCL
#error "SYCL is not supported right now"

#else // assume it is cpu-like
  return detail::to_dlpack(mdspan_type{m}, ::kDLCPU, 0);
#endif
}

PyObject* pass_versioned_dlpack_without_ownership(const ::DLTensor& src_tensor) {
    ::DLManagedTensorVersioned* managed_tensor = new ::DLManagedTensorVersioned();

    managed_tensor->version.major = DLPACK_MAJOR_VERSION;
    managed_tensor->version.minor = DLPACK_MINOR_VERSION;
    managed_tensor->manager_ctx   = nullptr;
    managed_tensor->flags         = 0;
    managed_tensor->dl_tensor     = src_tensor;

    int64_t* shape_copy = new int64_t[src_tensor.ndim];
    for (int64_t i = 0; i < src_tensor.ndim; ++i) {
        shape_copy[i] = src_tensor.shape[i];
    }
    managed_tensor->dl_tensor.shape = shape_copy;

    if (src_tensor.strides != nullptr) {
        int64_t* strides_copy = new int64_t[src_tensor.ndim];
        for (int64_t i = 0; i < src_tensor.ndim; ++i) {
            strides_copy[i] = src_tensor.strides[i];
        }
        managed_tensor->dl_tensor.strides = strides_copy;
    } else {
        managed_tensor->dl_tensor.strides = nullptr;
    }

    managed_tensor->deleter = detail::NonOwningDLPackDeleter;

    PyObject* capsule = PyCapsule_New(managed_tensor, "dltensor_versioned", [](PyObject* cap) {
        if (PyCapsule_IsValid(cap, "dltensor_versioned")) {
            auto* mt = static_cast<DLManagedTensorVersioned*>(
                PyCapsule_GetPointer(cap, "dltensor_versioned"));
            if (mt && mt->deleter) {
                mt->deleter(mt);
            }
        }
    });
    if (capsule == nullptr) {
        detail::NonOwningDLPackDeleter(managed_tensor);
        return nullptr;
    }

    return capsule;
}


PyObject* pass_dlpack_without_ownership(const ::DLTensor& src_tensor) {
    ::DLManagedTensor* managed_tensor = new ::DLManagedTensor();

    managed_tensor->dl_tensor = src_tensor;

    int64_t* shape_copy = new int64_t[src_tensor.ndim];
    for (int64_t i = 0; i < src_tensor.ndim; ++i) {
        shape_copy[i] = src_tensor.shape[i];
    }
    managed_tensor->dl_tensor.shape = shape_copy;

    if (src_tensor.strides != nullptr) {
        int64_t* strides_copy = new int64_t[src_tensor.ndim];
        for (int64_t i = 0; i < src_tensor.ndim; ++i) {
            strides_copy[i] = src_tensor.strides[i];
        }
        managed_tensor->dl_tensor.strides = strides_copy;
    } else {
        managed_tensor->dl_tensor.strides = nullptr;
    }

    managed_tensor->deleter = detail::NonOwningDLPackDeleter;

    PyObject* capsule = PyCapsule_New(managed_tensor, "dltensor", [](PyObject* cap) {
        // Safe fallback in case Python drops the capsule without consuming it
        DLManagedTensor* mt = (DLManagedTensor*)PyCapsule_GetPointer(cap, "dltensor");
        if (mt) {
            mt->deleter(mt);
        }
    });

    return capsule;
}

}
