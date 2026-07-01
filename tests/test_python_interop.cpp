#include <mdspan/mdspan.hpp>
#include <mdspan_to_dlpack.hpp>

#include <Python.h>
#include <iostream>
#include <vector>

int main() {
    Py_Initialize();

    // Define the type for our mdspan
    using Extents = MDSPAN_IMPL_STANDARD_NAMESPACE :: extents<int, 1>;
    using Layout = MDSPAN_IMPL_STANDARD_NAMESPACE :: layout_right;
    using MdspanType = MDSPAN_IMPL_STANDARD_NAMESPACE :: mdspan<float, Extents, Layout>;

    // Data and mdspan
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    MdspanType v(data.data(), 6);

    // Define Python function to modify data
    int rc = PyRun_SimpleString(R"(
import numpy as np

# NumPy's from_dlpack consumes the __dlpack__ protocol, not a bare capsule,
# so wrap the capsule in an object exposing that protocol.
class _DLPackCapsule:
    def __init__(self, capsule):
        self._capsule = capsule
    def __dlpack__(self, stream=None):
        return self._capsule
    def __dlpack_device__(self):
        return (1, 0)  # (kDLCPU, device_id=0)

def process_array(pack):
    # np.from_dlpack() creates a writable NumPy view of the C++ buffer
    arr = np.from_dlpack(_DLPackCapsule(pack))
    print(f"Python: Received array shape {arr.shape}")
    arr[0] = 99.0  # Zero-copy modification
    )");

    if (rc != 0) {
        PyErr_Print();
        Py_Finalize();
        return 1;
    }

    // Call Python function
    auto pack = interop ::to_dlpack_tensor(v);

    PyObject* capsule = interop::pass_versioned_dlpack_without_ownership(pack.get());
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* func = PyObject_GetAttrString(main_module, "process_array");
    PyObject* result = PyObject_CallFunctionObjArgs(func, capsule, nullptr);

    if (result == nullptr) {
        PyErr_Print();
    }

    Py_XDECREF(result);
    Py_DECREF(func);
    Py_DECREF(capsule);

    std::cout << "C++: View(0) after Python modification: " << v[0] << std::endl;

    Py_Finalize();
    return 0;
}
