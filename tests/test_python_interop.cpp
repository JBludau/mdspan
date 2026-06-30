#include <mdspan/mdspan.hpp>
#include <mdspan_to_dlpack.hpp>

#include <pybind11/embed.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};

    // Define the type for our mdspan
    using Extents = MDSPAN_IMPL_STANDARD_NAMESPACE :: extents<int, 1>;
    using Layout = MDSPAN_IMPL_STANDARD_NAMESPACE :: layout_right;
    using MdspanType = MDSPAN_IMPL_STANDARD_NAMESPACE :: mdspan<float, Extents, Layout>;

    // Data and mdspan
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    MdspanType v(data.data(), 6);

    try {
        // Define Python function to modify data
        py::exec(R"(
import numpy as np
def process_array(pack):
    # np.asarray() creates a writable NumPy view of the C++ buffer
    arr = np.from_dlpack(pack)
    print(f"Python: Received array shape {arr.shape}")
    arr[0] = 99.0  # Zero-copy modification
        )");

        // Call Python function
        auto pack = interop :: to_dlpack_tensor(v);
        py::module_::import("__main__").attr("process_array")(interop::pass_dlpack_without_ownership(pack.get()));

        std::cout << "C++: View(0) after Python modification: " << v[0]<< std::endl;

    } catch (py::error_already_set &e) {
        std::cerr << "Python Error: " << e.what() << std::endl;
    }

    return 0;
}
