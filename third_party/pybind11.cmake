# FetchContent_Declare(
#     pybind11
#     URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
#     DOWNLOAD_EXTRACT_TIMESTAMP ON
# )
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)
