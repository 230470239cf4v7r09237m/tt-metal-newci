# Define the SFPI version to use

set(SFPI_x86_64_Linux_RELEASE
    "v6.7.0/sfpi-release.tgz"
    "7d955cfce78e0bcf3325151b3bd478c8"
)

set(SFPI_VERSION_JSON_PATH "${PROJECT_BINARY_DIR}/sfpi-version.json")

file(WRITE "${SFPI_VERSION_JSON_PATH}" "{\n")
get_cmake_property(_vars VARIABLES)
foreach(_var ${_vars})
    if(_var MATCHES "^SFPI_.*_RELEASE$")
        string(REPLACE "SFPI_" "" _arch_os "${_var}")
        string(REPLACE "_RELEASE" "" _arch_os "${_arch_os}")
        list(GET ${_var} 0 sfpi_file)
        list(GET ${_var} 1 sfpi_md5)
        file(APPEND "${SFPI_VERSION_JSON_PATH}" "  \"${_arch_os}\": [\"${sfpi_file}\", \"${sfpi_md5}\"],\n")
    endif()
endforeach()
file(APPEND "${SFPI_VERSION_JSON_PATH}" "  \"_end\": null\n}\n")
