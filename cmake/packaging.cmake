# set(CPACK_GENERATOR "DEB")
# set(CPACK_DEBIAN_PACKAGE_MAINTAINER "support@tenstorrent.com")

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_CONTACT "support@tenstorrent.com")
set(CMAKE_PROJECT_HOMEPAGE_URL "https://tenstorrent.com")
#set(CPACK_DEBIAN_PACKAGE_MAINTAINER "support@tenstorrent.com")
set(CPACK_PACKAGE_NAME tt)

set(CPACK_COMPONENT_sdk_DESCRIPTION "For building apps")
set(CPACK_DEBIAN_sdk_PACKAGE_SECTION "libdevel")
set(CPACK_COMPONENT_runtime_DESCRIPTION "For using apps")
set(CPACK_DEBIAN_runtime_PACKAGE_SECTION "libs")

set(CPACK_COMPONENT_examples_DESCRIPTION "Some examples")
set(CPACK_DEBIAN_doc_PACKAGE_SECTION "doc")
set(CPACK_DEBIAN_sdk_PACKAGE_SECTION "libdevel")

set(CPACK_DEB_COMPONENT_INSTALL TRUE)
set(CPACK_DEBIAN_PACKAGE_VERSION "${VERSION_DEB}")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

set(CPACK_DEBIAN_dev_PACKAGE_NAME "devoverridedev")
set(CPACK_DEBIAN_metalium-dev_PACKAGE_NAME "devoverridedev")

set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)
set(CPACK_DEBIAN_DEBUGINFO_PACKAGE TRUE)

set(CPACK_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
)

# CPACK_DEBIAN_PACKAGE_SOURCE
# CPACK_DEBIAN_<COMPONENT>_PACKAGE_MULTIARCH

#set(CPACK_DEBIAN_PACKAGE_NAME ${CPACK_PACKAGE_NAME})
#set(CPACK_DEBIAN_PACKAGE_DEPENDS "")

# FIXME: use shlibdeps
# set(CPACK_DEBIAN_metalium_PACKAGE_DEPENDS "libnuma1 (>= 2.0)")
# list(APPEND CPACK_DEBIAN_METALIUM_PACKAGE_DEPENDS libnuma1)

set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS TRUE)

set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS TRUE)
# set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS FALSE) # FIXME
# TODO set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS TRUE)
# CPACK_DEBIAN_<COMPONENT>_PACKAGE_CONFLICTS
# CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS

get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
list(
    REMOVE_ITEM
    CPACK_COMPONENTS_ALL
    tt_pybinds # Wow this one is big!
    Unspecified # TODO: audit if there's anything we need to ship here
    Headers # TODO: Where is this coming from?
    Library # TODO: Where is this coming from?
    msgpack-cxx # TODO: Where is this coming from?
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/tt-metalium-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

# Configure the Config file
configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/packaging.d/tt-metalium-config.cmake.in
    ${PROJECT_BINARY_DIR}/tt-metalium-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium
)

# Install the Config and ConfigVersion files
install(
    FILES
        ${PROJECT_BINARY_DIR}/tt-metalium-config.cmake
        ${PROJECT_BINARY_DIR}/tt-metalium-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-metalium
    COMPONENT metalium-dev
)

# install(FILES packaging.d/metalium-config.cmake

cpack_add_component(umd-runtime GROUP metalium)

cpack_add_component(metalium-dev DESCRIPTION "TT-Metalium SDK (add cmpnt)" GROUP metalium-dev)
cpack_add_component(umd-dev GROUP metalium-dev)
cpack_add_component(fmt-core GROUP metalium-dev)
cpack_add_component(magic-enum-dev GROUP metalium-dev)
cpack_add_component(json-dev GROUP metalium-dev)
cpack_add_component_group(metalium-dev DESCRIPTION "TT-Metalium SDK (group)")

cpack_add_component(metalium-examples DEPENDS metalium-dev)

include(CPack)
