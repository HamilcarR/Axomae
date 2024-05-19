
include(FetchContent)
#########################################################################################
# Assimp
find_package(assimp)
if(NOT assimp_FOUND)
    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL " " FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
    set(ASSIMP_INSTALL OFF CACHE BOOL " " FORCE)
    add_subdirectory(assimp)

endif()
#########################################################################################
# GLM
set(GLM_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/glm PARENT_SCOPE)

#########################################################################################
# Google test
find_package(GTest)
if(NOT GTest_FOUND)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY  https://github.com/google/googletest.git
            GIT_TAG         release-1.12.0
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest INTERFACE IMPORTED)
    target_link_libraries(GTest::GTest INTERFACE gtest_main)
    include(GoogleTest)
endif()

#########################################################################################
# Optix

set(OPTIX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Optix PARENT_SCOPE)

#########################################################################################
# STB
set(STB_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/stb PARENT_SCOPE)
