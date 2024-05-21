
include(FetchContent)
#########################################################################################
# Assimp
message(NOTICE "Generating Assimp build. ")
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL " " FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL " " FORCE)
set(ASSIMP_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
add_subdirectory(assimp)
########################################################################################
#Boost
message(NOTICE "Initializing Boost build. ")
add_subdirectory(boost)
set(BOOST_PROGRAM_OPTIONS_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/boost/libs/program_options/include PARENT_SCOPE)
set(BOOST_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/boost/libs PARENT_SCOPE)
#########################################################################################
# GLM
set(GLM_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/glm PARENT_SCOPE)
#########################################################################################
# Optix
set(OPTIX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Optix PARENT_SCOPE)
#########################################################################################
# STB
set(STB_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/stb PARENT_SCOPE)
#########################################################################################
# Google test
if(${BUILD_TESTS})
    message(NOTICE "Initializing GTest build. ")
    add_subdirectory(googletest)
    set(GTEST_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include PARENT_SCOPE)
endif()


