
#########################################################################################
# Assimp
find_package(Assimp QUIET)
if(NOT Assimp_FOUND)
    message(NOTICE "Assimp not found on the system : Initializing Assimp build from sources. ")
    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL " " FORCE)
    set(ASSIMP_INSTALL OFF CACHE BOOL " " FORCE)
    set(BUILD_SHARED_LIBS ON)
    set(ASSIMP_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
    add_subdirectory(assimp EXCLUDE_FROM_ALL)
endif()
########################################################################################
#Boost
find_package(Boost COMPONENTS program_options random stacktrace)
if(NOT Boost_FOUND)
    message(NOTICE "Boost not found on the system : Initializing Boost build from sources. ")
    set(BOOST_INCLUDE_LIBRARIES program_options stacktrace random)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(boost)

endif()
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
# OpenExr
message(NOTICE "OpenEXR not found on the system : Initializing OpenEXR build from sources. ")
set(BUILD_SHARED_LIBS ON)
set(OpenEXR_BUILD_UTILS OFF)
set(OpenEXR_BUILD_TESTS OFF)
set(OpenEXR_INSTALL OFF)

add_subdirectory(imath EXCLUDE_FROM_ALL)
add_subdirectory(openexr EXCLUDE_FROM_ALL)


#########################################################################################
# Google test
if(${BUILD_TESTS})
    message(NOTICE "Initializing GTest build. ")
    add_subdirectory(googletest EXCLUDE_FROM_ALL)
    set(GTEST_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include PARENT_SCOPE)
endif()


