#########################################################################################
# Assimp
if(NOT Assimp_FOUND)
    message(NOTICE "Assimp not found on the system : Initializing Assimp build from sources. ")
    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL " " FORCE)
    set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL " " FORCE) # Need for https://github.com/assimp/assimp/issues/5557 
    set(ASSIMP_INSTALL OFF CACHE BOOL " " FORCE)
    set(BUILD_SHARED_LIBS ON)
    set(ASSIMP_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
    add_subdirectory(assimp EXCLUDE_FROM_ALL)
endif()
########################################################################################
#Boost

if(NOT Boost_FOUND)
    message(NOTICE "Boost not found on the system : Initializing Boost build from sources. ")
    set(BOOST_INCLUDE_LIBRARIES program_options stacktrace core random)
    set(BUILD_SHARED_LIBS ON)
    add_subdirectory(boost)

endif()
#########################################################################################
# GLM
set(GLM_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/glm PARENT_SCOPE)
message(NOTICE "Initializing GLM build from sources.")
add_subdirectory(glm EXCLUDE_FROM_ALL)

#########################################################################################
# Optix
set(OPTIX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Optix PARENT_SCOPE)
#########################################################################################
# STB
set(STB_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/stb PARENT_SCOPE)
#########################################################################################
# OpenExr
message(NOTICE "Initializing OpenEXR build from sources. ")
set(BUILD_SHARED_LIBS ON)
set(OPENEXR_BUILD_UTILS OFF)
set(OPENEXR_BUILD_EXAMPLES OFF)
set(OPENEXR_BUILD_TOOLS OFF)
set(OPENEXR_BUILD_TESTS OFF)
set(OPENEXR_INSTALL OFF)
set(OPENEXR_INSTALL_TOOLS OFF)
set(OPENEXR_TEST_LIBRARIES OFF)
set(OPENEXR_TEST_PYTHON OFF)
set(OPENEXR_TEST_TOOLS OFF)
add_subdirectory(imath EXCLUDE_FROM_ALL)
add_subdirectory(openexr EXCLUDE_FROM_ALL)


#########################################################################################
# Google test
if(AXOMAE_BUILD_TESTS)
    message(NOTICE "Initializing GTest build. ")
    set(INSTALL_GTEST OFF)
    set(BUILD_GMOCK OFF)
    set(GTEST_CREATE_SHARED_LIBRARY OFF)
    set(GTEST_LINKED_AS_SHARED_LIBRARY OFF)
    add_subdirectory(googletest EXCLUDE_FROM_ALL)
endif()


#########################################################################################

if(NOT SDL2_FOUND )
    message(NOTICE "SDL2 not found on the system : Initializing SDL2 from sources.")
    add_subdirectory(SDL)
endif()

if(NOT SDL2_image_FOUND)
    message(NOTICE "SDL2_image not found on the system : Initializing SDL2_image from sources.")
    add_subdirectory(SDL_image)
endif()
#########################################################################################
