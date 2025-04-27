set(VENDOR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ PARENT_SCOPE)
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
    set(BOOST_INCLUDE_LIBRARIES 
      program_options 
      stacktrace 
      core 
      random 
      container_hash
      math
    )
    set(BUILD_SHARED_LIBS ON)
    add_subdirectory(boost)

endif()
#########################################################################################
# GLM
set(GLM_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/glm PARENT_SCOPE)
message(NOTICE "Initializing GLM build from sources.")
add_subdirectory(glm EXCLUDE_FROM_ALL)

#########################################################################################
# Optix8 is present by default in vendor
if(OPTIX_INCLUDE_PATH STREQUAL "" )
  set(OPTIX_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Optix PARENT_SCOPE)
endif()
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
# Embree

if(AXOMAE_USE_EMBREE)
  find_package(TBB REQUIRED)
  find_package(embree 4)  
  if(NOT embree_FOUND)
    message(NOTICE "Embree not found on the system : Initializing Embree from sources.")
    set(EMBREE_ZIP_MODE OFF) 
    set(EMBREE_TUTORIALS OFF) 
    set(EMBREE_TUTORIALS_GLFW OFF)
    set(EMBREE_TESTS OFF)
    add_subdirectory(embree) 
  endif()
endif(AXOMAE_USE_EMBREE)



#########################################################################################
