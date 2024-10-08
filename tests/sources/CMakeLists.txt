set(PROCESSING_MODULE_SOURCES "" CACHE INTERNAL "")
set(DATABASE_MODULE_SOURCES "" CACHE INTERNAL "")
set(SCENE_MODULE_SOURCES "" CACHE INTERNAL "" )
set(MESH_MODULE_SOURCES "" CACHE INTERNAL "" )
set(CONTROLLER_MODULE_SOURCES "" CACHE INTERNAL "")
set(TEXTURE_MODULE_SOURCES "" CACHE INTERNAL "")


add_subdirectory(controller)
add_subdirectory(database)
add_subdirectory(mesh)
add_subdirectory(processing)
add_subdirectory(scene)
add_subdirectory(texture)




target_sources(Axomae_test PUBLIC
        ${PROCESSING_MODULE_SOURCES}
        ${DATABASE_MODULE_SOURCES}
        ${SCENE_MODULE_SOURCES}
        ${MESH_MODULE_SOURCES}
        ${CONTROLLER_MODULE_SOURCES}
        ${TEXTURE_MODULE_SOURCES}

)




target_include_directories(Axomae_test PUBLIC
        ${GTEST_INCLUDE_PATH}
        ${TEST_ROOT}
)




