
set(QT_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/vendor/qt)
set(QT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/qt)
set(QTBASE_BUILD_DIR ${QT_BUILD_DIR}/qtbase)
set(QTBASE_SOURCE_DIR ${QT_SOURCE_DIR}/qtbase)

message(STATUS "Building QT from source : ${QT_SOURCE_DIR}")
message(STATUS "Build directory is :  ${QT_BUILD_DIR}")

message(STATUS "Initializing qt repository ...")

message(STATUS "Configuring qtbase...")
if(NOT EXISTS "${QT_BUILD_DIR}")
    file(MAKE_DIRECTORY ${QT_BUILD_DIR})
endif()
execute_process(COMMAND bash ${QT_SOURCE_DIR}/configure
        -submodules qtbase
        -prefix ${QT_BUILD_DIR}

        WORKING_DIRECTORY ${QT_BUILD_DIR}
        ERROR_VARIABLE ERROR_MSG
)

message(STATUS ${ERROR_MSG})

message(STATUS "Building qtbase ...")
execute_process(COMMAND cmake --build . --parallel 4
       WORKING_DIRECTORY ${QT_BUILD_DIR}
)

message(STATUS "Installing qtbase ...")
execute_process(COMMAND cmake --install .
        WORKING_DIRECTORY ${QT_BUILD_DIR}
)


list(APPEND CMAKE_MODULE_PATH ${QT_BUILD_DIR}/lib/cmake)
list(APPEND CMAKE_PREFIX_PATH ${QT_BUILD_DIR}/lib/cmake)
message(STATUS "Setting up local qt build path ...")
find_package(Qt6
        COMPONENTS Gui OpenGLWidgets Widgets
        REQUIRED
        PATHS ${CMAKE_PREFIX_PATH}
        CONFIG
        NO_DEFAULT_PATH
)

message(STATUS "Qt6_DIR: ${Qt6_DIR}")
message(STATUS "Qt6Gui_DIR: ${Qt6Gui_DIR}")
message(STATUS "Qt6OpenGLWidgets_DIR: ${Qt6OpenGLWidgets_DIR}")
message(STATUS "Qt6Widgets_DIR: ${Qt6Widgets_DIR}")














