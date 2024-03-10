cmake_minimum_required(VERSION 3.16)
project(Tests VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)

find_package(QT NAMES Qt5 Qt6 REQUIRED COMPONENTS Core)
find_package(Qt6 REQUIRED COMPONENTS Gui OpenGLWidgets Test Widgets)

target_link_libraries(Tests PRIVATE
    GL
    GLEW
    GLU
    Qt::Core
    Qt::Gui
    Qt::OpenGLWidgets
    Qt::Test
    Qt::Widgets
    SDL2
    SDL2_image
    assimp
    cuda
    cudadevrt
    cudart
    dl
    glut
    gtest
    pthread
)

target_compile_options(Tests
    -Wall
    -Wno-unused
    -g
    -pedantic
    -pg
    -std=c++17
)

install(TARGETS Tests
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

qt_generate_deploy_app_script(
    TARGET Tests
    FILENAME_VARIABLE deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
