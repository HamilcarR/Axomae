
set(QT_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/qt)
set(QT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/qt)
set(QTBASE_BUILD_DIR ${QT_BUILD_DIR}/qtbase)
set(QTBASE_SOURCE_DIR ${QT_SOURCE_DIR}/qtbase)
message(STATUS "Building QT from source in ${QT_BUILD_DIR}")

message(STATUS "Configuring QT6 core library...")
file(MAKE_DIRECTORY ${QTBASE_BUILD_DIR})
execute_process(COMMAND ${QTBASE_SOURCE_DIR}/configure --prefix ${QTBASE_BUILD_DIR}
                WORKING_DIRECTORY ${QTBASE_BUILD_DIR}
)
