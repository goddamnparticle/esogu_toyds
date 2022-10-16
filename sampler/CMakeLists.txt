cmake_minimum_required(VERSION 3.6.2)
set(PROJECT_NAME Sampler)
project(${PROJECT_NAME})
set(CMAKE_BUILD_TYPE Debug)
# set( CMAKE_BUILD_TYPE Release )
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -std=c++17 -Wall")
# Inform CMake where the header files are
include_directories(include)
# Automatically add all *.cpp and *.h files to the project
file(GLOB SOURCES "./src/*.cpp" "./include/*.h")
add_executable(${PROJECT_NAME} ${SOURCES})
# Set the default project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT
                                                            ${PROJECT_NAME})
message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
