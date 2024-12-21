cmake_minimum_required(VERSION 3.10)

project(resize_func)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES
    src/main.cpp
    src/imgproc.cpp
    src/utils.cpp
)

add_executable(resize_func ${SOURCE_FILES})

target_link_libraries(resize_func ${OpenCV_LIBS})
