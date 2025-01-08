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
    src/test.cpp
    src/utils.cpp
    src/simd.cpp
)

# check if the compiler supports AVX2 and AVX 
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)

if(COMPILER_SUPPORTS_AVX2)
    message(STATUS "AVX2 is supported. Enabling USE_AVX2.")
    add_definitions(-DUSE_AVX2)      
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
elseif(MSVC)
    message(STATUS "Checking MSVC for AVX2 support.")
    add_definitions(-DUSE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
else()
    message(FATAL_ERROR "AVX2 is not supported by the compiler.")
endif()

# check if use Test Mode
# add_definitions(-DTEST)

add_executable(resize_func ${SOURCE_FILES})

target_link_libraries(resize_func ${OpenCV_LIBS})
