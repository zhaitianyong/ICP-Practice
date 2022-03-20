cmake_minimum_required(VERSION 3.10)
project(ICP)

set(CMAKE_CXX_STANDARD 14)
include_directories(./include)
# Eigen
include_directories("/usr/include/eigen3")



# sophus
# find_package(Sophus REQUIRED)
set(Sophus_INCLUDE_DIRS "./3rdparty/Sophus/")
include_directories(${Sophus_INCLUDE_DIRS})

# PCL
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ICP main.cpp ./src/ICP.cpp ./src/PLICP.cpp)
target_link_libraries(ICP  ${PCL_LIBRARIES})