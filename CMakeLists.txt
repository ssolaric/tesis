cmake_minimum_required(VERSION 2.8.12)
project(tesis)

set(CMAKE_CXX_STANDARD 11)

file(COPY ./Imagenes DESTINATION .)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(tesis.out tesis.cpp)
target_link_libraries(tesis.out ${OpenCV_LIBS} openvdb pthread Half tbb)

add_executable(contornos.out contornos.cpp)
target_link_libraries(contornos.out ${OpenCV_LIBS})
