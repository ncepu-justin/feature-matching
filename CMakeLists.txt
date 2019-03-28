cmake_minimum_required(VERSION 3.13)
project(orb_gms)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV  REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS " OpenCV version:${OpenCV_VERSION}")


add_executable(orb_gms grid-gms.cpp)
target_link_libraries(orb_gms ${OpenCV_LIBS})

project(grid_gms)
add_executable(grid_gms grid-gms.cpp)
target_link_libraries(grid_gms ${OpenCV_LIBS})

project(orb)
add_executable(orb orb.cpp)
target_link_libraries(orb ${OpenCV_LIBS})

project(orb__gms)
add_executable(orb__gms orb_gms.cpp)
target_link_libraries(orb__gms ${OpenCV_LIBS})

project(gridfast)
add_executable(gridfast main.cpp)
target_link_libraries(gridfast ${OpenCV_LIBS})