cmake_minimum_required(VERSION 2.6)
project(Pastec)

include_directories(include include/orb)

set(SOURCES     src/imagereranker.cpp
                src/imagererankerransac.cpp
                src/imageloader.cpp
                src/orb/orbfeatureextractor.cpp
                src/orb/orbindex.cpp
                src/orb/orbsearcher.cpp
                src/orb/orbwordindex.cpp)

set(BIN_SOURCES src/main.cpp
                src/httpserver.cpp
                src/requesthandler.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(pastec ${SOURCES})
