cmake_minimum_required(VERSION 3.24)
project(neuralcabin_inference_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenSSL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

add_executable(neuralcabin_server
  src/main.cpp
  src/server.cpp
  src/weight_store.cpp
  src/aes_gcm.cpp
)

target_include_directories(neuralcabin_server
  PRIVATE
    include
)

target_link_libraries(neuralcabin_server
  PRIVATE
    OpenSSL::SSL
    OpenSSL::Crypto
    Boost::system
    Boost::thread
)
