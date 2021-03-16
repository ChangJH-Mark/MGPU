include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(test_client ./test_client.cpp ${PROJECT_SOURCE_DIR}/src/client.cpp ${PROJECT_SOURCE_DIR}/src/common/IPC.cpp)