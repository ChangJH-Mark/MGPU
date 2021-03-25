add_executable(test_client ./test_client.cpp ${PROJECT_SOURCE_DIR}/src/api.cpp ${PROJECT_SOURCE_DIR}/src/common/IPC.cpp)
target_link_libraries(test_client pthread)