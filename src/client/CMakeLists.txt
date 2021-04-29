add_library(client api.cpp ${COMMON_SRCS} ${PROJECT_SOURCE_DIR}/include/common/message.h ${PROJECT_SOURCE_DIR}/include/client/api.h)
target_link_libraries(client pthread)