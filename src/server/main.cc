#include <iostream>
#include <csignal>
#include <cstring>
#include "server/server.h"
using namespace  std;

LogPool* logger = nullptr;
int max_level;

void sigint_handler(int signal) {
    dout(LOG) << "receive signal: " << signal << dendl;
    mgpu::destroy_server();
    dout(LOG) << "start deinit logger Module" << dendl;
    logger->destroy();
    delete logger;
    exit(EXIT_SUCCESS);
}

void init_logger(int argc, char **argv) {
    if(argc > 1 && strcmp(argv[1], "DEBUG") == 0)
        logger = new LogPool(DEBUG);
    else
        logger = new LogPool(LOG);
}

int main(int argc, char **argv) {
    using namespace mgpu;
    signal(SIGINT,sigint_handler);
    init_logger(argc, argv);

    dout(LOG) << "init server" << dendl;
    auto server = get_server();
    dout(LOG) << "initialization complete, wait for jobs" << dendl;
    server->join();
}
