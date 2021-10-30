#include <iostream>
#include <csignal>
#include <cstring>
#include "server/server.h"
using namespace  std;

LogPool* logger = nullptr;
int max_level;

void sigint_handler(int signal) {
    dout(LOG) << "receive signal: " << signal << dendl;
    dout(LOG) << "exit" << dendl;
    mgpu::destroy_server();
    exit(signal);
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
    pthread_exit(nullptr); // wait for worker thread exit

    logger->destroy();
    delete logger;
}
