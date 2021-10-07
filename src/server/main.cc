#include <iostream>
#include <csignal>
#include <cstring>
#include "server/server.h"
using namespace  std;

shared_ptr<LogPool> logger;

void sigint_handler(int signal) {
    cout << "receive signal: " << signal << endl;
    cout << "exit" << endl;
    mgpu::destroy_server();
    exit(signal);
}

void init_logger(int argc, char **argv) {
    if(argc > 1 && strcmp(argv[1], "DEBUG") == 0)
        logger = make_shared<LogPool>(DEBUG);
    else
        logger = make_shared<LogPool>(LOG);
}

int main(int argc, char **argv) {
    using namespace mgpu;
    signal(SIGINT,sigint_handler);
    cout << "init server" << endl;
    init_logger(argc, argv);
    auto server = get_server();
    cout << "initialization complete, wait for jobs" << endl;
    server->join();
    pthread_exit(nullptr); // wait for worker thread exit
}
