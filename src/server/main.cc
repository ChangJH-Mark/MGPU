#include <iostream>
#include <csignal>
#include "server/server.h"
using namespace  std;

void sigint_handler(int signal) {
    cout << "receive signal: " << signal << endl;
    cout << "exit" << endl;
    mgpu::destroy_server();
    exit(signal);
}

int main() {
    using namespace mgpu;
    signal(SIGINT,sigint_handler);
    cout << "init server" << endl;
    auto server = get_server();
    cout << "initialization complete, wait for jobs" << endl;
    server->join();
    return 0;
}
