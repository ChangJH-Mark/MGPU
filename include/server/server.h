//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_SERVER_H
#define FASTGPU_SERVER_H
#include <thread>
#include <iostream>
#include <mutex>
#include <vector>
#include <map>
#include "mod.h"
#include "commands.h"
#include "scheduler.h"
#include "device.h"
#include "receiver.h"
#include "conductor.h"

using namespace std;
namespace mgpu {
    class Server;
    extern Server * init_server();

    class Server {
    public:
        Server() = default;;
        void join();

    private:
        static Server *single_instance;
        mutex v_mutex;
        vector<Command> vec;
        map<string, shared_ptr<Module>> mod;
        friend Server* init_server();
    };
}
#endif //FASTGPU_SERVER_H
