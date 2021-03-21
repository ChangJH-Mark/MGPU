//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_SERVER_H
#define FASTGPU_SERVER_H
#include <thread>
#include <iostream>
#include <mutex>
#include <list>
#include <map>
#include "mod.h"
#include "commands.h"
#include "scheduler.h"
#include "device.h"
#include "receiver.h"
#include "conductor.h"

using namespace std;
namespace mgpu {
    class Server {
    public:
        Server() = default;
        void join();

        typedef list<shared_ptr<Command>> List;

    private:
        static Server *single_instance;
        std::map<string, shared_ptr<Module>> mod;

    private: // mgpu-Streams
        std::mutex map_mtx;
        std::map<uint, std::pair<shared_ptr<std::mutex>, shared_ptr<List>>> task_map; // key: pid << 16 + stream, value: <mutex, CmdList>

        friend Server* get_server();
        friend void destroy_server();
        friend class Receiver;
        friend class Scheduler;
    };
    extern Server * get_server();
    extern void destroy_server();
}
#endif //FASTGPU_SERVER_H
