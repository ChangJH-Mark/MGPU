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
#include <unordered_map>
#include <memory>
#include <future>
#include <condition_variable>
#include "mod.h"
#include "commands.h"
#include "scheduler.h"
#include "device.h"
#include "receiver.h"
#include "conductor.h"
#include "task.h"

using namespace std;
namespace mgpu {
    class Scheduler;
    class Device;
    class Receiver;
    class Conductor;

    class Server {
    public:
        Server() = default;
        void join();

        typedef list<shared_ptr<Command>> List;

    public:
        shared_ptr<Device> get_device(){return device;}
        shared_ptr<Scheduler> get_scheduler(){return scheduler;}
        shared_ptr<Receiver> get_receiver(){return receiver;}
        shared_ptr<Conductor> get_conductor(){return conductor;}
    private:
        static Server *single_instance;
        std::map<string, shared_ptr<Module>> mod;
        shared_ptr<Device> device;
        shared_ptr<Scheduler> scheduler;
        shared_ptr<Receiver> receiver;
        shared_ptr<Conductor> conductor;

    private: // mgpu-Streams
        std::mutex map_mtx;
        std::condition_variable cv;
        std::map<TASK_KEY, std::pair<shared_ptr<std::mutex>, shared_ptr<List>>, CompareKey> task_map; // key: TASK_KEY, value: <mutex, CmdList>
        std::map<TASK_KEY, shared_ptr<bool>, CompareKey> available_map; // key: pid << 16 + stream, value: isStreamBlocked

        friend Server* get_server();
        friend void destroy_server();
        friend class Receiver;
        friend class Scheduler;
        friend class Conductor;
    };
    extern Server * get_server();
    extern void destroy_server();
}
#endif //FASTGPU_SERVER_H
