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
#include "common/Log.h"

#define CONDUCTOR get_server()->get_conductor()
#define DEVICES get_server()->get_device()
#define SCHEDULER get_server()->get_scheduler()

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

        friend Server* get_server();
        friend void destroy_server();
    };
    extern Server * get_server();
    extern void destroy_server();
}
#endif //FASTGPU_SERVER_H