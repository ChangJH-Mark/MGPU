//
// Created by root on 2021/3/17.
//

#include "server/scheduler.h"

using namespace mgpu;

void Scheduler::init() {

}

void Scheduler::run() {
    this->scanner = std::move(std::thread(&Scheduler::do_scan, this));
    auto handler = this->scanner.native_handle();
    pthread_setname_np(handler, "Scheduler");
}

void Scheduler::do_scan() {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    auto server = get_server();
    while (1) {
        server->map_mtx.lock();
        for(auto iter = server->task_map.begin(); iter != server->task_map.end();)
        {
            // clear empty list
            if(iter->second.second->empty())
            {
                server->task_map.erase(iter++);
                continue;
            }
            auto mtx = iter->second.first;
            auto list = iter->second.second;
            lock_guard<std::mutex> list_guard(*mtx);
            shared_ptr<Command> cmd = list->front();
            list->pop_front();
            switch (cmd->get_type()) {
                case MSG_CUDA_MALLOC: {
                    auto conductor = dynamic_pointer_cast<Conductor>(server->mod["conductor"]);
                    conductor->conduct(cmd);
                    break;
                }
            }
            ++iter;
        }
        server->map_mtx.unlock();
    } // while loop
}

void Scheduler::destroy() {

}

void Scheduler::join() {
    scanner.detach();
}