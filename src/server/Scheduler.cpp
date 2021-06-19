//
// Created by root on 2021/3/17.
//

#include "server/scheduler.h"
#include "server/task.h"

using namespace mgpu;

void Scheduler::init() {

}

void Scheduler::run() {
    auto server = mgpu::get_server();
    this->conductor = server->get_conductor();
    this->scanner = std::move(std::thread(&Scheduler::do_scan, this));
    auto handler = this->scanner.native_handle();
    pthread_setname_np(handler, "Scheduler");
}

void Scheduler::do_scan() {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    while (!stopped) {
        Task::Jobs undojobs;
        Task::Jobs jobs = TASK_HOLDER->fetch();
        for(auto& job : jobs){
            dout(DEBUG) << " set conduct job " << job.second->get_id() << dendl;
            CONDUCTOR->conduct(job.second);
            dout(DEBUG) << " register conduct job " << job.second->get_id() << dendl;
            TASK_HOLDER->register_doing(job.first, job.second);
        }
        TASK_HOLDER->put_back(undojobs);
    } // while loop
}

void Scheduler::destroy() {
    stopped = true;
}

void Scheduler::join() {
    scanner.detach();
}