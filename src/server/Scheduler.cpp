//
// Created by root on 2021/3/17.
//

#include "server/scheduler.h"

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
}

void Scheduler::destroy() {
    dout(LOG) << "start destroy Scheduler Module" << dendl;
    stopped = true;
}

void Scheduler::join() {
    scanner.detach();
}