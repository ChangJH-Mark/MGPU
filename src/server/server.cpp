//
// Created by root on 2021/3/11.
//
#include "server/server.h"
#include "server/device.h"
#include "server/scheduler.h"
#include "server/receiver.h"
#include "server/task.h"
#include "common/Log.h"
#include <functional>

shared_ptr<LogPool> logger = make_shared<LogPool>(DEBUG);
using namespace mgpu;

Server *Server::single_instance = nullptr;

Server *mgpu::get_server() {
    if (Server::single_instance != nullptr) {
        return Server::single_instance;
    }

    auto server = new Server;
    Server::single_instance = server;
    server->device = make_shared<Device>();
    server->scheduler = make_shared<Scheduler>();
    server->receiver = make_shared<Receiver>();
    server->conductor = make_shared<Conductor>();
    server->task_holder = make_shared<Task>();

    server->mod["device"] = server->device;
    server->mod["scheduler"] = server->scheduler;
    server->mod["receiver"] = server->receiver;
    server->mod["conductor"] = server->conductor;
    server->mod["task_holder"] = server->task_holder;
    for (const auto &m : server->mod) {
        dout(DEBUG) << "start init mod: " << m.first << dendl;
        m.second->init();
    }

    for (const auto &m : server->mod) {
        m.second->run();
    }
    return Server::single_instance;
}

void Server::join() {
    for (const auto &m : this->mod) {
        if (m.second->joinable) {
            m.second->join();
        }
    }
}

void mgpu::destroy_server() {
    auto server = Server::single_instance;
    if (server == nullptr)
        return;
    for (const auto &m : server->mod) {
        m.second->destroy();
    }
    logger->destroy();
}