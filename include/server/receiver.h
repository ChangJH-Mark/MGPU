//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_RECEIVER_H
#define FASTGPU_RECEIVER_H
#include <thread>
#include <atomic>
#include <map>
#include <sys/socket.h>
#include "mod.h"
#include "common/message.h"
#include "commands.h"
#include "server.h"

namespace mgpu{
    class Receiver : public Module {
    public:
        Receiver(){
            joinable = true;
        };
        void init() override;
        void run() override;
        void join() override;
        void destroy() override;

    private:
        void do_accept();
        void do_worker(uint socket, sockaddr* cli, socklen_t* len);
        void push_command(AbMsg*, uint cli);

    private:
        uint server_socket{};
        uint max_worker{};
        std::atomic<uint> num_worker = 0;
        std::thread listener;
    };
}
#endif //FASTGPU_RECEIVER_H
