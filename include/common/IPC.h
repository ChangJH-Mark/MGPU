//
// Created by root on 2021/3/16.
//

#ifndef FASTGPU_IPC_H
#define FASTGPU_IPC_H
#include <sys/socket.h>
#include <sys/un.h>
#include <string>
#include "common/message.h"

namespace mgpu {
    extern pid_t pid;
    extern const char *server_path;

    class IPCClient {
    public:
        static IPCClient* get_client();
        IPCClient& operator=(const IPCClient &) = delete;
        void connect();
        void * send(cudaMallocMSG*);

    private:
        IPCClient(const std::string &string);
        ~IPCClient();
        uint socket;
        std::string address;
        static IPCClient* single_instance;
    };
}
#endif //FASTGPU_IPC_H
