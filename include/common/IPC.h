//
// Created by root on 2021/3/16.
//

#ifndef FASTGPU_IPC_H
#define FASTGPU_IPC_H
#include <sys/socket.h>
#include <sys/un.h>
#include <string>
#include <map>
#include "common/message.h"

namespace mgpu {
    extern pid_t pid;
    extern const char *server_path;

    class IPCClient {
    public:
        static IPCClient* get_client();
        IPCClient& operator=(const IPCClient &) = delete;
        void * send(CudaMallocMsg*);
        void * send(CudaMallocHostMsg*);
        bool send(CudaFreeMsg*);
        bool send(CudaFreeHostMsg*);
        bool send(CudaMemsetMsg*);
        bool send(CudaMemcpyMsg*);
        bool send(CudaLaunchKernelMsg*);
        bool send(CudaStreamCreateMsg*, int * streams);
        bool send(CudaStreamSyncMsg *msg);
    public:
        ~IPCClient();
    private:
        IPCClient();
        uint connect();
        void socket_send(uint cli, void* msg, size_t size, uint flag, const char *err_msg);
        void socket_recv(uint cli, void* dst, size_t size, uint flag, const char *err_msg);
        void socket_clear(uint socket);
        static IPCClient* single_instance;
    };

    void destroy_client();
}
#endif //FASTGPU_IPC_H