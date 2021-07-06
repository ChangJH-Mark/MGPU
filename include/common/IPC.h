//
// Created by root on 2021/3/16.
//

#ifndef FASTGPU_IPC_H
#define FASTGPU_IPC_H
#include <sys/socket.h>
#include <sys/un.h>
#include <string>
#include <map>
#include <memory>
#include <unistd.h>
#include "common/message.h"

namespace mgpu {
    class IPCClient;
    extern pid_t pid;
    static char *server_path = "/opt/custom/server.sock";
    static std::shared_ptr<IPCClient> single_instance;

    class IPCClient {
    public:
        static std::shared_ptr<IPCClient> get_client();
        IPCClient& operator=(const IPCClient &) = delete;
        int send(CudaGetDeviceCountMsg*);
        void * send(CudaMallocMsg*);
        void * send(CudaMallocHostMsg*);
        bool send(CudaFreeMsg*);
        bool send(CudaFreeHostMsg*);
        bool send(CudaMemsetMsg*);
        bool send(CudaMemcpyMsg*);
        bool send(CudaLaunchKernelMsg*);
        bool send(CudaStreamCreateMsg*, stream_t * stream);
        bool send(CudaStreamSyncMsg *msg);
        bool send(CudaEventCreateMsg*, event_t * event);
        bool send(CudaEventDestroyMsg*);
        bool send(CudaEventRecordMsg*);
        bool send(CudaEventSyncMsg*);
        bool send(CudaEventElapsedTimeMsg*, float *ms);
        std::future<void *> send(MatrixMulMsg *msg);
        IPCClient() : conn(-1) {} // only make_shared use it
    public:
        ~IPCClient(){ if(conn != -1) close(conn);}
    private:
        int conn; // connection socket
        int connect();
        void socket_send(uint cli, void* msg, size_t size, uint flag, const char *err_msg);
        void socket_recv(uint cli, void* dst, size_t size, uint flag, const char *err_msg);
    };
}
#endif //FASTGPU_IPC_H
