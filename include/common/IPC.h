//
// Created by root on 2021/3/16.
//

#ifndef FASTGPU_IPC_H
#define FASTGPU_IPC_H

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/time.h>
#include <linux/futex.h>
#include <string>
#include <map>
#include <memory>
#include <unistd.h>
#include <cassert>
#include "common/message.h"

namespace mgpu {
    class IPCClient;

    extern pid_t pid;
    static const char *server_path = "/opt/custom/server.sock";
    static std::shared_ptr<IPCClient> single_instance;

    class IPCClient {
    public:
        static std::shared_ptr<IPCClient> get_client();

        IPCClient &operator=(const IPCClient &) = delete;

        int send(CudaGetDeviceCountMsg *);

        void *send(CudaMallocMsg *);

        void *send(CudaMallocHostMsg *);

        bool send(CudaFreeMsg *);

        bool send(CudaFreeHostMsg *);

        bool send(CudaMemsetMsg *);

        bool send(CudaMemcpyMsg *);

        bool send(CudaLaunchKernelMsg *);

        bool send(CudaStreamCreateMsg *, stream_t *stream);

        bool send(CudaStreamSyncMsg *msg);

        bool send(CudaEventCreateMsg *, event_t *event);

        bool send(CudaEventDestroyMsg *);

        bool send(CudaEventRecordMsg *);

        bool send(CudaEventSyncMsg *);

        bool send(CudaEventElapsedTimeMsg *, float *ms);

        std::future<void *> send(MatrixMulMsg *msg);

        std::future<MulTaskRet> send(MulTaskMsg *msg);

        // only used for make_shared
        IPCClient() : conn(-1) {
            pid = getpid();
            if(0 != connect()) {
                perror("error to connect to server");
                exit(1);
            }
        }

    public:
        ~IPCClient() {
            if (conn != -1) close(conn);
            deinit_shm();
        }

    private:
        int conn; // connection socket
        pid_t pid; // client pid
        Futex c_fut; // client futex
        Futex s_fut; // server futex

        int connect();

        int init_shm();

        int deinit_shm();

        void send_msg(void *msg, size_t size, const char *err_msg);

        void recv_msg(void *dst, size_t size, const char *err_msg);
    };
}
#endif //FASTGPU_IPC_H
