//
// Created by root on 2021/3/16.
//
#include <cstring>
#include <unistd.h>
#include <cstdio>
#include <future>
#include <sys/shm.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <cassert>
#include <string>
#include "common/IPC.h"
#include "common/message.h"

using namespace mgpu;

pid_t mgpu::pid = getpid();

std::shared_ptr<IPCClient> IPCClient::get_client() {
    if (!single_instance) {
        single_instance = std::make_shared<IPCClient>();
    }
    return single_instance;
}

int IPCClient::init_shm() {
    using std::string;
    string names[2] = {"mgpu.0."+ std::to_string(pid), "mgpu.1."+std::to_string(pid)};
    string root = "/dev/shm/";
    void *shms[2];
    int cnt = 0;

    for(auto & n : names) {
        if(0 == access((root + n).c_str(), F_OK)) {
            assert(shm_unlink(n.c_str()));
        }
        int fd = shm_open(n.c_str(), O_CREAT | O_CLOEXEC | O_RDWR, 0644);
        assert(fd > 0);
        ftruncate(fd, PAGE_SIZE);

        auto shm_ptr = mmap(NULL, 4096,PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        assert(shm_ptr != BAD_UADDR);
        shms[cnt++] = shm_ptr;
        close(fd);
    }
    c_fut = Futex(shms[0]);
    s_fut = Futex(shms[1]);
    c_fut.setState(NOT_READY);
    s_fut.setState(NOT_READY);
    return 0;
}

int IPCClient::deinit_shm() {
    std::string names[2] = {"mgpu.0."+ std::to_string(pid), "mgpu.1."+std::to_string(pid)};
    int cnt = 0;
    for(auto & n : names) {
        if (0 != access(n.c_str(), F_OK)) {
            munmap(cnt ? c_fut.shm_ptr : s_fut.shm_ptr, PAGE_SIZE);
            shm_unlink(n.c_str());
            cnt++;
        }
    }
    return 0;
}

int IPCClient::connect() {
    init_shm();

    // connect mgpu server
    conn = socket(PF_LOCAL, SOCK_STREAM, 0);
    struct sockaddr_un server_addr{PF_LOCAL};
    strcpy(server_addr.sun_path, server_path);
    if (0 > ::connect(conn, (struct sockaddr *) (&server_addr), SUN_LEN(&server_addr))) {
        perror("fail to connect to server:");
        return errno;
    }
    // tell server two shm_ptrs
    write(conn, &c_fut.shm_ptr, sizeof(void *));
    write(conn, &s_fut.shm_ptr, sizeof(void *));
    return 0;
}

void IPCClient::send_msg(void *msg, size_t size, const char *err_msg) {
    if(c_fut.state() == READY) {
        printf("error to send msg, c_fut is READY before set data\n");
        exit(1);
    }
    c_fut.setData(msg, size);
    c_fut.setState(READY);
    // if server did not take data
    if(c_fut.state() == READY) {
        int res = syscall(__NR_futex, c_fut.shm_ptr, FUTEX_WAKE, 1, NULL);
        if (res == -1) {
            printf("error to wake server, syscall return %d, errno is %d\n", res, errno);
            exit(1);
        }
    }
}

void IPCClient::recv_msg(void *dst, size_t size, const char *err_msg) {
    if(!s_fut.ready()) {
        int res = syscall(__NR_futex, s_fut.shm_ptr, FUTEX_WAIT, NOT_READY, NULL);
        if(res == -1) {
            if(errno != EAGAIN) {
                printf("error to recv from server, syscall return %d, errno is %d before read data\n", res, errno);
                exit(1);
            }
        }
    }
    if(size != s_fut.size()) {
        printf("want %d bytes, got %d bytes from mgpu\n", size, s_fut.size());
        exit(1);
    }
    s_fut.copyTo(dst, size);
    s_fut.setState(NOT_READY);
}

int IPCClient::send(CudaGetDeviceCountMsg *msg) {
    send_msg(msg, sizeof(CudaGetDeviceCountMsg), "fail to send cudaGetDeviceCount message");
    int ret;
    recv_msg(&ret, sizeof(ret), "error to receive cudaGetDeviceCount return");
    return ret;
}

void *IPCClient::send(CudaMallocMsg *msg) {
    send_msg(msg, sizeof(CudaMallocMsg), "fail to send cudaMalloc message");
    void *ret;
    recv_msg(&ret, sizeof(ret), "error to receive cudaMalloc return");
    return ret;
}

void *IPCClient::send(CudaMallocHostMsg *msg) {
    send_msg(msg, sizeof(CudaMallocHostMsg), "fail to send cudaMallocHost message");
    mgpu::CudaMallocHostRet ret;
    recv_msg(&ret, sizeof(ret), "error to receive cudaMallocHost return");
    auto addr = shmat(ret.shmid, ret.ptr, 0);
    if (ret.ptr != addr) {
        printf("err %s, return addr %lx, attached %lx, shm_id %d\n", strerror(errno), ret.ptr, addr, ret.shmid);
        perror("share memory with different address");
        exit(1);
    }
    return ret.ptr;
}

bool IPCClient::send(CudaFreeMsg *msg) {
    send_msg(msg, sizeof(CudaFreeMsg), "fail to send cudaFree message");
    bool ret;
    recv_msg(&ret, sizeof(ret), "error to receive cudaFree return");
    return ret;
}

bool IPCClient::send(CudaFreeHostMsg *msg) {
    if (0 > shmdt(msg->ptr)) {
        perror("fail to release share memory");
        exit(1);
    }

    send_msg(msg, sizeof(CudaFreeHostMsg), "fail to send cudaFreeHost message");
    bool ret;
    recv_msg(&ret, sizeof(ret), "error to receive cudaFreeHost return");
    return ret;
}

bool IPCClient::send(CudaMemsetMsg *msg) {
    send_msg(msg, sizeof(CudaMemsetMsg), "fail to send cudaMemset message");
    bool ret;
    recv_msg(&ret, sizeof(ret), "error to receive cudaMemset return");
    return ret;
}

bool IPCClient::send(CudaMemcpyMsg *msg) {
    send_msg(msg, sizeof(CudaMemcpyMsg), "fail to send cudaMemcpy message");
    bool ret;
    recv_msg(&ret, sizeof(ret), "error to receive cudaMemcpy return");
    return ret;
}

bool IPCClient::send(CudaLaunchKernelMsg *msg) {
    send_msg(msg, sizeof(CudaLaunchKernelMsg), "fail to send cudaLaunchKernel message");
    bool ret;
    recv_msg(&ret, sizeof(ret), "error to receive cudaLaunchKernel return");
    return ret;
}

bool IPCClient::send(CudaStreamCreateMsg *msg, stream_t *stream) {
    send_msg(msg, sizeof(CudaStreamCreateMsg), "fail to send cudaStreamCreate message");
    recv_msg(stream, sizeof(stream_t), "error to receive cudaStreamCreate return");
    return true;
}

bool IPCClient::send(CudaStreamSyncMsg *msg) {
    send_msg(msg, sizeof(CudaStreamSyncMsg), "fail to send cudaStreamSynchronize message");
    bool ret;
    recv_msg(&ret, sizeof(bool), "error to receive cudaStreamSynchronize return");
    return ret;
}

bool IPCClient::send(CudaEventCreateMsg *msg, event_t *event) {
    send_msg(msg, sizeof(CudaEventSyncMsg), "fail to send cudaEventCreate message");
    recv_msg(event, sizeof(event_t), "error to receive cudaEventCreate return");
    return true;
}

bool IPCClient::send(CudaEventDestroyMsg *msg) {
    send_msg(msg, sizeof(CudaEventDestroyMsg), "fail to send cudaEventDestroy message");
    bool ret;
    recv_msg(&ret, sizeof(bool), "error to receive cudaEventDestroy return");
    return ret;
}

bool IPCClient::send(CudaEventRecordMsg *msg) {
    send_msg(msg, sizeof(CudaEventRecordMsg), "fail to send cudaEventRecord message");
    bool ret;
    recv_msg(&ret, sizeof(bool), "error to receive cudaEventRecord return");
    return ret;
}

bool IPCClient::send(CudaEventSyncMsg *msg) {
    send_msg(msg, sizeof(CudaEventSyncMsg), "fail to send cudaEventSync message");
    bool ret;
    recv_msg(&ret, sizeof(bool), "error to receive cudaEventSync return");
    return ret;
}

bool IPCClient::send(CudaEventElapsedTimeMsg *msg, float *ms) {
    send_msg(msg, sizeof(CudaEventElapsedTimeMsg), "fail to send cudaEventElapsedTime message");
    recv_msg(ms, sizeof(float), "error to receive cudaEventElapsedTime return");
    return true;
}

std::future<void *> IPCClient::send(MatrixMulMsg *msg) {
//     send_msg( msg, sizeof(MatrixMulMsg), "fail to send MatrixMulGPU message");
//    auto func = [cli, ipc = single_instance]() -> void * {
//        CudaMallocHostRet ret;
//        ipc-> recv_msg( &ret, sizeof(ret), 0, "error to receive MatrixMulGPU return");
//        if (ret.ptr != shmat(ret.shmid, ret.ptr, 0)) {
//            perror("share memory with different address");
//            return nullptr;
//        }
//        return ret.ptr;
//    };
//    return std::async(func);
}

MulTaskRet IPCClient::send(MulTaskMsg *msg) {
    send_msg(msg, sizeof(MulTaskMsg), "fail to send MulTaskMulGPU message");
    MulTaskRet ret;
    recv_msg(&ret, sizeof(ret), "error to receive MulTaskMulGPU return");
    return ret;
}