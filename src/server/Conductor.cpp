//
// Created by root on 2021/3/21.
//
#include <cuda_runtime.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "server/conductor.h"
#include "common/helper.h"

using namespace mgpu;

void Conductor::conduct(std::shared_ptr<Command> cmd) {
    switch (cmd->get_type()) {
        case (MSG_CUDA_MALLOC) : {
            std::async(&Conductor::do_cudamalloc, this, cmd);
            break;
        }
        case (MSG_CUDA_MALLOC_HOST) : {
            std::async(&Conductor::do_cudamallochost, this, cmd);
            break;
        }
        case (MSG_CUDA_FREE) : {
            std::async(&Conductor::do_cudafree, this, cmd);
            break;
        }
        case (MSG_CUDA_FREE_HOST) : {
            std::async(&Conductor::do_cudafreehost, this, cmd);
            break;
        }
        case (MSG_CUDA_MEMSET) : {
            std::async(&Conductor::do_cudamemset, this, cmd);
            break;
        }
        case (MSG_CUDA_MEMCPY) : {
            std::async(&Conductor::do_cudamemcpy, this, cmd);
            break;
        }
    }
}

void Conductor::do_cudamalloc(const std::shared_ptr<Command>& cmd) {
    void *dev_ptr;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    cudaCheck(::cudaMalloc(&dev_ptr, cmd->get_msg<CudaMallocMsg>()->size));
    cmd->finish<void *>(dev_ptr);
}

void Conductor::do_cudamallochost(const std::shared_ptr<Command>& cmd) {
    auto msg = cmd->get_msg<CudaMallocHostMsg>();
    key_t shm_key = ftok("./", 0x1);
    int shm_id = shmget(shm_key, msg->size, IPC_CREAT);
    if(shm_id < 0){
        perror("fail to shmget");
        exit(1);
    } else {
        std::cout << "create share memory id: " << shm_id << std::endl;
    }
    void * host_ptr = shmat(shm_id, NULL, 0);
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    cudaCheck(::cudaHostRegister(host_ptr, msg->size, cudaHostRegisterDefault));
    cmd->finish<CudaMallocHostRet>(mgpu::CudaMallocHostRet{host_ptr, shm_id});
}

void Conductor::do_cudafree(const std::shared_ptr<Command>& cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    cudaCheck(::cudaFree(cmd->get_msg<CudaFreeMsg>()->devPtr));
    cmd->finish<bool>(true);
}

void Conductor::do_cudafreehost(const std::shared_ptr<Command> &cmd) {
    auto host_ptr = cmd->get_msg<CudaFreeHostMsg>()->ptr;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    cudaCheck(::cudaHostUnregister(host_ptr))
    cmd->finish<bool>(0 > shmdt(host_ptr));
}

void Conductor::do_cudamemset(const std::shared_ptr<Command>& cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaMemsetMsg>();
    cudaCheck(::cudaMemset(msg->devPtr, msg->value, msg->count));
    cmd->finish<bool>(true);
}

void Conductor::do_cudamemcpy(const std::shared_ptr<Command>& cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaMemcpyMsg>();
    cudaCheck(::cudaMemcpy(msg->dst, msg->src, msg->count, msg->kind));
    cmd->finish<bool>(true);
}