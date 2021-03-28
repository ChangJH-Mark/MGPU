//
// Created by root on 2021/3/21.
//
#include <cuda_runtime.h>
#include <cuda.h>
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
        case (MSG_CUDA_LAUNCH_KERNEL) : {
            std::async(&Conductor::do_cudalaunchkernel, this, cmd);
            break;
        }
    }
}

void Conductor::do_cudamalloc(const std::shared_ptr<Command>& cmd) {
    std::cout << __FUNCTION__ << " size: " << cmd->get_msg<CudaMallocMsg>()->size << std::endl;
    void *dev_ptr;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    cudaCheck(::cudaMalloc(&dev_ptr, cmd->get_msg<CudaMallocMsg>()->size));
    std::cout << __FUNCTION__ << " address: " << dev_ptr << std::endl;
    cmd->finish<void *>(dev_ptr);
}

void Conductor::do_cudamallochost(const std::shared_ptr<Command>& cmd) {
    std::cout << __FUNCTION__ << " size: " << cmd->get_msg<CudaMallocHostMsg>()->size << std::endl;
    auto msg = cmd->get_msg<CudaMallocHostMsg>();
    key_t shm_key = ftok("./", 0x1);
    std::cout << __FUNCTION__ << " shm_key: " << shm_key << std::endl;
    int shm_id = shmget(shm_key, msg->size, IPC_CREAT);
    if(shm_id < 0){
        perror("fail to shmget");
        exit(1);
    } else {
        std::cout << __FUNCTION__ << " shm_id: " << shm_id << std::endl;
    }
    void * host_ptr = shmat(shm_id, NULL, 0);
    std::cout << __FUNCTION__  << " share memory address: " << host_ptr << std::endl;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    cudaCheck(::cudaHostRegister(host_ptr, msg->size, cudaHostRegisterDefault));
    cmd->finish<CudaMallocHostRet>(mgpu::CudaMallocHostRet{host_ptr, shm_id});
}

void Conductor::do_cudafree(const std::shared_ptr<Command>& cmd) {
    std::cout << __FUNCTION__ << " free: " << cmd->get_msg<CudaFreeMsg>()->devPtr << std::endl;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto dev_ptr = cmd->get_msg<CudaFreeMsg>()->devPtr;
    cudaCheck(::cudaFree(dev_ptr));
    cmd->finish<bool>(true);
}

void Conductor::do_cudafreehost(const std::shared_ptr<Command> &cmd) {
    std::cout << __FUNCTION__ << " free: " << cmd->get_msg<CudaFreeHostMsg>()->ptr << std::endl;
    auto host_ptr = cmd->get_msg<CudaFreeHostMsg>()->ptr;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    cudaCheck(::cudaHostUnregister(host_ptr))
    cmd->finish<bool>(0 > shmdt(host_ptr));
}

void Conductor::do_cudamemset(const std::shared_ptr<Command>& cmd) {
    std::cout << __FUNCTION__ << " set address: " << cmd->get_msg<CudaMemsetMsg>()->devPtr << std::endl;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaMemsetMsg>();
    cudaCheck(::cudaMemset(msg->devPtr, msg->value, msg->count));
    cmd->finish<bool>(true);
}

void Conductor::do_cudamemcpy(const std::shared_ptr<Command>& cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaMemcpyMsg>();
    std::cout << __FUNCTION__ << " copy from: " << msg->src << " to: " << msg->dst << std::endl;
    cudaCheck(::cudaMemcpy(msg->dst, msg->src, msg->count, msg->kind));
    cmd->finish<bool>(true);
}

void Conductor::do_cudalaunchkernel(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaLaunchKernelMsg>();
    CUmodule cuModule;
    cudaCheck(static_cast<cudaError_t>(::cuModuleLoad(&cuModule, "/opt/custom/ptx/vecAdd.cubin")));
    CUfunction vecAdd;
    cudaCheck(static_cast<cudaError_t>(::cuModuleGetFunction(&vecAdd, cuModule, "vecAdd")));
    void * extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, msg->param,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &(msg->p_size),
            CU_LAUNCH_PARAM_END
    };
    cudaCheck(static_cast<cudaError_t>(::cuLaunchKernel(vecAdd, msg->conf.grid.x, 1, 1,
                                                        msg->conf.block.x, 1, 1,
                                                        msg->conf.share_memory,
                                                        (CUstream) msg->conf.stream,
                                                        NULL, extra)));
    cmd->finish<bool>(true);
}