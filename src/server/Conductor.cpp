//
// Created by root on 2021/3/21.
//
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "server/server.h"
#include "server/conductor.h"
#include "common/helper.h"
#define MAX_STREAMS 32
#define key2stream(key) (((key) >> 16 + (key) & 0xffff) % MAX_STREAMS)

using namespace mgpu;

cudaStream_t Conductor::get_stream(uint device, uint key) {
    return *(get_server()->get_device()->getStream(device, key2stream(key)));
}

std::shared_ptr<bool> Conductor::conduct(std::shared_ptr<Command> cmd) {
    switch (cmd->get_type()) {
        case (MSG_CUDA_MALLOC) : {
            std::thread worker(&Conductor::do_cudamalloc, this, cmd);
            worker.detach();
            break;
        }
        case (MSG_CUDA_MALLOC_HOST) : {
            std::thread worker(&Conductor::do_cudamallochost, this, cmd);
            worker.detach();
            break;
        }
        case (MSG_CUDA_FREE) : {
            std::thread worker(&Conductor::do_cudafree, this, cmd);
            worker.detach();
            break;
        }
        case (MSG_CUDA_FREE_HOST) : {
            std::thread worker(&Conductor::do_cudafreehost, this, cmd);
            worker.detach();
            break;
        }
        case (MSG_CUDA_MEMSET) : {
            std::thread worker(&Conductor::do_cudamemset, this, cmd);
            worker.detach();
            break;
        }
        case (MSG_CUDA_MEMCPY) : {
            std::thread worker(&Conductor::do_cudamemcpy, this, cmd);
            worker.detach();
            break;
        }
        case (MSG_CUDA_LAUNCH_KERNEL) : {
            std::thread worker(&Conductor::do_cudalaunchkernel, this, cmd);
            worker.detach();
            break;
        }
        case (MSG_CUDA_STREAM_CREATE) : {
            std::thread streamCreate(&Conductor::do_cudastreamcreate, this, cmd);
            streamCreate.detach();
            break;
        }
        case (MSG_CUDA_STREAM_SYNCHRONIZE) : {
            std::thread worker(&Conductor::do_cudastreamsynchronize, this, cmd);
            worker.detach();
            break;
        }
    }// switch
    return cmd->get_status();
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
    key_t shm_key = ftok("./", msg->key >> 16);
    std::cout << __FUNCTION__ << " shm_key: " << hex << shm_key << dec << std::endl;
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
    cudaCheck(static_cast<cudaError_t>(::cuModuleLoad(&cuModule, msg->ptx)));
    CUfunction vecAdd;
    cudaCheck(static_cast<cudaError_t>(::cuModuleGetFunction(&vecAdd, cuModule, msg->kernel)));
    void * extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, msg->param,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &(msg->p_size),
            CU_LAUNCH_PARAM_END
    };
    std::cout << __FUNCTION__ << " launch from ptx" << msg->ptx << " kernel: " << msg->kernel << " at : device : "
              << cmd->get_device() << " stream: " << key2stream(msg->key) << " cudaStream_t : "
              << get_stream(cmd->get_device(), msg->key) << std::endl;
    cudaCheck(static_cast<cudaError_t>(::cuLaunchKernel(vecAdd, msg->conf.grid.x, 1, 1,
                                                        msg->conf.block.x, 1, 1,
                                                        msg->conf.share_memory,
                                                        get_stream(cmd->get_device(), msg->key),
                                                        NULL, extra)));
    cmd->finish<bool>(true);
}

void Conductor::do_cudastreamcreate(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaStreamCreateMsg>();
    int * ret = new int[msg->num];
    auto server = get_server();
    std::lock_guard<std::mutex> streamCreateGuard(server->map_mtx);
    std::map<int, bool> used;
    for(int i = 0; i < msg->num; i++) {
        bool found = false;
        for(int stream = 1; stream <= MAX_STREAMS; stream++){
            if(server->task_map.find(msg->key & 0x0000 + stream) == server->task_map.end() && !used[stream]){
                found = true;
                ret[i] = stream;
                used[stream] = true;
                break;
            }
        }
        if(!found){
            cmd->finish<int>(ret, i + 1);
        }
    }
    cmd->finish<int>(ret, msg->num);
}

void Conductor::do_cudastreamsynchronize(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaStreamSyncMsg>();
    std::cout << __FUNCTION__ << " synchronize stream: at device: " << cmd->get_device() << " stream: "
              << key2stream(msg->key) << " cudaStream_t: " << get_stream(cmd->get_device(), msg->key) << std::endl;
    cudaCheck(::cudaStreamSynchronize(get_stream(cmd->get_device(), msg->key)));
    cmd->finish<bool>(true);
}