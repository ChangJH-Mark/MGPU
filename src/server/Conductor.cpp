//
// Created by root on 2021/3/21.
//
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string>
#include "server/server.h"
#include "server/conductor.h"
#include "common/helper.h"

#define MAX_STREAMS 32
#define WORKER_GRID 60

using namespace mgpu;

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
        case (MSG_CUDA_GET_DEVICE_COUNT) : {
            std::thread worker(&Conductor::do_cudagetdevicecount, this, cmd);
            worker.detach();
            break;
        }
        case (MSG_MATRIX_MUL_GPU) : {
            std::thread worker(&Conductor::do_matrixmultgpu, this, cmd);
            worker.detach();
            break;
        }
    }// switch
    return cmd->get_status();
}

void Conductor::do_cudamalloc(const std::shared_ptr<Command> &cmd) {
    std::cout << __FUNCTION__ << " size: " << cmd->get_msg<CudaMallocMsg>()->size << std::endl;
    void *dev_ptr;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    cudaCheck(::cudaMalloc(&dev_ptr, cmd->get_msg<CudaMallocMsg>()->size));
    std::cout << __FUNCTION__ << " address: " << dev_ptr << std::endl;
    cmd->finish<void *>(dev_ptr);
}

void Conductor::do_cudamallochost(const std::shared_ptr<Command> &cmd) {
    std::cout << __FUNCTION__ << " size: " << cmd->get_msg<CudaMallocHostMsg>()->size << std::endl;
    auto msg = cmd->get_msg<CudaMallocHostMsg>();
    int shm_id;
    if ((shm_id  = shmget(IPC_PRIVATE, msg->size, IPC_CREAT))< 0) {
        perror("fail to shmget");
        exit(1);
    } else {
        std::cout << __FUNCTION__ << " shm_id: " << shm_id << std::endl;
    }
    void *host_ptr = shmat(shm_id, NULL, 0);
    if(!shms_id.count(host_ptr))
    {
        shms_id[host_ptr] = shm_id;
        std::cout << __FUNCTION__ << " share memory address: " << host_ptr << std::endl;
    } else {
        std::cout << __FUNCTION__  << " share memory already exist shm_id : " << shms_id[host_ptr] << std::endl;
        perror("fail to shmget");
        exit(1);
    }
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    cudaCheck(::cudaHostRegister(host_ptr, msg->size, cudaHostRegisterDefault));
    cmd->finish<CudaMallocHostRet>(mgpu::CudaMallocHostRet{host_ptr, shm_id});
}

void Conductor::do_cudafree(const std::shared_ptr<Command> &cmd) {
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
    if(0 > shmdt(host_ptr))
    {
        perror("server fail to release share memory");
        exit(1);
    }
    if(0 > shmctl(shms_id[host_ptr], IPC_RMID, NULL))
    {
        perror("server fail to delete share memory");
        exit(1);
    }
    shms_id.erase(host_ptr);
    cmd->finish<bool>(true);
}

void Conductor::do_cudamemset(const std::shared_ptr<Command> &cmd) {
    std::cout << __FUNCTION__ << " set address: " << cmd->get_msg<CudaMemsetMsg>()->devPtr << std::endl;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaMemsetMsg>();
    cudaCheck(::cudaMemset(msg->devPtr, msg->value, msg->count));
    cmd->finish<bool>(true);
}

void Conductor::do_cudamemcpy(const std::shared_ptr<Command> &cmd) {
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
    CUfunction func;
    cudaCheck(
            static_cast<cudaError_t>(::cuModuleGetFunction(&func, cuModule, (string(msg->kernel) + "Proxy").c_str())));
    msg->p_size = fillParameters(msg->param, msg->p_size, 0, 6, msg->conf.grid,
                                 (msg->conf.grid.x * msg->conf.grid.y * msg->conf.grid.z));
    void *extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, msg->param,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &(msg->p_size),
            CU_LAUNCH_PARAM_END
    };
    std::cout << __FUNCTION__ << " launch from ptx" << msg->ptx << " kernel: " << std::string(msg->kernel) + "Proxy"
              << " at : device : "
              << cmd->get_device() << " cudaStream_t : " << cmd->get_stream() << " gridDim " + to_string(msg->conf.grid.x) + " "
              << to_string(msg->conf.grid.y) + " " << to_string(msg->conf.grid.z) + " "
              << std::endl;
    cudaCheck(static_cast<cudaError_t>(::cuLaunchKernel(func, WORKER_GRID, 1, 1,
                                                        msg->conf.block.x, msg->conf.block.y, msg->conf.block.z,
                                                        msg->conf.share_memory,
                                                        cmd->get_stream(),
                                                        NULL, extra)));
    cmd->finish<bool>(true);
}

void Conductor::do_cudastreamcreate(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaStreamCreateMsg>();
    stream_t *ret = new stream_t[msg->num];
    auto server = get_server();
    std::lock_guard<std::mutex> streamCreateGuard(server->map_mtx);

    std::map<stream_t, bool> unused;
    for(int i =0;i<msg->num;i++)
    {
        bool found = false;
        for(auto s : server->get_device()->getStream(cmd->get_device()))
        {
            auto k = ListKey{msg->key, s};
            if(server->task_map.find(k) == server->task_map.end())
            {
                found = true;
                ret[i] = s;
                break;
            } else {
                unused[s] = true;
            }
        }
        if(!found) {
            stream_t s;
            if(unused.size())
            {
                s = unused.begin()->first;
                unused.erase(unused.begin());
            } else {
                s = server->get_device()->getStream(cmd->get_device())[random() % MAX_STREAMS];
            }
            ret[i] = s;
        }
    }
    cmd->finish<stream_t>(ret, msg->num);
}

void Conductor::do_cudastreamsynchronize(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    std::cout << __FUNCTION__ << " synchronize stream: at device: " << cmd->get_device() << " stream: "
              << cmd->get_stream() << std::endl;
    cudaCheck(::cudaStreamSynchronize(cmd->get_stream()));
    cmd->finish<bool>(true);
}

void Conductor::do_cudagetdevicecount(const std::shared_ptr<Command> &cmd) {
    std::cout << __FUNCTION__ << " cuda get device count " << std::endl;
    int count;
    cudaCheck(::cudaGetDeviceCount(&count));
    cmd->finish<int>(count);
}

void Conductor::do_matrixmultgpu(const std::shared_ptr<Command> &cmd) {
    auto msg = cmd->get_msg<MatrixMulMsg>();
    Matrix A = msg->A;
    Matrix B = msg->B;
    auto conf = cmd->get_msg<MatrixMulMsg>()->conf;
    int shm_id = shmget(IPC_PRIVATE, sizeof(float) * A.height * B.width, IPC_CREAT);
    if(shm_id < 0)
    {
        perror("fail to shmget");
        exit(1);
    } else {
        std::cout << __FUNCTION__ << " shm_id: " << shm_id << std::endl;
    }
    void * res = shmat(shm_id, NULL, 0);
    if (!shms_id.count(res)) {
        shms_id[res] = shm_id;
    } else {
        perror("fail to shmat, address already used");
        exit(1);
    }
    cudaCheck(::cudaHostRegister(res, sizeof(float) * A.height * B.width, cudaHostRegisterDefault));

    auto device = get_server()->get_device();
    std::vector<cudaEvent_t> starts(device->counts());
    std::vector<cudaEvent_t> ends(device->counts());

    for(int i = 0; i < device->counts(); i++)
    {
        std::cout << __FUNCTION__ << " device: " << i << " do matrix multi " << std::endl;
        cudaCheck(::cudaSetDevice(i));
        cudaCheck(::cudaEventCreateWithFlags(&(starts[i]), cudaEventBlockingSync));
        cudaCheck(::cudaEventCreateWithFlags(&(ends[i]), cudaEventBlockingSync));
        void * dev_A, *dev_B, *dev_C;
        cudaCheck(::cudaMalloc(&dev_A, sizeof(float) * A.height * A.width));
        cudaCheck(::cudaMalloc(&dev_B, sizeof(float) * B.height * B.width));
        cudaCheck(::cudaMalloc(&dev_C, sizeof(float) * A.height * B.width));
        cudaEventRecord(starts[i], 0);
        cudaMemcpyAsync(dev_A, A.data, sizeof(float) * A.height * A.width, cudaMemcpyHostToDevice, 0);
        cudaMemcpyAsync(dev_B, B.data, sizeof(float) * B.height * B.width, cudaMemcpyHostToDevice, 0);
        CUmodule module;
        cudaCheck(static_cast<cudaError_t>(::cuModuleLoad(&module, "/opt/custom/ptx/matrixMul.ptx")));
        CUfunction func;
        cudaCheck(static_cast<cudaError_t>(::cuModuleGetFunction(&func, module, "matrixMulProxy")));
        char params[1024];
        auto p_size = fillParameters(params, 0, static_cast<void*>(dev_C), static_cast<void*>(dev_A), static_cast<void*>(dev_B), A.width, B.width, 0,2, conf.grid, (conf.grid.x * conf.grid.y));
        void *extra[] = {
                CU_LAUNCH_PARAM_BUFFER_POINTER, params,
                CU_LAUNCH_PARAM_BUFFER_SIZE, &p_size,
                CU_LAUNCH_PARAM_END
        };
        cudaCheck(static_cast<cudaError_t>(::cuLaunchKernel(func, WORKER_GRID, 1, 1, conf.block.x, conf.block.y, conf.block.z, conf.share_memory, 0, nullptr, extra)));
        int area = A.height * B.width / device->counts();
        cudaCheck(::cudaMemcpyAsync(static_cast<float*>(res) + area * i, static_cast<float*>(dev_C) + area * i, sizeof(float) * area, cudaMemcpyDeviceToHost, 0));
        cudaCheck(::cudaEventRecord(ends[i], 0));
    }
    for(int i = 0; i<device->counts();i++){
        cudaCheck(::cudaEventSynchronize(ends[i]));
    }
    std::cout << __FUNCTION__ << " finish " << std::endl;
    cmd->finish<CudaMallocHostRet>(mgpu::CudaMallocHostRet{res, shm_id});
}