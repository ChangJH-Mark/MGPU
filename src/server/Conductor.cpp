//
// Created by root on 2021/3/21.
//
#include <cuda_runtime.h>
#include "server/conductor.h"
#include "common/helper.h"
using namespace mgpu;

void Conductor::conduct(std::shared_ptr<Command> cmd) {
    switch (cmd->get_type()) {
        case(MSG_CUDA_MALLOC) :
            std::thread worker(&Conductor::do_cudamalloc, this, cmd);
            worker.detach();
            break;
    }
}

void Conductor::do_cudamalloc(std::shared_ptr<Command> cmd) {
    void * dev_ptr;
    cudaCheck(cudaSetDevice(cmd->get_device()));
    cudaCheck(cudaMalloc(&dev_ptr, cmd->get_msg<cudaMallocMSG>()->size));
    cmd->finish<void *>(dev_ptr);
}