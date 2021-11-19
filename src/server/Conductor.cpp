//
// Created by root on 2021/3/21.
//
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string>
#include <functional>
#include "server/conductor.h"
#include "common/helper.h"
#include "common/Log.h"
#include "server/server.h"
#include "server/device.h"
#include "server/kernel.h"
#include "server/scheduler.h"

using namespace mgpu;

void Conductor::init() {
    func_table[MSG_CUDA_MALLOC] = do_cudamalloc;
    func_table[MSG_MOCK_MALLOC] = &Conductor::do_mockmalloc;
    func_table[MSG_CUDA_MALLOC_HOST] = &Conductor::do_cudamallochost;
    func_table[MSG_CUDA_FREE] = &Conductor::do_cudafree;
    func_table[MSG_CUDA_FREE_HOST] = &Conductor::do_cudafreehost;
    func_table[MSG_CUDA_MEMSET] = &Conductor::do_cudamemset;
    func_table[MSG_CUDA_MEMCPY] = &Conductor::do_cudamemcpy;
    func_table[MSG_CUDA_LAUNCH_KERNEL] = &Conductor::do_cudalaunchkernel;
    func_table[MSG_MOCK_LAUNCH_KERNEL] = &Conductor::do_mocklaunchkernel;
    func_table[MSG_CUDA_STREAM_CREATE] = &Conductor::do_cudastreamcreate;
    func_table[MSG_CUDA_STREAM_SYNCHRONIZE] = &Conductor::do_cudastreamsynchronize;
    func_table[MSG_CUDA_GET_DEVICE_COUNT] = &Conductor::do_cudagetdevicecount;
    func_table[MSG_CUDA_EVENT_CREATE] = &Conductor::do_cudaeventcreate;
    func_table[MSG_CUDA_EVENT_DESTROY] = &Conductor::do_cudaeventdestroy;
    func_table[MSG_CUDA_EVENT_RECORD] = &Conductor::do_cudaeventrecord;
    func_table[MSG_CUDA_EVENT_SYNCHRONIZE] = &Conductor::do_cudaeventsynchronize;
    func_table[MSG_CUDA_EVENT_ELAPSED_TIME] = &Conductor::do_cudaeventelapsedtime;
    func_table[MSG_MATRIX_MUL_GPU] = &Conductor::do_matrixmultgpu;
    func_table[MSG_MUL_TASK] = &Conductor::do_multask;
}

void Conductor::conduct(const std::shared_ptr<Command> &cmd) {
    auto task = std::bind(func_table[cmd->get_type()], cmd);
    task();
}

void Conductor::do_cudamalloc(const std::shared_ptr<Command> &cmd) {
    // dout(DEBUG) << " cmd_id: " << cmd->get_id() << " size: " << cmd->get_msg<CudaMallocMsg>()->size <<dendl;
    void *dev_ptr;
    int size = cmd->get_msg<CudaMallocMsg>()->size;
    if (false) {
        // dout(DEBUG) << " cmd_id: " << cmd->get_id() << " with pool address: " << dev_ptr <<dendl;
        MEMPOOL->gpuMemoryAlloc(cmd->get_device(), &dev_ptr, cmd->get_msg<CudaMallocMsg>()->size, cmd->get_stream());
    } else {
        cudaCheck(::cudaSetDevice(cmd->get_device()));
        cudaCheck(::cudaMalloc(&dev_ptr, cmd->get_msg<CudaMallocMsg>()->size));
        //dout(DEBUG) << " cmd_id: " << cmd->get_id() << " address: " << dev_ptr <<dendl;
    }
    cmd->finish<void *>(dev_ptr);
}

void Conductor::do_mockmalloc(const std::shared_ptr<Command> &cmd) {
    //dout(DEBUG) << " cmd_id: " << cmd->get_id() << " size: " << cmd->get_msg<MockMallocMsg>()->size <<dendl;
    void *dev_ptr = nullptr;
    cmd->finish<void *>(dev_ptr);
}

void Conductor::do_cudamallochost(const std::shared_ptr<Command> &cmd) {
    auto msg = cmd->get_msg<CudaMallocHostMsg>();
    int shm_id = -1;
    void *host_ptr = nullptr;
    if (false) {
        host_ptr = MEMPOOL->cpuMemoryAlloc(msg->size);
        if (host_ptr == nullptr) {
            perror("fail to allocate share memory");
            exit(EXIT_FAILURE);
        }
        shm_id = *(int *) host_ptr;
        //dout(DEBUG) << "share memory id " << shm_id << " address " << host_ptr << " size : " << msg->size <<dendl;
    } else {
        shm_id = shmget(IPC_PRIVATE, msg->size, IPC_CREAT | IPC_EXCL);
        if (shm_id < 0) {
            perror("fuck to shmget");
            exit(errno);
        } else {
            // dout(DEBUG) << " cmd_id: " << cmd->get_id() << " shm_id: " << shm_id  << dendl;
            if (0 > shmctl(shm_id, SHM_LOCK, nullptr)) {
                perror("fail to lock shmget");
                exit(1);
            }
        }
        host_ptr = shmat(shm_id, nullptr, 0);
        auto &shms_id = cmd->get_worker()->shms_id;
        if (!shms_id.count(host_ptr)) {
            shms_id[host_ptr] = shm_id;
            //dout(DEBUG) << " cmd_id: " << cmd->get_id() << " share memory address: " << host_ptr  << dendl;
        } else {
            //dout(DEBUG) << " cmd_id: " << cmd->get_id() << " share memory already exist shm_id: " << shms_id[host_ptr] <<dendl;
            perror("fail to shmget");
            exit(1);
        }
        cudaCheck(cudaHostRegister(host_ptr, msg->size, cudaHostRegisterDefault));
    }
    cmd->finish<CudaMallocHostRet>(mgpu::CudaMallocHostRet{host_ptr, shm_id});
    // dout(DEBUG) << " cmd_id: " << cmd->get_id() << " finished "  << dendl;
}

void Conductor::do_cudafree(const std::shared_ptr<Command> &cmd) {
    // dout(DEBUG) << " cmd_id: " << cmd->get_id() << " free: " << cmd->get_msg<CudaFreeMsg>()->devPtr  << dendl;
    if (false) {
        MEMPOOL->gpuMemoryDeAlloc(cmd->get_device(), cmd->get_msg<CudaFreeMsg>()->devPtr, cmd->get_stream());
        // dout(DEBUG) << " cmd_id: " << cmd->get_id() << " free with pool: " << cmd->get_msg<CudaFreeMsg>()->devPtr << dendl;
    } else {
        cudaCheck(::cudaSetDevice(cmd->get_device()));
        auto dev_ptr = cmd->get_msg<CudaFreeMsg>()->devPtr;
        cudaCheck(::cudaFree(dev_ptr));
    }
    cmd->finish<bool>(true);
}

void Conductor::do_cudafreehost(const std::shared_ptr<Command> &cmd) {
    //dout(DEBUG) << " free: " << (cmd->get_msg<CudaFreeHostMsg>()->ptr) << dendl;
    auto host_ptr = cmd->get_msg<CudaFreeHostMsg>()->ptr;
    if (false) {
        MEMPOOL->cpuMemoryDeAlloc(host_ptr);
    } else {
        auto &shms_id = cmd->get_worker()->shms_id;
        cudaCheck(cudaHostUnregister(host_ptr));
        if (0 > shmdt(host_ptr)) {
            perror("server fail to release share memory");
            exit(1);
        }
        if (0 > shmctl(shms_id[host_ptr], IPC_RMID, nullptr)) {
            perror("server fail to delete share memory");
            exit(1);
        }
        shms_id.erase(host_ptr);
    }
    cmd->finish<bool>(true);
}

void Conductor::do_cudamemset(const std::shared_ptr<Command> &cmd) {
    // dout(DEBUG) << " set address: " << cmd->get_msg<CudaMemsetMsg>()->devPtr << dendl;
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaMemsetMsg>();
    cudaCheck(::cudaMemset(msg->devPtr, msg->value, msg->count));
    cmd->finish<bool>(true);
}

void Conductor::do_cudamemcpy(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaMemcpyMsg>();
    //dout(DEBUG) << " copy from: " << msg->src << " to: " << msg->dst << dendl;
    cudaCheck(cudaMemcpy(msg->dst, msg->src, msg->count, msg->kind));
    cmd->finish<bool>(true);
}

// start @Kname in @ptx file, with launch @conf, kernel has @param with @size bytes
void
launchKernel(const string &ptx, const string &kname, LaunchConf conf, void *param, unsigned int size,
             stream_t stream = nullptr) {
    CUmodule cuModule;
    cudaCheck(static_cast<cudaError_t>(cuModuleLoad(&cuModule, ptx.c_str())));
    CUfunction func;
    dout(DEBUG) << "ptx: " << ptx << " kernel: " << kname << " address :" << func << dendl;
    cudaCheck(static_cast<cudaError_t>(cuModuleGetFunction(&func, cuModule, kname.c_str())));

    void *extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, param,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &size,
            CU_LAUNCH_PARAM_END
    };
    cudaCheck(static_cast<cudaError_t>(::cuLaunchKernel(func, conf.grid.x, conf.grid.y, conf.grid.z,
                                                        conf.block.x, conf.block.y, conf.block.z,
                                                        conf.share_memory,
                                                        stream,
                                                        nullptr, extra)));
    cudaCheck(static_cast<cudaError_t>(cuModuleUnload(cuModule)));
    cudaCheck(cudaStreamSynchronize(stream));
}

void Conductor::do_cudalaunchkernel(const std::shared_ptr<Command> &cmd) {
    KernelInstance ki(cmd->get_msg<CudaLaunchKernelMsg>(), cmd->get_device(), cmd->get_worker());
    ki.init();
    SCHEDULER->apply_slot(&ki);
    ki.launch();
    ki.sync();
    SCHEDULER->release_slot(&ki);
    cmd->finish<bool>(true);
}

//void Conductor::do_cudalaunchkernel(const std::shared_ptr<Command> &cmd) {
//    cudaCheck(::cudaSetDevice(cmd->get_device()));
//    auto msg = cmd->get_msg<CudaLaunchKernelMsg>();
//    msg->p_size = fillParameters(msg->param, msg->p_size, 0, 6, msg->conf.grid,
//                                 (msg->conf.grid.x * msg->conf.grid.y * msg->conf.grid.z));
//    launchKernel(msg->ptx, msg->kernel + string("Proxy"), msg->conf, msg->param, msg->p_size, cmd->get_stream());
//    cmd->finish<bool>(true);
//}

void Conductor::do_mocklaunchkernel(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<MockLaunchKernelMsg>();
    msg->p_size = fillParameters(msg->param, msg->p_size, 0, 6, msg->conf.grid,
                                 (msg->conf.grid.x * msg->conf.grid.y * msg->conf.grid.z));
    cmd->finish<bool>(true);
}

void Conductor::do_cudastreamcreate(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    // dout(DEBUG) << " cmd_id: " << cmd->get_id() << " create stream at device: " << cmd->get_device() << dendl;
    cudaStream_t ret;
    cudaCheck(cudaStreamCreate(&ret));
    cmd->finish<stream_t>(ret);
}

void Conductor::do_cudastreamsynchronize(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
//    dout(DEBUG) << " synchronize stream: at device: " << cmd->get_device() << " stream: " << cmd->get_stream() << dendl;
    cudaCheck(::cudaStreamSynchronize(cmd->get_stream()));
    cmd->finish<bool>(true);
}

void Conductor::do_cudagetdevicecount(const std::shared_ptr<Command> &cmd) {
    // dout(DEBUG) << " cuda get device count " << dendl;
    int count;
    cudaCheck(::cudaGetDeviceCount(&count));
    cmd->finish<int>(count);
}

void Conductor::do_cudaeventcreate(const std::shared_ptr<Command> &cmd) {
    cudaEvent_t event;
    // dout(DEBUG) << " create event at device: " << cmd->get_device() << " stream: " << cmd->get_stream() << dendl;
    cudaCheck(::cudaEventCreate(&event));
    cmd->finish<cudaEvent_t>(event);
}

void Conductor::do_cudaeventdestroy(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaEventDestroy(cmd->get_msg<CudaEventDestroyMsg>()->event));
    // dout(DEBUG) << " event destroy device: " << cmd->get_device() << " stream: " << cmd->get_stream() << dendl;
    cmd->finish<bool>(true);
}

void Conductor::do_cudaeventrecord(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaEventRecordMsg>();
    // dout(DEBUG) << " device: " << cmd->get_device() << " stream: " << cmd->get_stream() << dendl;
    cudaCheck(::cudaEventRecord(msg->event, msg->stream));
    cmd->finish<bool>(true);
}

void Conductor::do_cudaeventsynchronize(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaEventSyncMsg>();
    cudaCheck(::cudaEventSynchronize(msg->event));
    cmd->finish<bool>(true);
}

void Conductor::do_cudaeventelapsedtime(const std::shared_ptr<Command> &cmd) {
    cudaCheck(::cudaSetDevice(cmd->get_device()));
    auto msg = cmd->get_msg<CudaEventElapsedTimeMsg>();
    float ret;
    // dout(DEBUG) << " start device: " << cmd->get_device() << " stream: " << cmd->get_stream() << dendl;
    cudaCheck(::cudaEventElapsedTime(&ret, msg->start, msg->end));
    // dout(DEBUG) << " end device: " << cmd->get_device() << " stream: " << cmd->get_stream() << dendl;
    cmd->finish(ret);
}

bool singleTask(Task *t, int dev, bool *success, int *bind_dev, char result[128]);

void Conductor::do_multask(const std::shared_ptr<Command> &cmd) {
    auto msg = cmd->get_msg<MulTaskMsg>();
    int dev_count = DEVICES->counts();
    future<bool> result[MAX_TASK_NUM];
    mgpu::MulTaskRet res;

    for (int i = 0; i < msg->task_num; i++) {
        int dev = DEVICES->GetBestDev(msg->task[i]);
        res.success[i] = false;
        auto ret = std::async(&singleTask, (msg->task + i), dev, res.success + i, res.bind_dev + i, res.msg[i]);
        result[i] = std::move(ret);
    }

    for (int i = 0; i < msg->task_num; i++) {
        result[i].get();
    }
    cmd->finish<MulTaskRet>(res);
}

#define COPYMESSAGE(x) \
{                      \
   cudaError_t err = (cudaError_t) (x); \
   if(err != cudaSuccess) {             \
        memcpy(result, cudaGetErrorString(err), 128); \
        return false;\
   }\
}

bool singleTask(Task *t, int dev, bool *success, int *bind_dev, char result[128]) {
    COPYMESSAGE(cudaSetDevice(dev));
    *bind_dev = dev;

    unsigned int p_size = 0;
    stream_t stream;
    COPYMESSAGE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // copy host data to GPU
    for (int i = 0; i < t->hdn; i++) {
        void *dev_ptr;
        COPYMESSAGE(cudaMalloc(&dev_ptr, t->hds[i]));
        void *src_ptr = reinterpret_cast<void *>(*((unsigned long long *) (t->param + p_size)));
        COPYMESSAGE(cudaMemcpy(dev_ptr, src_ptr, t->hds[i], cudaMemcpyHostToDevice));
        p_size = fillParameters(t->param, p_size, dev_ptr);
        memcpy(result + i * sizeof(dev_ptr), &dev_ptr, sizeof(dev_ptr));
    }
    // allocate GPU for result
    for (int i = 0; i < t->dn; i++) {
        void *dev_ptr;
        COPYMESSAGE(cudaMalloc(&dev_ptr, t->dev_alloc_size[i]));
        p_size = fillParameters(t->param, p_size, dev_ptr);
        memcpy(result + (i + t->hdn) * sizeof(dev_ptr), &dev_ptr, sizeof(dev_ptr));
    }
    launchKernel(t->ptx, t->kernel, t->conf, t->param, t->p_size, stream);
    COPYMESSAGE(cudaStreamSynchronize(stream));
    DEVICES->ReleaseDev(dev, *t);
    *success = true;
    return *success;
}

void Conductor::do_matrixmultgpu(const std::shared_ptr<Command> &cmd) {
    auto msg = cmd->get_msg<MatrixMulMsg>();
    Matrix A = msg->A;
    Matrix B = msg->B;
    auto conf = cmd->get_msg<MatrixMulMsg>()->conf;
    int shm_id = shmget(IPC_PRIVATE, sizeof(float) * A.height * B.width, IPC_CREAT);
    if (shm_id < 0) {
        perror("fail to shmget");
        exit(1);
    } else {
        dout(DEBUG) << " shm_id: " << shm_id << dendl;
    }
    void *res = shmat(shm_id, nullptr, 0);
    auto &shms_id = cmd->get_worker()->shms_id;
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

    for (int i = 0; i < device->counts(); i++) {
        dout(DEBUG) << " device: " << i << " do matrix multi " << dendl;
        cudaCheck(::cudaSetDevice(i));
        cudaCheck(::cudaEventCreateWithFlags(&(starts[i]), cudaEventBlockingSync));
        cudaCheck(::cudaEventCreateWithFlags(&(ends[i]), cudaEventBlockingSync));
        void *dev_A, *dev_B, *dev_C;
        cudaCheck(::cudaMalloc(&dev_A, sizeof(float) * A.height * A.width));
        cudaCheck(::cudaMalloc(&dev_B, sizeof(float) * B.height * B.width));
        cudaCheck(::cudaMalloc(&dev_C, sizeof(float) * A.height * B.width));
        cudaEventRecord(starts[i], nullptr);
        cudaMemcpyAsync(dev_A, A.data, sizeof(float) * A.height * A.width, cudaMemcpyHostToDevice, nullptr);
        cudaMemcpyAsync(dev_B, B.data, sizeof(float) * B.height * B.width, cudaMemcpyHostToDevice, nullptr);
        CUmodule module;
        cudaCheck(static_cast<cudaError_t>(::cuModuleLoad(&module, "/opt/custom/ptx/matrixMul.ptx")));
        CUfunction func;
        cudaCheck(static_cast<cudaError_t>(::cuModuleGetFunction(&func, module, "matrixMulProxy")));
        char params[1024];
        auto p_size = fillParameters(params, 0, static_cast<void *>(dev_C), static_cast<void *>(dev_A),
                                     static_cast<void *>(dev_B), A.width, B.width, 0, 2, conf.grid,
                                     (conf.grid.x * conf.grid.y));
        void *extra[] = {
                CU_LAUNCH_PARAM_BUFFER_POINTER, params,
                CU_LAUNCH_PARAM_BUFFER_SIZE, &p_size,
                CU_LAUNCH_PARAM_END
        };
//        cudaCheck(static_cast<cudaError_t>(::cuLaunchKernel(func, WORKER_GRID, 1, 1, conf.block.x, conf.block.y,
//                                                            conf.block.z, conf.share_memory, 0, nullptr, extra)));
        int area = A.height * B.width / device->counts();
        cudaCheck(::cudaMemcpyAsync(static_cast<float *>(res) + area * i, static_cast<float *>(dev_C) + area * i,
                                    sizeof(float) * area, cudaMemcpyDeviceToHost, 0));
        cudaCheck(::cudaEventRecord(ends[i], nullptr));
    }
    for (int i = 0; i < device->counts(); i++) {
        cudaCheck(::cudaEventSynchronize(ends[i]));
    }
    dout(DEBUG) << " finish " << dendl;
    cmd->finish<CudaMallocHostRet>(mgpu::CudaMallocHostRet{res, shm_id});
}