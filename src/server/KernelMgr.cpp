//
// Created by root on 2021/10/31.
//
#include "server/server.h"
#include "server/device.h"
#include "server/kernel.h"
#include "common/Log.h"
#include "common/helper.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace mgpu;
using namespace rapidjson;
using namespace std;

void KernelMgr::init() {
    string path = "/opt/custom/kernels.json";

    struct stat file{};
    stat(path.c_str(), &file);
    char *json = new char[file.st_size + 1];
    int fd = open(path.c_str(), O_RDONLY);
    read(fd, json, file.st_size);
    close(fd);
    json[file.st_size] = '\0';

    Document d;
    d.Parse(json);
    delete[] json;

    //init kernels
    for (auto it = d.MemberBegin(); it != d.MemberEnd(); ++it) {
        string name = it->name.GetString();
        Value &v = it->value.GetObj();
        kns[name] = Kernel{
                .property = 0.0,
                .insts_per_warp = v["insts_per_warp"].GetDouble(),
                .memTrans_per_warp = v["memTrans_per_warp"].GetDouble(),
                .aveBytes_per_trans = v["aveBytes_per_trans"].GetDouble(),
                .regs = v["register_per_thread"].GetInt(),
                .shms = v["share_mem_per_block"].GetInt(),

        };
        kns[name].property = kns[name].memTrans_per_warp / kns[name].insts_per_warp * kns[name].aveBytes_per_trans;
    }
}

void KernelMgr::run() {
//    obverse();
}

void KernelMgr::obverse() {
    cout << "=======Kernels Config=========" << endl;
    for (auto &k : kns) {
        cout << "\tKernel: " << k.first << endl;
        cout << "\t\tproperty: " << k.second.property << endl;
        cout << "\t\tregister per thread: " << k.second.regs << endl;
        cout << "\t\tshare mem per block: " << k.second.shms << endl;
        cout << "\t\tinstructions executed per warp: " << k.second.insts_per_warp << endl;
        cout << "\t\tmemory transactions per warp: " << k.second.memTrans_per_warp << endl;
        cout << "\t\taverage byte counts per transaction: " << k.second.aveBytes_per_trans << endl;
    }
}

int calMaxBlock(const Device::GPU *gpu, const Kernel &prop, dim3 &block) {
    unsigned int max_blocks = gpu->max_blocks;
    max_blocks = min(max_blocks,
                     (gpu->max_warps * gpu->warp_size) / (block.x * block.y * block.z)); /* max_threads_per_sm limit */
    max_blocks = min(max_blocks,
                     (gpu->share_mem) / (prop.shms));                                    /* share memory limit */
    max_blocks = min(max_blocks, gpu->regs / (prop.regs * block.x * block.y * block.z)); /* register limit */

    return max_blocks;
}

KernelInstance::KernelInstance(CudaLaunchKernelMsg *msg, int gpuid, ProxyWorker* worker) : finished(false) {
    // name
    name = msg->kernel;
    if (KERNELMGR->kns.find(name) == KERNELMGR->kns.end()) {
        cout << "Kernel " << name << " not exist!" << endl;
        exit(EXIT_FAILURE);
    }
    prop = KERNELMGR->kns[name];

    // config
    block = msg->conf.block;
    grid = msg->conf.grid;
    stream = msg->conf.stream;

    // parameter
    memcpy(param_buf, msg->param, msg->p_size);
    p_size = msg->p_size;

    // cpu conf
    gpu = DEVICES->getDev(gpuid);
    cbytes = sizeof(int) * (6 + gpu->sms);
    max_block_per_sm = calMaxBlock(gpu, prop, block);

    // mod
    worker->get_funcs(msg->ptx, (name + "_V1"), func_v1, devConf, cpuConf, cbytes);
}

void KernelInstance::init() {
    cpuConf[0] = 0 + (5 << 8) + (max_block_per_sm << 16);  /* sms flag */
    cpuConf[1] = grid.x * grid.y * grid.z;                 /* total blocks */
    cpuConf[2] = 0;                                        /* finished blocks */
    cpuConf[3] = grid.x;                                   /* origin grid */
    cpuConf[4] = grid.y;                                   /* origin grid */
    cpuConf[5] = grid.z;                                   /* origin grid */
    for (int i = 6; i < 6 + gpu->sms; i++) {                /* sm-worker count */
        cpuConf[i] = 0;
    }
    cudaCheck(cudaSetDevice(gpu->ID));
    // set run time resource conf
    cudaCheck(cudaMemcpyAsync((void *) devConf, cpuConf, cbytes, cudaMemcpyHostToDevice, stream));
    grid_v1 = dim3(max_block_per_sm * gpu->sms, 1, 1);
}

void KernelInstance::launch() {
    void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, param_buf,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &p_size,
                     CU_LAUNCH_PARAM_END};
    cudaCheck(cuLaunchKernel(func_v1, grid_v1.x, grid_v1.y, grid_v1.z, block.x, block.y,
                             block.z, 0, stream, nullptr, extra));
}

void KernelInstance::sync() {
    cudaCheck(cudaStreamSynchronize(stream));
}

void KernelInstance::occupancy_all(stream_t ctrl) {
    if ((cpuConf[0] >> 16) != max_block_per_sm)
        set_config(0, gpu->sms, max_block_per_sm, ctrl);
}

void KernelInstance::set_config(int sm_low, int sm_high, int wlimit, stream_t ctrl) {
    cpuConf[0] = sm_low + (sm_high << 8) + (wlimit << 16);
    cudaCheck(cudaMemcpyAsync((void *) devConf, cpuConf, sizeof(int), cudaMemcpyHostToDevice, ctrl));
}

void KernelInstance::get_runinfo(stream_t ctrl) {
    unsigned long long off = 6 * sizeof(int);
    cudaCheck(cudaMemcpyAsync(cpuConf + off, (void *) (devConf + off), cbytes - off, cudaMemcpyDeviceToHost, ctrl));
    cudaCheck(cudaStreamSynchronize(ctrl));
}

void KernelInstance::print_runinfo() {
    dout(LOG) << "Kernel name : " << name << dendl;
    dout(LOG) << "assigned sm low: " << (cpuConf[0] & 0xff) << " sm high: " << ((cpuConf[0] & 0xff00) >> 8)
              << " worker limit: " << (cpuConf[0] >> 16) << dendl;
    for (int i = 0; i < gpu->sms; i++) {
        dout(LOG) << "sm : " << i << " have worker: " << cpuConf[6 + i] << dendl;
    }
    dout(LOG) << "===================" << dendl;
}

int KernelInstance::get_config() {
    return cpuConf[0];
}

KernelInstance::~KernelInstance() {
}