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
        kns[name] = Kernel{.property = v["property"].GetFloat(), .regs = v["register_per_thread"].GetInt(), .shms = v["share_mem_per_block"].GetInt()};
    }
}

void KernelMgr::obverse() {
    cout << "=======Kernels Config=========" << endl;
    for (auto &k : kns) {
        cout << "\tKernel: " << k.first << endl;
        cout << "\t\tproperty: " << k.second.property << endl;
        cout << "\t\tregister per thread: " << k.second.regs << endl;
        cout << "\t\tshare mem per block: " << k.second.shms << endl;
    }
}

int calMaxBlock(const Device::GPU* gpu, const Kernel& prop) {
    return prop.shms;
}

KernelInstance::KernelInstance(CudaLaunchKernelMsg *msg, int gpuid) : dev(gpuid) {
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

    // mod
    cudaCheck(cuModuleLoad(&mod, msg->ptx));
    cudaCheck(cuModuleGetFunction(&func, mod, msg->kernel));
    cudaCheck(cuModuleGetFunction(&func_v1, mod, (name + "_V1").c_str()));
    cudaCheck(cuModuleGetGlobal(&conf_ptr, nullptr, mod, "configs"));

    // parameter
    memcpy(param_buf, msg->param, msg->p_size);
    p_size = msg->p_size;

    // cpu conf
    auto gpu = DEVICES->getDev(gpuid);
    conf = new int[gpu->sms + 6];
    max_block_per_sm = calMaxBlock(gpu, prop);
}

KernelInstance::~KernelInstance() {
    cudaCheck(cuModuleUnload(mod));
    delete[] conf;
}