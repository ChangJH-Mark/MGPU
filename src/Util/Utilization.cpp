//
// Created by root on 2021/3/7.
//

#include <nvml.h>
#include <iostream>
using namespace std;

int main() {
    auto err = nvmlInit_v2();
    if(err != NVML_SUCCESS) {
        cout << " err happen " << endl;
        exit(EXIT_FAILURE);
    }
    nvmlDevice_t device;
    err = nvmlDeviceGetHandleByIndex(0, &device);
    nvmlMemory_t mem_usage;
    err = nvmlDeviceGetMemoryInfo(device, &mem_usage);
    if(err != NVML_SUCCESS) {
        cout << " err happen " << endl;
        exit(EXIT_FAILURE);
    }
    if (err != NVML_SUCCESS) {
        cout << "mem err happen errcode " << err << endl;
        exit(EXIT_FAILURE);
    }
    cout << "total: " << mem_usage.total / (1 << 20) << endl <<
    "used: " << mem_usage.used / ( 1 << 20) << endl <<
    "free: " << mem_usage.free / (1 << 20) << endl;
    exit(EXIT_SUCCESS);
}