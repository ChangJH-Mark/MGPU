//
// Created by mark on 2021/3/1.
//

#include <iostream>
#include <nvml.h>

#define CHECK_ERR(x) if(x!=NVML_SUCCESS){ \
exit(1);}


using namespace std;

int main() {
    auto err = nvmlInit_v2();
    CHECK_ERR(err);

    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex_v2(0, &dev);
    nvmlEnableState_t stat;
    err = nvmlDeviceSetAccountingMode(dev, NVML_FEATURE_ENABLED);
    CHECK_ERR(err);
    err = nvmlDeviceGetAccountingMode(dev, &stat);
    cout << "accounting mode: " << stat << endl;
    unsigned int cnt = 4000;
    unsigned int pids[4000];
    err = nvmlDeviceGetAccountingPids(dev, &cnt, pids);
    CHECK_ERR(err);
    cout << "count : " << cnt << endl;
    for (int i = 0; i < cnt; i++) {
        nvmlAccountingStats_t stat;
        nvmlDeviceGetAccountingStats(dev, pids[i], &stat);
        cout << "time: " << stat.time << "\n"
             << "gpuUtilization: " << stat.gpuUtilization << "\n"
             << "isRunning: " << stat.isRunning << "\n"
             << "maxMemoryUsage: " << stat.maxMemoryUsage << "\n"
             << "memoryUtilization: " << stat.memoryUtilization << "\n"
             << "startTime: " << stat.startTime << endl;
    }
    cout << endl;

    err = nvmlShutdown();
    CHECK_ERR(err);
    cout << " test passed " << endl;
}