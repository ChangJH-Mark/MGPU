//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_SCHEDULER_H
#define FASTGPU_SCHEDULER_H

#include <thread>
#include <list>
#include "mod.h"
#include "server/server.h"
#include "server/kernel.h"

using std::list;

namespace mgpu {
    class Scheduler : public Module {
    public:
        Scheduler() {
            slot[0] = nullptr;
            slot[1] = nullptr;
            run_cnt = 0;
            joinable = true;
            stopped = false;
        };

        Scheduler(const Scheduler &) = delete;

        Scheduler(const Scheduler &&) = delete;

        void init() override;

        void run() {};

        void destroy() override;

        void join() override;

        ~Scheduler() override = default;

        void apply_slot(KernelInstance *);

        void release_slot(KernelInstance *);

    private:
        bool find_corun(KernelInstance *running, KernelInstance **candidate);

    private:
        list<KernelInstance *> pending;
        mutex spin;                     // spin lock for pending
        stream_t ctrl;                  // a stream for control devConfs

        condition_variable signal;      // signal indicates that make schedule decision

        atomic<KernelInstance *> slot[2]; // two run slot for KernelInstance
        atomic<int> run_cnt;
        mutex run_stat;                 // state lock, for slots & run_cnt

        thread s;                       // scheduler thread

        void sched(); // schedule thread
    };
}
#endif //FASTGPU_SCHEDULER_H
