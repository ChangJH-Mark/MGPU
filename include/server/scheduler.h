//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_SCHEDULER_H
#define FASTGPU_SCHEDULER_H
#include <thread>
#include "mod.h"
#include "server/server.h"

namespace mgpu {
    class Scheduler : public Module {
    public:
        Scheduler() {
            joinable = true;
        };

        virtual void init() override;
        virtual void run() override;
        virtual void destroy() override;
        virtual void join() override;

    private:
        void do_scan();

    private:
        std::thread scanner;
    };
}
#endif //FASTGPU_SCHEDULER_H
