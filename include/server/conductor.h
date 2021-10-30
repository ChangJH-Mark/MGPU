//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_CONDUCTOR_H
#define FASTGPU_CONDUCTOR_H

#include <future>
#include <map>
#include "common/ThreadPool.h"
#include "server/commands.h"
#include "mod.h"

namespace mgpu {
    class Conductor : public Module {
    public:
        Conductor() {
            joinable = false;
        };

        Conductor(const Conductor &) = delete;

        Conductor(const Conductor &&) = delete;

        void init() override;

        void run() override {};

        void destroy() override {
            stopped = true;
        };

        void join() override {};

    public:
        void conduct(const std::shared_ptr<Command> &cmd);

    private:
        std::map<int, void (*)(const std::shared_ptr<Command> &)> func_table;

    private:
        static void do_cudamalloc(const std::shared_ptr<Command> &cmd);

        static void do_mockmalloc(const std::shared_ptr<Command> &cmd);

        static void do_cudamallochost(const std::shared_ptr<Command> &cmd);

        static void do_cudafree(const std::shared_ptr<Command> &cmd);

        static void do_cudafreehost(const std::shared_ptr<Command> &cmd);

        static void do_cudamemset(const std::shared_ptr<Command> &cmd);

        static void do_cudamemcpy(const std::shared_ptr<Command> &cmd);

        static void do_cudalaunchkernel(const std::shared_ptr<Command> &cmd);

        static void do_mocklaunchkernel(const std::shared_ptr<Command> &cmd);

        static void do_cudastreamcreate(const std::shared_ptr<Command> &cmd);

        static void do_cudastreamsynchronize(const std::shared_ptr<Command> &cmd);

        static void do_cudagetdevicecount(const std::shared_ptr<Command> &cmd);

        static void do_cudaeventcreate(const std::shared_ptr<Command> &cmd);

        static void do_cudaeventdestroy(const std::shared_ptr<Command> &cmd);

        static void do_cudaeventrecord(const std::shared_ptr<Command> &cmd);

        static void do_cudaeventsynchronize(const std::shared_ptr<Command> &cmd);

        static void do_cudaeventelapsedtime(const std::shared_ptr<Command> &cmd);

        static void do_matrixmultgpu(const std::shared_ptr<Command> &cmd);

        static void do_multask(const std::shared_ptr<Command> &cmd);
    };
}
#endif //FASTGPU_CONDUCTOR_H
