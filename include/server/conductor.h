//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_CONDUCTOR_H
#define FASTGPU_CONDUCTOR_H
#include <future>
#include <unordered_map>
#include "server/commands.h"
#include "mod.h"

namespace mgpu {
    class Conductor : public Module{
    public:
        Conductor() {
            joinable = false;
        };

        virtual void init() override{};
        virtual void run() override{};
        virtual void destroy() override{};
        virtual void join() override{};

    public:
        std::shared_ptr<bool> conduct(std::shared_ptr<Command> cmd);

    private:
        std::unordered_map<void*, int> shms_id;

    private:
        cudaStream_t get_stream(uint device, uint key);
        void do_cudamalloc(const std::shared_ptr<Command>& cmd);
        void do_cudamallochost(const std::shared_ptr<Command>& cmd);
        void do_cudafree(const std::shared_ptr<Command>& cmd);
        void do_cudafreehost(const std::shared_ptr<Command>& cmd);
        void do_cudamemset(const std::shared_ptr<Command>& cmd);
        void do_cudamemcpy(const std::shared_ptr<Command>& cmd);
        void do_cudalaunchkernel(const std::shared_ptr<Command>& cmd);
        void do_cudastreamcreate(const std::shared_ptr<Command>& cmd);
        void do_cudastreamsynchronize(const std::shared_ptr<Command>& cmd);
        void do_matrixmultgpu(const std::shared_ptr<Command>& cmd);
    };
}
#endif //FASTGPU_CONDUCTOR_H
