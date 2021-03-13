//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_RECEIVER_H
#define FASTGPU_RECEIVER_H
#include <thread>
#include "mod.h"

namespace mgpu{
    class Receiver : public Module {
    public:
        Receiver(){
            hasThread = true;
        };
        void init() override;
        void run() override{}
        void join() override{};

    private:
        uint max_worker;
    };
}
#endif //FASTGPU_RECEIVER_H
