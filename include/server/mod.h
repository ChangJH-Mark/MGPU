//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_MOD_H
#define FASTGPU_MOD_H
namespace mgpu {
    class Module {
    public:
        Module() : hasThread(false) {}
        virtual void init(){}
        virtual void run(){}
        virtual void join(){}
        virtual void destroy(){}
        bool hasThread;
    };
}
#endif //FASTGPU_MOD_H
