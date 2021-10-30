//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_MOD_H
#define FASTGPU_MOD_H
namespace mgpu {
    class Module {
    public:
        Module() : joinable(false), stopped(false) {}
        virtual void init(){}
        virtual void run(){}
        virtual void join(){}
        virtual void destroy(){}
        bool joinable;
        bool stopped;
        virtual ~Module() {}
    };
}
#endif //FASTGPU_MOD_H
