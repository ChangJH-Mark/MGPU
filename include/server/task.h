//
// Created by root on 2021/5/27.
//

#ifndef FASTGPU_TASK_H
#define FASTGPU_TASK_H
#include "mod.h"
#include <list>
#include <memory>
#include <functional>

namespace mgpu {
    struct TASK_KEY {
        uint key; // pid << 16 + device
        stream_t stream;
    };

    struct CompareKey {
        int operator()(const TASK_KEY &x, const TASK_KEY &k) const {
            if (x.key < k.key) {
                return 1;
            } else if (x.key == k.key && x.stream < k.stream) {
                return 1;
            } else
                return 0;
        }
    };

    class Task : public Module {
    Task() {
        joinable = false;
    }

    public:
        void init() {}
        void run() {}

    private:
        typedef std::list<std::shared_ptr<Command>> TASK_LIST;
        std::unordered_map<TASK_KEY, std::pair<std::mutex, TASK_LIST>> tasks;
    };
}

#endif //FASTGPU_TASK_H
