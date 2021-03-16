//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_COMMANDS_H
#define FASTGPU_COMMANDS_H
#include "common/message.h"

namespace mgpu {
    class Command {
    public:
        Command(AbMSG* m) : type(m->type), msg(m) {};
        Command(const Command &) = delete;
        Command(Command &&origin)  noexcept {
            msg = origin.msg;
            origin.msg = nullptr;
        }
    private:
        message_t type;
        AbMSG* msg;
    };
}

#endif //FASTGPU_COMMANDS_H
