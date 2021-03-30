//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_COMMANDS_H
#define FASTGPU_COMMANDS_H
#include <functional>
#include <sys/socket.h>
#include <unistd.h>
#include "common/message.h"

namespace mgpu {
    class Command {
    public:
        Command(AbMsg* m, uint cli) : type(m->type), msg(m), socket(cli), device(0), status(std::make_shared<bool>(false)) {}
        Command(const Command &) = delete;
        Command(Command &&origin)  noexcept {
            socket = origin.socket;
            msg = origin.msg;
            origin.msg = nullptr;
        }
        ~Command(){
            free(msg);
        }

    public:
        std::shared_ptr<bool> get_status() {return status;}
        msg_t get_type() const{ return type;}
        uint get_device() const{ return device;}
        template<class T> T* get_msg() { return (T*)msg;}
        template<class T> void finish(T);
        template<class T> void finish(T*, uint);

    private:
        std::shared_ptr<bool> status;
    private:
        uint socket;
        msg_t type;
        AbMsg* msg;
        uint device;
    };

    template<class T>
    void Command::finish(T value) {
        ::send(socket, &value, sizeof(T), 0);
        ::close(socket);
        *status = true;
    }

    template<typename T>
    void Command::finish(T* ptr, uint num) {
        ::send(socket, ptr, sizeof(T) * num, 0);
        ::close(socket);
        *status = true;
    }
}

#endif //FASTGPU_COMMANDS_H
