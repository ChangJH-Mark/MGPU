//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_COMMANDS_H
#define FASTGPU_COMMANDS_H

#include <functional>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include "common/message.h"

namespace mgpu {
    class Command {
    public:
        Command(AbMsg *m, uint cli) : type(m->type), msg(m), conn(cli), stream(m->stream),
                                      status(std::make_shared<bool>(false)) {
            id = id_cnt++;
            device = m->key & 0xffff;
            pid = m->key >> 16;
        }

        Command(const Command &) = delete;

        Command(Command &&origin) noexcept {
            conn = origin.conn;
            type = origin.type;
            msg = origin.msg;
            device = origin.device;
            stream = origin.stream;
            pid = origin.pid;
            origin.msg = nullptr;
        }

        ~Command() {
            delete msg;
        }

    public:
        std::shared_ptr<bool> get_status() { return status; }

        api_t get_type() const { return type; }

        uint get_device() const { return device; }

        stream_t get_stream() const { return stream; }

        pid_t get_pid() const { return pid; }

        uint get_id() const { return id; }

        template<class T>
        T *get_msg() { return (T *) msg; }

        template<class T>
        void finish(T &);

        template<class T>
        void finish(T &&);

        template<class T>
        void finish(T *, uint);

    private:
        std::shared_ptr<bool> status;
    private:
        uint conn;
        pid_t pid;
        api_t type;
        AbMsg *msg;
        uint device;
        stream_t stream;
        uint id;
        static std::atomic<uint> id_cnt;
    };

    template<class T>
    void Command::finish(T &value) {
        finish<T>(std::move(value));
    }

    template<class T>
    void Command::finish(T &&value) {
        ::send(conn, &value, sizeof(T), 0);
        *status = true;
    }


    template<typename T>
    void Command::finish(T *ptr, uint num) {
        ::send(conn, ptr, sizeof(T) * num, 0);
        *status = true;
        delete[] ptr;
    }
}

#endif //FASTGPU_COMMANDS_H
