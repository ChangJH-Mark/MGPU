//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_COMMANDS_H
#define FASTGPU_COMMANDS_H

#include <functional>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cstring>
#include <linux/futex.h>
#include <sys/time.h>
#include "common/message.h"

namespace mgpu {
    class Command {
    public:
        Command(AbMsg *m, void * shm_ptr) : type(m->type), fut(shm_ptr), msg(m), stream(m->stream),
                                            status(std::make_shared<bool>(false)) {
            id = id_cnt++;
            device = m->key & 0xffff;
            pid = m->key >> 16;
        }

        Command(const Command &) = delete;

        Command(Command &&origin) noexcept {
            fut = origin.fut;
            type = origin.type;
            msg = origin.msg;
            device = origin.device;
            stream = origin.stream;
            pid = origin.pid;
            origin.msg = nullptr;
        }

        ~Command() {
        }

    public:
        std::shared_ptr<bool> get_status() { return status; }

        api_t get_type() const { return type; }

        uint get_id() const { return id; }

        uint get_device() const { return device; }

        stream_t get_stream() const { return stream; }

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
        Futex fut;
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
        fut.setData(&value, sizeof(T));
        fut.setState(READY);
        // client did not take data
        if(fut.ready()) {
            int res = syscall(__NR_futex, fut.shm_ptr, FUTEX_WAKE, 1, NULL);
            if (res == -1) {
                printf("error to wake client, syscall return %d, errno is %d\n", res, errno);
                exit(1);
            }
        }
        *status = true;
    }

    template<typename T>
    void Command::finish(T *ptr, uint num) {
        fut.setData(ptr, sizeof(T) * num);
        fut.setState(READY);
        // client did not take data
        if(fut.ready()) {
            int res = syscall(__NR_futex, fut.shm_ptr, FUTEX_WAKE, 1, NULL);
            if (res == -1) {
                printf("error to wake client, syscall return %d, errno is %d\n", res, errno);
                exit(1);
            }
        }
        *status = true;
        delete[] ptr;
    }
}

#endif //FASTGPU_COMMANDS_H
