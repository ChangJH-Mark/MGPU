//
// Created by root on 2021/7/3.
//

#include "server/proxy_worker.h"
#include "server/conductor.h"
#include "server/server.h"
#include "server/commands.h"
#include <string.h>
#include <memory>
#include <linux/futex.h>
#include <sys/mman.h>
#include <fcntl.h>

#ifndef MAX_MSG_SIZE
#define MAX_MSG_SIZE (1 << 12) // 4KB
#endif

using namespace mgpu;

void ProxyWorker::init_pid() {
    int size = read(m_conn, &m_p, sizeof(pid_t));
    if(size != sizeof(pid_t)) {
        printf("error to read pid from client\n");
        exit(1);
    }
    if(m_p == 0) {
        printf("m_p is set to 0\n");
        exit(1);
    }
}

int ProxyWorker::init_shm() {
    using std::string;
    string names[2] = {"mgpu.0." + to_string(m_p), "mgpu.1." + to_string(m_p)};
    string root = "/dev/shm/";
    void *shms[2];
    int cnt = 0;
    for(auto & n : names) {
        auto bytes = read(m_conn, shms + cnt, sizeof(void *));
        assert(bytes == sizeof(void *));
        if(0 != access((root + n).c_str(), F_OK)) {
            printf("error to open shm %s\n", (root + n).c_str());
            exit(1);
        }
        int fd = shm_open(n.c_str(), O_CLOEXEC | O_RDWR, 0644);
        assert(fd > 0);
        if(shms[cnt] != mmap(shms[cnt], PAGE_SIZE,PROT_WRITE | PROT_READ, MAP_SHARED , fd, 0)) {
            printf("mgpu mapped at distinct address\n");
            exit(1);
        }
        close(fd);
        cnt++;
    }
    c_fut = Futex(shms[0]);
    s_fut = Futex(shms[1]);
    return 0;
}

void ProxyWorker::work() {
    //pthread_setname_np(native_handle(), "proxy_worker");
    init_pid();
    init_shm();

    // start listen to client
    while (!m_stop) {
        if(!c_fut.ready()) {
            int res = syscall(__NR_futex, c_fut.shm_ptr, FUTEX_WAIT, NOT_READY, NULL);
            if(res == -1) {
                if(errno != EAGAIN) {
                    printf("error to recv from client, syscall return %d, errno is %d before read data\n", res, errno);
                    exit(1);
                }
            }
        }
        c_fut.copyTo(buf, c_fut.size());
        buf[c_fut.size()] = '\0';
        c_fut.setState(NOT_READY);

        auto cmd = std::make_shared<Command>((AbMsg *) buf, s_fut.shm_ptr);
        CONDUCTOR->conduct(cmd);
    }
    close(m_conn);
}