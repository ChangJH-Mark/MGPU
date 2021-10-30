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

#ifndef MAX_MSG_SIZE
#define MAX_MSG_SIZE (1 << 12) // 4KB
#endif

using namespace mgpu;

void ProxyWorker::work() {
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);

    // make sure futex is ok
    while(c_fut.shm_ptr == BAD_UADDR);
    while(s_fut.shm_ptr == BAD_UADDR);

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

        auto cmd = std::make_shared<Command>((AbMsg *) buf, s_fut.shm_ptr, this);
        CONDUCTOR->conduct(cmd);
    }
}