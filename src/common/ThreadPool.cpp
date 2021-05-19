//
// Created by root on 2021/5/18.
//

#include "common/ThreadPool.h"

ThreadPool::ThreadPool(uint init_num, uint max_num) : max_num(max_num), stopped(false) {
    for(int i=0; i<init_num && !stopped; i++)
    {
        idlThrNums++;
        workers.emplace_back(&ThreadPool::worker, this);
        workers[workers.size() - 1].detach();
    }
}

void ThreadPool::worker() {
    while(!stopped)
    {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queue_lock);
            while(!stopped && tasks.empty())
                cv.wait(lock);
            if(stopped)
                break;
            task = tasks.front();
            tasks.pop();
        }
        idlThrNums --;
        task();
        idlThrNums ++;
    }
}