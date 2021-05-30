//
// Created by root on 2021/5/28.
//

#ifndef FASTGPU_LOG_H
#define FASTGPU_LOG_H

#include <chrono>
#include <pthread.h>
#include <condition_variable>
#include <ctime>
#include <iomanip>
#include <string>
#include <memory>
#include <map>
#include "ThreadPool.h"

#define DEBUG 3
#define NOTICE 2
#define LOG 1

#define dout(l) (LogEntry(logger, l, std::chrono::system_clock::now(), pthread_self(), __FILE__, __LINE__, __FUNCTION__))
#define  dendl e

class log_endl {
};
static log_endl e;

class LogPool;
extern shared_ptr<LogPool> logger;

class LogPool {
public:
    explicit LogPool(int level) : log_level(level), max_out(20), pool(2, 5) {
        pool.commit(&LogPool::flush, this);
    }

    void destroy() {
        pool.stop();
        mut.lock();
        for (auto &log : logs) {
            cout << log.second << "\n";
        }
        cout << std::flush;
        mut.unlock();
    }

public:
    void submit(int level, chrono::system_clock::time_point time, string &msg) {
        if (level > log_level)
            return;
        pool.commit(&LogPool::insert, this, time, msg);
    }

private:
    int log_level, max_out;
    ThreadPool pool;
    map<chrono::system_clock::time_point, string> logs;
    std::mutex mut;
    std::condition_variable cv;

    void insert(chrono::system_clock::time_point time, string& msg) {
        mut.lock();
        logs.emplace(time, std::move(msg));
        mut.unlock();
        cv.notify_one();
    }

    void flush() {
        while (1) {
            std::unique_lock<std::mutex> ulk(mut);
            cv.wait(ulk, [&]() -> bool { return !logs.empty(); });
            int count = 0;
            for (auto it = logs.begin(); it != logs.end() && count < max_out; count++) {
                cout << it->second << "\n";
                logs.erase(it++);
            }
            cout << std::flush;
            ulk.unlock();
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    }
};

class LogEntry {
public:
    LogEntry(const shared_ptr<LogPool> &p, int l, chrono::system_clock::time_point t, pthread_t tid, const string &file,
             int line, const string &func)
            : pool(p), level(l), time(t) {
        message += to_string(level) + " ";
        time_t epoch_time = chrono::system_clock::to_time_t(t);
        auto *ptm = std::localtime(&epoch_time);
        char times[40] = {0};
        strftime(times, 40, "%Y-%m-%d %H:%M:%S.", ptm);
        message += times;
        char micros[7] = {0};
        sprintf(micros, "%06d", chrono::duration_cast<chrono::microseconds>(t.time_since_epoch()).count() % 1000000);
        message += micros;
        message += " " + to_string(tid) + " " + file + " " + to_string(line) + " " + func + " ";
    }

    LogEntry &operator<<(const string& msg) {
        message += msg;
        return *this;
    }

    LogEntry &operator<<(int msg) {
        message += std::to_string(msg);
        return *this;
    }
    LogEntry &operator<<(void *ptr) {
        std::stringstream ss;
        ss << hex << ptr;
        message += ss.str();
        return *this;
    }
#ifdef FASTGPU_TASK_H
    struct TASK_KEY;
    friend LogEntry &operator<<(LogEntry &le, const TASK_KEY &key);
#endif

    LogEntry &operator<<(log_endl e) {
        pool->submit(level, time, message);
        return *this;
    }

private:
    std::shared_ptr<LogPool> pool;
    int level;
    std::chrono::system_clock::time_point time;
    string message;
};

#endif //FASTGPU_LOG_H
