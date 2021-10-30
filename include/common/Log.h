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
#include <iostream>
#include "ThreadPool.h"
using namespace std;

#define DEBUG 3
#define NOTICE 2
#define LOG 1

extern int max_level;

class LogPool {
public:
    explicit LogPool(int level) : max_out(200), pool(2, 5), stopped(false) {
        pool.commit(&LogPool::flush, this);
        max_level = level;
    }

    void destroy() {
        stopped = true;
        cv.notify_all();
        pool.stop();
    }

public:
    void submit(chrono::system_clock::time_point time, string &&msg) {
        pool.commit(&LogPool::insert, this, time, msg);
    }

private:
    int max_out;
    ThreadPool pool;
    map<chrono::system_clock::time_point, string> logs;
    std::mutex mut;
    std::condition_variable cv;
    atomic<bool> stopped;

    void insert(chrono::system_clock::time_point time, string& msg) {
        mut.lock();
        logs.emplace(time, std::move(msg));
        mut.unlock();
        cv.notify_one();
    }

    void flush() {
        while (!stopped) {
            std::unique_lock<std::mutex> ulk(mut);
            cv.wait(ulk, [&]() -> bool { return !logs.empty() || stopped; });
            int count = 0;
            for (auto it = logs.begin(); it != logs.end() && (count < max_out || stopped); count++) {
                cout << it->second << "\n";
                logs.erase(it++);
            }
            cout << std::flush;
            ulk.unlock();
            this_thread::sleep_for(chrono::milliseconds(10));
        }
    }
};

extern LogPool *logger;
typedef struct{int a;} LogEnd;
#define dendl ((LogEnd) {0})

class LogEntry {
public:
    LogEntry(int level, int line, const char *func) : legal(level <= max_level) {
        if(legal) {
            // time info
            t = chrono::system_clock::now();
            time_t epoch_time = chrono::system_clock::to_time_t(t);
            auto *ptm = std::localtime(&epoch_time);
            char times[40] = {0};
            strftime(times, 40, "%Y-%m-%d %H:%M:%S.", ptm);
            char micros[7] = {0};
            sprintf(micros, "%06d", chrono::duration_cast<chrono::microseconds>(t.time_since_epoch()).count() % 1000000);

            tid = pthread_self();

            // entry header
            if(level == LOG)
                ss << "==LOG== ";
            else if(level == NOTICE)
                ss << "==NOTICE==";
            else if(level == DEBUG)
                ss << "==DEBUG==";
            ss << "[" << times << micros << " " << tid << "]" << " <" << line << " " << func << "> ";
        }
    }
public:
    template<class T>
    LogEntry& operator<<(T &&msg) {
        if(legal)
            ss << msg;
        return *this;
    }

    LogEntry &operator<<(const void *ptr) {
        if(legal)
            ss << hex << ptr << dec;
        return *this;
    }

    void operator<<(LogEnd end) {
        if(legal) {
            logger->submit(t, ss.str());
            ss.clear();
        }
    }

private:
    stringstream ss;
    chrono::system_clock::time_point t;
    pthread_t tid;
    bool legal;
};

#define dout(l) (LogEntry(l, __LINE__, __FUNCTION__))

#endif //FASTGPU_LOG_H