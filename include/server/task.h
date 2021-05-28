//
// Created by root on 2021/5/27.
//

#ifndef FASTGPU_TASK_H
#define FASTGPU_TASK_H

#include "server.h"
#include "commands.h"
#include "device.h"
#include "mod.h"
#include <list>
#include <memory>
#include <mutex>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

namespace mgpu {
    struct TASK_KEY {
        uint key; // pid << 16 + device
        stream_t stream;

        bool operator==(const TASK_KEY &key1) const {
            return key1.key == key && key1.stream == stream;
        }
    };
}

namespace std {
    using mgpu::TASK_KEY;

    template<>
    struct hash<TASK_KEY> {
        std::size_t operator()(const TASK_KEY &key) const {
            std::size_t h1 = hash<decltype(key.key)>{}(key.key);
            std::size_t h2 = hash<decltype(key.stream)>{}(key.stream);
            return h1 ^ (h2 << 1);
        }
    };
}

namespace mgpu {
    using namespace std::chrono;
    class Task : public Module {
    public:
        Task() {
            joinable = false;
        }

    public:
        void init() override {}
        void run() override {}

    public:
        typedef std::list<std::shared_ptr<Command>> TASK_LIST;
        typedef std::unordered_map<TASK_KEY, std::shared_ptr<Command>> Jobs;

        Jobs fetch() {
            Jobs res;
            mlocks.lock();
            res.swap(pending);
            for (auto &item : tasks) {
                auto& key = item.first;
                auto& list = item.second;
                // list's first task already done
                if(finished.count(key) && finished[key])
                    finished.erase(key);
                // list has task pending
                if(res.count(key))
                    continue;
                // list not blocked not pending
                if(!list.empty()) {
                    res[key] = list.front();
                    list.erase(list.begin());
                }
            }
            mlocks.unlock();
            return res;
        }

        void register_doing(TASK_KEY key, const std::shared_ptr<Command> &cmd) {
            if (finished.count(key))
                std::cout << " bug " << std::endl;
            finished[key] = cmd->get_status();
        }

        void put_back(Jobs &jobs) {
            pending.swap(jobs);
        }

        void insert_cmd(const std::shared_ptr<Command> &cmd) {
            pid_t pid = cmd->get_pid();
            TASK_KEY key{cmd->get_type(), cmd->get_stream()};
            mlocks.lock();
            if (llocks.count(key)) {
                // has list
                std::lock_guard<std::mutex> lk(*llocks[key]);
                mlocks.unlock();
                tasks[key].push_back(cmd);
                return;
            } else {
                // has no list
                if(pid_keys.count(pid) == 0) {
                    // insert pid
                    pid_keys[pid] = unordered_set<TASK_KEY>();
                    tasks[key] = TASK_LIST{};
                    llocks[key] = make_shared<std::mutex>();
                }
                // insert list
                std::lock_guard<std::mutex> lk(*llocks[key]);
                mlocks.unlock();
                tasks[key].push_back(cmd);
            }
        }

        stream_t *create_streams(const CudaStreamCreateMsg *msg) {
            if (msg->num >= MAX_STREAMS)
                return nullptr;
            auto res = new stream_t[msg->num];
            int size = 0;
            vector<stream_t> unused;
            auto streams = DEVICES->getStream(msg->key & 0xffff);
            std::lock_guard<std::mutex> lk(mlocks);
            for (auto s : streams) {
                if (llocks.count(TASK_KEY{msg->key, s})) {
                    unused.push_back(s);
                } else {
                    res[size++] = s;
                }
            }
            for (auto iter = unused.begin(); size < msg->num; iter++) {
                res[size] = *iter;
            }
            return res;
        }

        void clear_pid(const pid_t pid) {
            mlocks.lock();
            if(pid_keys.count(pid) == 0)
                return;
            for(auto& key : pid_keys[pid])
            {
                auto tmp = llocks[key];
                tmp->lock();
                llocks.erase(key);
                pending.erase(key);
                tasks.erase(key);
                tmp->unlock();
            }
            mlocks.unlock();
        }

    private:
        std::unordered_map<TASK_KEY, TASK_LIST> tasks;
        std::mutex mlocks; // map lock
        std::unordered_map<TASK_KEY, std::shared_ptr<std::mutex>> llocks; // list lock
        std::unordered_map<TASK_KEY, std::shared_ptr<bool>> finished;
        std::unordered_map<uint, unordered_set<TASK_KEY>> pid_keys; // pid - stream map
        Jobs pending;
    };
}

#endif //FASTGPU_TASK_H
