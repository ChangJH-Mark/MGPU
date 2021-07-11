//
// Created by root on 2021/5/18.
//

#ifndef FASTGPU_THREADPOOL_H
#define FASTGPU_THREADPOOL_H
#include <atomic>
#include <functional>
#include <future>
#include <queue>

class ThreadPool {
public:
    ThreadPool(uint init_num, uint max_num);
    void stop() {
        stopped = true;
        cv.notify_all();
        for(auto & w : workers) {
            if(w.joinable())
                w.join();
        }
    }

    template<class F, typename... Args>
    auto commit(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    void worker();
    void addWorker() {
        workers.emplace_back(&ThreadPool::worker, this);
        idlThrNums++;
    }

    typedef std::function<void()> Task;

    uint max_num;  // max workers that can stay

    std::mutex queue_lock;
    std::queue<Task> tasks;
    std::condition_variable cv;

    std::vector<std::thread> workers;
    std::atomic<uint> idlThrNums = 0;

    std::atomic<bool> stopped;
};

template<class F, typename... Args>
auto ThreadPool::commit(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
    using RetType = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<RetType()>> (
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );
    std::future<RetType> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_lock);
        tasks.emplace([task] {(*task)();});
        if(idlThrNums == 0 && workers.size() < max_num && !stopped)
            addWorker();
    }
    cv.notify_one();
    return res;
}

#endif //FASTGPU_THREADPOOL_H