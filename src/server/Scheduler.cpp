//
// Created by root on 2021/3/17.
//

#include "server/scheduler.h"
#include "common/helper.h"

using namespace mgpu;

void Scheduler::init() {
    cudaCheck(cudaStreamCreateWithFlags(&ctrl, cudaStreamNonBlocking));
    s = std::move(thread(&Scheduler::sched, this));
    auto handle = s.native_handle();
    pthread_setname_np(handle, "Scheduler");
}

void Scheduler::destroy() {
    dout(LOG) << "start destroy Scheduler Module" << dendl;
    stopped = true;
    signal.notify_one();
    if (s.joinable())
        s.join();
}

void Scheduler::join() {
    if (s.joinable())
        s.join();
}

void Scheduler::apply_slot(KernelInstance *k) {
    while(!spin.try_lock());
    pending.push_back(k);
    spin.unlock();
    signal.notify_one();

    // wait till one
    while (slot[0] != k && slot[1] != k);
}

void Scheduler::release_slot(KernelInstance *k) {
    k->finished = true;

    int index = -1;
    if (k == slot[0])
        index = 0;
    else if (k == slot[1])
        index = 1;
    else {
        cout << "release a non-exist Kernel" << endl;
        exit(EXIT_FAILURE);
    }
    run_stat.lock();
    slot[index] = nullptr;
    run_cnt--;
    if (run_cnt == 1) {
        auto another = slot[1 - index].load();
        another->occupancy_all(ctrl);   // this kernel might stop running
    }
    run_stat.unlock();
    signal.notify_one();
}

void Scheduler::sched() {
    // schedule strategy
    while (!stopped) {
        std::unique_lock<std::mutex> ul(spin);
        signal.wait(ul, [this]() -> bool {
            return stopped || !pending.empty();
        });
        // stop running
        if (stopped)
            return;
        std::lock_guard<std::mutex> lg(run_stat);
        if(run_cnt == 2)
            continue;

        if (run_cnt != 1 && run_cnt != 0) {
            cout << "wrong run_cnt " << run_cnt << " exception" << endl;
            exit(EXIT_FAILURE);
        }
        KernelInstance *candidate = nullptr;
        KernelInstance *running = nullptr;
        int id = 0;
        if (run_cnt == 0) {
            running = pending.front();
            pending.pop_front();
            slot[id] = running;
            run_cnt++;
        } else {
            id = (slot[0] == nullptr) ? 1 : 0;
            running = slot[id];
        }

        if (find_corun(running, &candidate)) {
            slot[1 - id] = candidate;
            run_cnt++;
        }
    } // while
}

bool Scheduler::find_corun(KernelInstance *running, KernelInstance **candidate) {
    std::pair<int, int> res(-1, -1);
    double best_score = 0.0f, low_bound, high_bound;
    int best_warps = -1;
    auto best_it = pending.end();
    auto &gpu = running->gpu;

    low_bound = 0.9 * 2 * running->gpu->gmem_max_tp;
    high_bound = 1.1 * 2 * running->gpu->gmem_max_tp;

    int r_thds = (running->block.x * running->block.y * running->block.z), c_thds = 0;
    int r_warps = (r_thds + gpu->warp_size - 1) / gpu->warp_size, c_warps = 0;
    int r_regs = running->prop.regs * r_thds, c_regs = 0;
    int r_shms = running->prop.shms, c_shms = 0;
    int r_prop = running->prop.property, c_prop = 0;

    for (auto c = pending.begin(); c != pending.end(); c++) {
        if(running->is_finished())
            goto single;

        if (running->prop.property >= high_bound && (*c)->prop.property >= high_bound)
            continue;
        else if (running->prop.property <= low_bound && (*c)->prop.property <= low_bound)
            continue;

        c_thds = (*c)->block.x * (*c)->block.y * (*c)->block.z;
        c_warps = (c_thds + gpu->warp_size - 1) / gpu->warp_size;
        c_regs = c_thds * (*c)->prop.regs;
        c_shms = (*c)->prop.shms;
        c_prop = (*c)->prop.property;

        // find every possible combination
        for (int i = 1; i < running->max_block_per_sm; i++) {
            if(running->is_finished())
                goto single;

            int j = gpu->max_blocks - i;
            j = min(j, (int) (gpu->max_warps - i * r_warps) / c_warps);                       /* warps limit */
            j = min(j, (int) (gpu->regs - i * r_regs) / c_regs);                              /* registers limit */
            j = min(j, (int) (gpu->share_mem - i * r_shms) / c_shms);                         /* share memory limit */

            unsigned total_warps = i * r_warps + j * c_warps;
            double score = (i * r_warps * r_prop + j * c_warps * c_prop) * 1.0 / (total_warps);
            if (score <= low_bound || score >= high_bound)
                continue;

            // total_warps less than best_warps ever
            if (total_warps < best_warps)
                continue;
            // total_warps equal best_warps ever, but get worse score
            if (total_warps == best_warps &&
                (abs(score - 2 * gpu->gmem_max_tp) >= abs(best_score - 2 * gpu->gmem_max_tp)))
                continue;

            // new best result
            best_warps = total_warps;
            best_score = score;
            best_it = c;
            res = make_pair(i, j);
        } // loop for config
    } // for each pending

    single:
    if(running->is_finished()) {
        *candidate = pending.front();
        pending.pop_front();
        return true;
    }

    // not corun
    if (res == std::pair<int, int>(-1, -1))
        return false;

    // corun
    *candidate = *best_it;
    running->set_config(0, gpu->sms, res.first, ctrl);
    (*best_it)->set_config(0, gpu->sms, res.second, ctrl);
    pending.erase(best_it);
    return true;
}