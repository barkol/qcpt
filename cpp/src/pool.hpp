// One thread pool for the whole process. Every member of a race submits its parallel loops here, so the machine is never oversubscribed;
// the submitting thread works on its own job too, which makes nested and concurrent use safe. A thread-local cancel flag stops the remaining tasks of a job.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "env.hpp"

namespace qcpt {

inline std::atomic<bool>*& current_cancel() { thread_local std::atomic<bool>* c = nullptr; return c; }   // set by the member thread that wants to be cancellable
inline double& current_weight() { thread_local double w = 1.0; return w; }                      // share of the pool for the jobs this thread submits
inline bool& current_low() { thread_local bool l = false; return l; }                         // jobs of this thread run only on workers that no other job needs
inline bool cancelled() { auto* c = current_cancel(); return c && c->load(std::memory_order_relaxed); }

class ThreadPool {
public:
    static ThreadPool& get() { static ThreadPool pool(std::max(1u, std::thread::hardware_concurrency())); return pool; }

    void run(size_t n, const std::function<void(size_t)>& f) {
        if (n == 0) return;
        auto job = std::make_shared<Job>(); job->f = &f; job->n = n; job->cancel = current_cancel(); job->weight = current_weight(); job->low = current_low();
        if (n > 1 && !workers_.empty()) { { std::lock_guard<std::mutex> l(m_); jobs_.push_back(job); } cv_.notify_all(); }
        work(*job);                                                        // the caller takes tasks too
        std::unique_lock<std::mutex> l(job->m); job->cv.wait(l, [&] { return job->done.load() == job->n; });
    }
    ~ThreadPool() { { std::lock_guard<std::mutex> l(m_); stop_ = true; } cv_.notify_all(); for (auto& t : workers_) t.join(); }

private:
    struct Job { const std::function<void(size_t)>* f = nullptr; size_t n = 0; std::atomic<size_t> next{0}, done{0}, served{0}; double weight = 1.0; bool low = false; std::atomic<bool>* cancel = nullptr; std::mutex m; std::condition_variable cv; };

    explicit ThreadPool(unsigned nt) { for (unsigned i = 1; i < nt; ++i) workers_.emplace_back([this] { loop(); }); }   // the submitting thread is the n-th worker

    static void exec(Job& j, size_t k) {
        auto*& cc = current_cancel(); auto* prev = cc; cc = j.cancel;              // inside a task, cancelled() reports the flag of the job's owner (the task may run on a pool thread)
        if (!(j.cancel && j.cancel->load(std::memory_order_relaxed))) (*j.f)(k);
        cc = prev;
        if (j.done.fetch_add(1) + 1 == j.n) { std::lock_guard<std::mutex> l(j.m); j.cv.notify_all(); }
    }
    static void work(Job& j) { for (size_t k; (k = j.next.fetch_add(1)) < j.n;) exec(j, k); }
    // workers take one task at a time from the jobs in turn, so a small job (a template search) is not stuck behind a large one (a beam level)
    void loop() {
        for (;;) {
            std::shared_ptr<Job> job; size_t k = 0;
            { std::unique_lock<std::mutex> l(m_);
              for (;;) {
                  cv_.wait(l, [&] { return stop_ || !jobs_.empty(); }); if (stop_) return;
                  bool got = false;
                  while (!jobs_.empty()) {
                      // the job with the smallest (tasks served + 1) / weight goes next; exhausted jobs are dropped
                      constexpr size_t npos = (size_t)-1; size_t bi = npos; double bs = 1e300;
                      for (size_t i = 0; i < jobs_.size();) {
                          Job& j = *jobs_[i]; if (j.next.load() >= j.n) { jobs_.erase(jobs_.begin() + i); continue; }
                          const double sc = ((double)j.served.load() + 1.0) / j.weight + (j.low ? 1e12 : 0.0); if (sc < bs) { bs = sc; bi = i; } ++i;
                      }
                      if (bi == npos) break;
                      const size_t kk = jobs_[bi]->next.fetch_add(1);
                      if (kk < jobs_[bi]->n) { job = jobs_[bi]; k = kk; job->served.fetch_add(1); got = true; break; }
                  }
                  if (got) break;
              } }
            exec(*job, k);
        }
    }

    std::vector<std::thread> workers_; std::deque<std::shared_ptr<Job>> jobs_; std::mutex m_; std::condition_variable cv_; bool stop_ = false; size_t rr_ = 0;
};

// parallel loop over tasks 0..ntasks-1 on the shared pool; `nthreads <= 1` runs inline (used for tiny problems and in tests)
template <class F> void parallel_for(int nthreads, size_t ntasks, F&& f) {
    if (nthreads <= 1 || ntasks <= 1) { for (size_t k = 0; k < ntasks && !cancelled(); ++k) f(k); return; }
    ThreadPool::get().run(ntasks, std::function<void(size_t)>(std::forward<F>(f)));
}

}  // namespace qcpt
