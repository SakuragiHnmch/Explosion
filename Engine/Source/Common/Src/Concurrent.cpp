//
// Created by johnk on 2024/4/14.
//

#include <Common/Concurrent.h>

namespace Common {
    NamedThread::NamedThread() = default;

    NamedThread::NamedThread(NamedThread&& other) noexcept
        : thread(std::move(other.thread))
    {
    }

    NamedThread::~NamedThread() = default;

    NamedThread& NamedThread::operator=(NamedThread&& other) noexcept
    {
        thread = std::move(other.thread);
        return *this;
    }

    void NamedThread::Join()
    {
        thread.join();
    }

    void NamedThread::SetThreadName(const std::string& name)
    {
#if PLATFORM_WINDOWS
        SetThreadDescription(thread.native_handle(), Common::StringUtils::ToWideString(name).c_str());
#elif PLATFORM_MACOS
        pthread_setname_np(name.c_str());
#else
        pthread_setname_np(thread.native_handle(), name.c_str());
#endif
    }

    ThreadPool::ThreadPool(const std::string& name, uint8_t threadNum)
        : stop(false)
    {
        threads.reserve(threadNum);
        for (auto i = 0; i < threadNum; i++) {
            std::string fullName = name + "-" + std::to_string(i);
            threads.emplace_back(NamedThread(fullName, [this]() -> void {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        condition.wait(lock, [this]() -> bool { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) {
                            return;
                        }
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            }));
        }
    }

    ThreadPool::~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& thread : threads) {
            thread.Join();
        }
    }

    WorkerThread::WorkerThread(const std::string& name)
        : stop(false), flush(false)
    {
        thread = NamedThread(name, [this]() -> void {
            while (true) {
                bool needNotifyMainThread = false;
                std::vector<std::function<void()>> tasksToExecute;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    taskCondition.wait(lock, [this]() -> bool { return stop || flush || !tasks.empty(); });
                    if (stop && tasks.empty()) {
                        return;
                    }
                    if (flush) {
                        tasksToExecute.reserve(tasks.size());
                        while (!tasks.empty()) {
                            tasksToExecute.emplace_back(std::move(tasks.front()));
                            tasks.pop();
                        }
                        flush = false;
                        needNotifyMainThread = true;
                    } else {
                        tasksToExecute.emplace_back(std::move(tasks.front()));
                        tasks.pop();
                    }
                }
                for (auto& task : tasksToExecute) {
                    task();
                }
                if (needNotifyMainThread) {
                    flushCondition.notify_one();
                }
            }
        });
    }

    WorkerThread::~WorkerThread()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }
        taskCondition.notify_all();
        thread.Join();
    }

    void WorkerThread::Flush()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            flush = true;
        }
        taskCondition.notify_one();
        {
            std::unique_lock<std::mutex> lock(mutex);
            flushCondition.wait(lock);
        }
    }
}
