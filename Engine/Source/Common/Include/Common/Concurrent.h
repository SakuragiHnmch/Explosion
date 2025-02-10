//
// Created by johnk on 2022/7/20.
//

#pragma once

#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <future>
#include <functional>
#include <type_traits>

#include <Common/Debug.h>
#include <Common/Memory.h>
#include <Common/Utility.h>

namespace Common {
    class NamedThread {
    public:
        DefaultMovable(NamedThread)

        NamedThread();

        template <typename F, typename... Args>
        explicit NamedThread(const std::string& name, F&& task, Args&&... args);

        void Join();

    private:
        void SetThreadName(const std::string& name);

        std::thread thread;
    };

    class ThreadPool {
    public:
        ThreadPool(const std::string& name, uint8_t threadNum);
        ~ThreadPool();

        template <typename F, typename... Args> auto EmplaceTask(F&& task, Args&&... args);
        template <typename F, typename... Args> void ExecuteTasks(size_t taskNum, F&& task, Args&&... args);

    private:
        template <typename Ret> auto EmplaceTaskInternal(Common::SharedRef<std::packaged_task<Ret()>> packedTask);

        bool stop;
        std::mutex mutex;
        std::condition_variable condition;
        std::vector<NamedThread> threads;
        std::queue<std::function<void()>> tasks;
    };

    class WorkerThread {
    public:
        explicit WorkerThread(const std::string& name);
        ~WorkerThread();

        void Flush();

        template <typename F, typename... Args>
        auto EmplaceTask(F&& task, Args&&... args);

    private:
        bool stop;
        bool flush;
        std::mutex mutex;
        std::condition_variable taskCondition;
        std::condition_variable flushCondition;
        NamedThread thread;
        std::queue<std::function<void()>> tasks;
    };
}

namespace Common {
    template <typename F, typename... Args>
    NamedThread::NamedThread(const std::string& name, F&& task, Args&& ... args)
    {
        thread = std::thread([this, task = std::forward<F>(task), name](Args&&... args) -> void {
            SetThreadName(name);
            task(args...);
        }, std::forward<Args>(args)...);
    }

    template <typename F, typename... Args>
    auto ThreadPool::EmplaceTask(F&& task, Args&&... args)
    {
        using RetType = std::invoke_result_t<F, Args...>;
        return EmplaceTaskInternal<RetType>(Common::MakeShared<std::packaged_task<RetType()>>(std::bind(std::forward<F>(task), std::forward<Args>(args)...)));
    }

    template <typename F, typename... Args>
    void ThreadPool::ExecuteTasks(size_t taskNum, F&& task, Args&&... args)
    {
        using RetType = std::invoke_result_t<F, size_t, Args...>;

        std::vector<std::future<RetType>> futures;
        futures.reserve(taskNum);
        for (size_t i = 0; i < taskNum; i++) {
            futures.emplace_back(EmplaceTaskInternal<RetType>(Common::MakeShared<std::packaged_task<RetType()>>(std::bind(std::forward<F>(task), i, std::forward<Args>(args)...))));
        }

        for (const auto& future : futures) {
            future.wait();
        }
    }

    template <typename Ret>
    auto ThreadPool::EmplaceTaskInternal(Common::SharedRef<std::packaged_task<Ret()>> packedTask)
    {
        auto result = packedTask->get_future();
        {
            std::unique_lock lock(mutex);
            Assert(!stop);
            tasks.emplace([packedTask]() -> void { (*packedTask)(); });
        }
        condition.notify_one();
        return result;
    }

    template <typename F, typename... Args>
    auto WorkerThread::EmplaceTask(F&& task, Args&& ... args)
    {
        using RetType = std::invoke_result_t<F, Args...>;
        auto packagedTask = Common::MakeShared<std::packaged_task<RetType()>>(std::bind(std::forward<F>(task), std::forward<Args>(args)...));
        auto result = packagedTask->get_future();
        {
            std::unique_lock lock(mutex);
            Assert(!stop);
            tasks.emplace([packagedTask]() -> void { (*packagedTask)(); });
        }
        taskCondition.notify_one();
        return result;
    }
}
