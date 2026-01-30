module;
#include <vector>
#include <queue>
#include <thread>
#include <future>
#include <functional>
#include <condition_variable>
#include <atomic>
#include <optional>

export module threadpool;


export class ThreadPool {
public:
	explicit ThreadPool(size_t therad_count = std::thread::hardware_concurrency());
	~ThreadPool();

	template<typename F, class...Args>
	auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

private:
	std::vector<std::jthread> m_workers;
	std::queue<std::function<void()>> m_tasks;
	std::mutex m_mutex;
	std::condition_variable m_condition;
	std::atomic<bool> m_stop{ false };
};

ThreadPool::ThreadPool(const size_t thread_count) {
	m_workers.reserve(thread_count);
	for (size_t i{ 0 }; i < thread_count; ++i) {
		m_workers.emplace_back([this](const std::stop_token&  st) {
			while (!st.stop_requested()) {
				std::function<void()> task;
				{
					std::unique_lock lock(this->m_mutex);
					this->m_condition.wait(lock, [this] {
						return this->m_stop.load() || !this->m_tasks.empty();
						});
					if (this->m_stop.load() && this->m_tasks.empty())
						return;
					task = std::move(this->m_tasks.front());
					this->m_tasks.pop();
				}
				if (task) task();
			}
		});
	}
}

ThreadPool::~ThreadPool() {
	{
		std::unique_lock lock(m_mutex);
		m_stop.store(true);
	}
	m_condition.notify_all();
}

//template<typename F, class...Args>
//auto ThreadPool::submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
//	using return_type = std::invoke_result_t<F, Args...>;
//	auto task = std::make_shared<std::packaged_task<return_type()>>(
//		std::bind(std::forward<F>(f), std::forward<Args>(args)...)
//	);
//	std::future<return_type> res = task->get_future();
//	{
//		std::unique_lock lock(m_mutex);
//		if (m_stop.load()) {
//			throw std::runtime_error("错误");
//		}
//		m_tasks.emplace([task]() { (*task)(); });
//	}
//	m_condition.notify_one();
//	return res;
//}

template<typename F, class... Args>
auto ThreadPool::submit(F&& f, Args&&... args)
-> std::future<std::invoke_result_t<F, Args...>>
{
	using return_type = std::invoke_result_t<F, Args...>;
	if constexpr (std::is_void_v<return_type>) {
		auto task = std::make_shared<std::packaged_task<void()>>(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
		);
		std::future<void> res = task->get_future();
		{
			std::unique_lock lock(m_mutex);
			if (m_stop.load()) {
				throw std::runtime_error("线程池已停止");
			}
			m_tasks.emplace([task]() { (*task)(); });
		}
		m_condition.notify_one();
		return res;
	}
	else {
		auto task = std::make_shared<std::packaged_task<return_type()>>(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
		);
		std::future<return_type> res = task->get_future();
		{
			std::unique_lock lock(m_mutex);
			if (m_stop.load()) {
				throw std::runtime_error("线程池已停止");
			}
			m_tasks.emplace([task]() { (*task)(); });
		}
		m_condition.notify_one();
		return res;
	}
}