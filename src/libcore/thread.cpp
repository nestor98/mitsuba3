#include <mitsuba/core/thread.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/tls.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/fresolver.h>
#include <tbb/task_scheduler_observer.h>
#include <condition_variable>
#include <thread>
#include <sstream>
#include <chrono>

// Required for native thread functions
#if defined(__LINUX__)
#  include <sys/prctl.h>
#elif defined(__OSX__)
#  include <pthread.h>
#elif defined(__WINDOWS__)
#  include <windows.h>
#endif

NAMESPACE_BEGIN(mitsuba)

static ThreadLocal<Thread> *self = nullptr;
static std::atomic<uint32_t> thread_id { 0 };
#if defined(__LINUX__) || defined(__OSX__)
static pthread_key_t this_thread_id;
#elif defined(__WINDOWS__)
static __declspec(thread) int this_thread_id;
#endif

#if defined(_MSC_VER)
namespace {
    // Helper function to set a native thread name. MSDN:
    // http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
    #pragma pack(push, 8)
        struct THREADNAME_INFO {
            DWORD dwType;     // Must be 0x1000.
            LPCSTR szName;    // Pointer to name (in user addr space).
            DWORD dwThreadID; // Thread ID (-1=caller thread).
            DWORD dwFlags;    // Reserved for future use, must be zero.
        };
    #pragma pack(pop)

    void set_thread_name_(const char* thread_name, DWORD thread_id = -1) {
        THREADNAME_INFO info;
        info.dwType     = 0x1000;
        info.szName     = thread_name;
        info.dwThreadID = thread_id;
        info.dwFlags    = 0;
        __try {
            const DWORD MS_VC_EXCEPTION = 0x406D1388;
            RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR),
                           (ULONG_PTR *) &info);
        } __except(EXCEPTION_EXECUTE_HANDLER) { }
    }
} // namespace
#endif // _MSC_VER

/// Dummy class to associate a thread identity with the main thread
class MainThread : public Thread {
public:
    MainThread() : Thread("main") { }

    virtual void run() override { Log(EError, "The main thread is already running!"); }

    MTS_DECLARE_CLASS()
protected:
    virtual ~MainThread() { }
};

/// Dummy class to associate a thread identity with the main thread
class WorkerThread : public Thread {
public:
    WorkerThread() : Thread(tfm::format("wrk%i", idx++)) { }

    virtual void run() override { Log(EError, "The worker thread is already running!"); }

    MTS_DECLARE_CLASS()
protected:
    virtual ~WorkerThread() { }
    static std::atomic<uint32_t> idx;
};

std::atomic<uint32_t> WorkerThread::idx{0};

struct Thread::ThreadPrivate {
    std::thread thread;
    std::thread::native_handle_type native_handle;
    std::string name;
    bool running = false;
    bool tbb_thread = false;
    bool critical = false;
    int core_affinity = -1;
    Thread::EPriority priority;
    ref<Logger> logger;
    ref<Thread> parent;
    ref<FileResolver> fresolver;

    ThreadPrivate(const std::string &name) : name(name) { }
};

Thread::Thread(const std::string &name)
 : d(new ThreadPrivate(name)) { }

Thread::~Thread() {
    if (d->running)
        Log(EWarn, "Destructor called while thread '%s' was still running", d->name);
}

void Thread::set_critical(bool critical) {
    d->critical = critical;
}

bool Thread::is_critical() const {
    return d->critical;
}

const std::string &Thread::name() const {
    return d->name;
}

void Thread::set_name(const std::string &name) {
    d->name = name;
}

void Thread::set_logger(Logger *logger) {
    d->logger = logger;
}

Logger* Thread::logger() {
    return d->logger;
}

void Thread::set_file_resolver(FileResolver *fresolver) {
    d->fresolver = fresolver;
}

FileResolver* Thread::file_resolver() {
    return d->fresolver;
}

const FileResolver* Thread::file_resolver() const {
    return d->fresolver;
}

Thread* Thread::thread() {
    return *self;
}

bool Thread::is_running() const {
    return d->running;
}

Thread* Thread::parent() {
    return d->parent;
}

const Thread* Thread::parent() const {
    return d->parent.get();
}

Thread::EPriority Thread::priority() const {
    return d->priority;
}

int Thread::core_affinity() const {
    return d->core_affinity;
}

uint32_t Thread::id() {
#if defined(__WINDOWS__)
    return this_thread_id;
#elif defined(__OSX__) || defined(__LINUX__)
    /* pthread_self() doesn't provide nice increasing IDs, and syscall(SYS_gettid)
       causes a context switch. Thus, this function uses a thread-local variable
       to provide a nice linearly increasing sequence of thread IDs */
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_getspecific(this_thread_id)));
#endif
}

bool Thread::set_priority(EPriority priority) {
    d->priority = priority;
    if (!d->running)
        return true;

#if defined(__LINUX__) || defined(__OSX__)
    Float factor;
    switch (priority) {
        case EIdlePriority: factor = 0.0f; break;
        case ELowestPriority: factor = 0.2f; break;
        case ELowPriority: factor = 0.4f; break;
        case EHighPriority: factor = 0.6f; break;
        case EHighestPriority: factor = 0.8f; break;
        case ERealtimePriority: factor = 1.0f; break;
        default: factor = 0.0f; break;
    }

    const pthread_t thread_id = d->native_handle;
    struct sched_param param;
    int policy;
    int retval = pthread_getschedparam(thread_id, &policy, &param);
    if (retval) {
        Log(EWarn, "pthread_getschedparam(): %s!", strerror(retval));
        return false;
    }

    int min = sched_get_priority_min(policy);
    int max = sched_get_priority_max(policy);

    if (min == max) {
        Log(EWarn, "Could not adjust the thread priority -- valid range is zero!");
        return false;
    }
    param.sched_priority = (int) (min + (max-min)*factor);

    retval = pthread_setschedparam(thread_id, policy, &param);
    if (retval) {
        Log(EWarn, "Could not adjust the thread priority to %i: %s!",
            param.sched_priority, strerror(retval));
        return false;
    }
#elif defined(__WINDOWS__)
    int win32_priority;
    switch (priority) {
        case EIdlePriority:     win32_priority = THREAD_PRIORITY_IDLE; break;
        case ELowestPriority:   win32_priority = THREAD_PRIORITY_LOWEST; break;
        case ELowPriority:      win32_priority = THREAD_PRIORITY_BELOW_NORMAL; break;
        case EHighPriority:     win32_priority = THREAD_PRIORITY_ABOVE_NORMAL; break;
        case EHighestPriority:  win32_priority = THREAD_PRIORITY_HIGHEST; break;
        case ERealtimePriority: win32_priority = THREAD_PRIORITY_TIME_CRITICAL; break;
        default:                win32_priority = THREAD_PRIORITY_NORMAL; break;
    }

    // If the function succeeds, the return value is nonzero
    const HANDLE handle = d->native_handle;
    if (SetThreadPriority(handle, win32_priority) == 0) {
        Log(EWarn, "Could not adjust the thread priority to %i: %s!",
            win32_priority, util::last_error());
        return false;
    }
#endif
    return true;
}

void Thread::set_core_affinity(int coreID) {
    d->core_affinity = coreID;
    if (!d->running)
        return;

#if defined(__OSX__)
    /* CPU affinity not supported on OSX */
#elif defined(__LINUX__)
    int core_count = sysconf(_SC_NPROCESSORS_CONF),
        logical_core_count = core_count;

    size_t size = 0;
    cpu_set_t *cpuset = nullptr;
    int retval = 0;

    /* The kernel may expect a larger cpu_set_t than would be warranted by the
       physical core count. Keep querying with increasingly larger buffers if
       the pthread_getaffinity_np operation fails */
    for (int i = 0; i<10; ++i) {
        size = CPU_ALLOC_SIZE(logical_core_count);
        cpuset = CPU_ALLOC(logical_core_count);
        if (!cpuset) {
            Log(EWarn, "Thread::set_core_affinity(): could not allocate cpu_set_t");
            return;
        }

        CPU_ZERO_S(size, cpuset);

        int retval = pthread_getaffinity_np(d->native_handle, size, cpuset);
        if (retval == 0)
            break;

        /* Something went wrong -- release memory */
        CPU_FREE(cpuset);

        if (retval == EINVAL) {
            /* Retry with a larger cpuset */
            logical_core_count *= 2;
        } else {
            break;
        }
    }

    if (retval) {
        Log(EWarn, "Thread::set_core_affinity(): pthread_getaffinity_np(): could "
            "not read thread affinity map: %s", strerror(retval));
        return;
    }

    int actual_core_id = -1, available = 0;
    for (int i=0; i<logical_core_count; ++i) {
        if (!CPU_ISSET_S(i, size, cpuset))
            continue;
        if (available++ == coreID) {
            actual_core_id = i;
            break;
        }
    }

    if (actual_core_id == -1) {
        Log(EWarn, "Thread::set_core_affinity(): out of bounds: %i/%i cores "
                   "available, requested #%i!",
            available, core_count, coreID);
        CPU_FREE(cpuset);
        return;
    }

    CPU_ZERO_S(size, cpuset);
    CPU_SET_S(actual_core_id, size, cpuset);

    retval = pthread_setaffinity_np(d->native_handle, size, cpuset);
    if (retval) {
        Log(EWarn,
            "Thread::set_core_affinity(): pthread_setaffinity_np: failed: %s",
            strerror(retval));
        CPU_FREE(cpuset);
        return;
    }

    CPU_FREE(cpuset);
#elif defined(__WINDOWS__)
    int core_count = util::core_count();
    const HANDLE handle = d->native_handle;

    DWORD_PTR mask;

    if (coreID != -1 && coreID < core_count)
        mask = (DWORD_PTR) 1 << coreID;
    else
        mask = (1 << core_count) - 1;

    if (!SetThreadAffinityMask(handle, mask))
        Log(EWarn, "Thread::set_core_affinity(): SetThreadAffinityMask : failed");
#endif
}

void Thread::start() {
    if (d->running)
        Log(EError, "Thread is already running!");
    if (!self)
        Log(EError, "Threading has not been initialized!");

    Log(EDebug, "Spawning thread \"%s\"", d->name);

    d->parent = Thread::thread();

    /* Inherit the parent thread's logger if none was set */
    if (!d->logger)
        d->logger = d->parent->logger();

    /* Inherit the parent thread's file resolver if none was set */
    if (!d->fresolver)
        d->fresolver = d->parent->file_resolver();

    d->running = true;

    inc_ref();
    d->thread = std::thread(&Thread::dispatch, this);
}

void Thread::dispatch() {
    d->native_handle = d->thread.native_handle();

    ThreadLocalBase::register_thread();

    uint32_t id = thread_id++;
    #if defined(__LINUX__) || defined(__OSX__)
        pthread_setspecific(this_thread_id, reinterpret_cast<void *>(id));
    #elif defined(__WINDOWS__)
        this_thread_id = id;
    #endif

    *self = this;

    if (d->priority != ENormalPriority)
        set_priority(d->priority);

    if (!d->name.empty()) {
        const std::string thread_name = "Mitsuba: " + name();
        #if defined(__LINUX__)
            pthread_setname_np(pthread_self(), thread_name.c_str());
        #elif defined(__OSX__)
            pthread_setname_np(thread_name.c_str());
        #elif defined(__WINDOWS__)
            set_thread_name_(thread_name.c_str());
        #endif
    }

    if (d->core_affinity != -1)
        set_core_affinity(d->core_affinity);

    try {
        run();
    } catch (std::exception &e) {
        ELogLevel warnLogLevel = logger()->error_level() == EError ? EWarn : EInfo;
        Log(warnLogLevel, "Fatal error: uncaught exception: \"%s\"", e.what());
        if (d->critical)
            abort();
    }

    exit();

}

void Thread::exit() {
    Log(EDebug, "Thread \"%s\" has finished", d->name);
    d->running = false;
    Assert(*self == this);
    ThreadLocalBase::unregister_thread();
    dec_ref();
}

void Thread::join() {
    d->thread.join();
}

void Thread::detach() {
    d->thread.detach();
}

void Thread::sleep(uint32_t ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void Thread::yield() {
    std::this_thread::yield();
}

std::string Thread::to_string() const {
    std::ostringstream oss;
    oss << "Thread[" << std::endl
        << "  name = \"" << d->name << "\"," << std::endl
        << "  running = " << d->running << "," << std::endl
        << "  priority = " << d->priority << "," << std::endl
        << "  critical = " << d->critical << std::endl
        << "]";
    return oss.str();
}

class Thread::TaskObserver : public tbb::task_scheduler_observer {
public:
    TaskObserver() {
        observe();
    }

    void on_scheduler_entry(bool) {
        if (ThreadLocalBase::register_thread()) {
            uint32_t id = thread_id++;
            WorkerThread *thr = new WorkerThread();
            #if defined(__LINUX__) || defined(__OSX__)
                thr->d->native_handle = pthread_self();
                pthread_setspecific(this_thread_id, reinterpret_cast<void *>(id));
            #elif defined(__WINDOWS__)
                thr->d->native_handle = GetCurrentThread();
                this_thread_id = id;
            #endif
            thr->d->running = true;
            thr->d->tbb_thread = true;
            *self = thr;

            uint32_t worker_id;
            /* critical section */ {
                std::unique_lock<std::mutex> lock(m_mutex);
                worker_id = m_started_counter++;
            }

            const std::string thread_name = tfm::format("tbb_%03i", worker_id);
            #if defined(__LINUX__)
                pthread_setname_np(pthread_self(), thread_name.c_str());
            #elif defined(__OSX__)
                pthread_setname_np(thread_name.c_str());
            #elif defined(__WINDOWS__)
                set_thread_name_(thread_name.c_str());
            #endif

            thr->set_core_affinity(worker_id);
        }
    }

    void on_scheduler_exit(bool) {
        Thread *thr = *self;
        if (!thr || !thr->d->tbb_thread)
            return;
        thr->d->running = false;
        ThreadLocalBase::unregister_thread();
        /* critical section */ {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_stopped_counter++;
            m_cv.notify_all();
        }
    }

    void wait() {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_started_counter != m_stopped_counter)
            m_cv.wait(lock);
    }
private:
    uint32_t m_started_counter{0};
    uint32_t m_stopped_counter{0};
    std::condition_variable m_cv;
    std::mutex m_mutex;
};

static std::unique_ptr<Thread::TaskObserver> observer;

void Thread::static_initialization() {
    #if defined(__LINUX__) || defined(__OSX__)
        pthread_key_create(&this_thread_id, nullptr);
    #endif
    ThreadLocalBase::static_initialization();
    ThreadLocalBase::register_thread();

    self = new ThreadLocal<Thread>();
    Thread *mainThread = new MainThread();
    mainThread->d->running = true;
    mainThread->d->fresolver = new FileResolver();
    *self = mainThread;

    observer = std::unique_ptr<Thread::TaskObserver>(
        new Thread::TaskObserver());
}

void Thread::static_shutdown() {
    observer->wait();
    observer.reset();
    thread()->d->running = false;
    ThreadLocalBase::unregister_thread();
    delete self;
    self = nullptr;
    ThreadLocalBase::static_shutdown();

    #if defined(__LINUX__) || defined(__OSX__)
        pthread_key_delete(this_thread_id);
    #endif
}

ThreadEnvironment::ThreadEnvironment(Thread *other) {
    auto thread = Thread::thread();
    Assert(thread);
    m_logger = thread->logger();
    m_file_resolver = thread->file_resolver();
    thread->set_logger(other->logger());
    thread->set_file_resolver(other->file_resolver());
}

ThreadEnvironment::~ThreadEnvironment() {
    auto thread = Thread::thread();
    thread->set_logger(m_logger);
    thread->set_file_resolver(m_file_resolver);
}

MTS_IMPLEMENT_CLASS(Thread, Object)
MTS_IMPLEMENT_CLASS(MainThread, Thread)
MTS_IMPLEMENT_CLASS(WorkerThread, Thread)
NAMESPACE_END(mitsuba)
