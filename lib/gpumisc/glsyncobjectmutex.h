#ifndef GLSYNCOBJECTMUTEX_H
#define GLSYNCOBJECTMUTEX_H

/**
 * @brief The GlSyncObjectMutex class should provide a mutex mechanism for
 * OpenGL resources.
 */
class GlSyncObjectMutexPrivate;
class GlSyncObjectMutex
{
public:
    GlSyncObjectMutex(bool clientsync=true);
    ~GlSyncObjectMutex();
    GlSyncObjectMutex(const GlSyncObjectMutex&) = delete;
    GlSyncObjectMutex& operator=(const GlSyncObjectMutex&) = delete;

    void lock();
    void unlock();

private:
    GlSyncObjectMutexPrivate* p;

public:
    static void test();
};


// for use as a shared_state_traits::shared_state_mutex
class shared_state_glmutex: public GlSyncObjectMutex {
public:
    void lock_shared() { lock(); }
    bool try_lock() { return false; }
    bool try_lock_shared() { return false; }
    void unlock_shared() { unlock(); }

    bool try_lock_for(...) { lock(); return true; }
    bool try_lock_shared_for(...) { lock_shared(); return true; }
};


#endif // GLSYNCOBJECTMUTEX_H
