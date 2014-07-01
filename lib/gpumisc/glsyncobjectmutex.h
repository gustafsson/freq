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
    GlSyncObjectMutex(bool clientsync=false);
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

#endif // GLSYNCOBJECTMUTEX_H
