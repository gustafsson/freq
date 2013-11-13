#ifndef VOLATILELOCK_H
#define VOLATILELOCK_H

#include "backtrace.h"
#include "unused.h"

#include <QReadWriteLock>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>


class LockFailed: public virtual boost::exception, public virtual std::exception {
public:
    typedef boost::error_info<struct timeout, int> timeout_value;
};


#ifdef _DEBUG
// disable timeouts during debug sessions
//#define VolatilePtr_lock_timeout_ms -1
#define VolatilePtr_lock_timeout_ms 1000
#else
#define VolatilePtr_lock_timeout_ms 1000
#endif

/**
 * The VolatilePtr class guarantees compile-time thread safe access to objects.
 *
 * For examples of usage see
 * VolatilePtrTest::test ()
 *
 * To use VolatilePtr to manage thread safe access to some previously
 * un-protected data you first need to create a new class representing the
 * state which needs to be managed. If this is just a single existing class
 * you can either 1) change the existing class to make it inherit
 * VolatilePtr<ClassType>, or 2) create a wrapper class.
 *
 * The idea is to use a pointer to a volatile object when juggling references to
 * objects. From a volatile object you can only access methods that are volatile
 * (just like only const methods are accessible from a const object). Using the
 * volatile classifier blocks access to use any "regular" (non-volatile) methods.
 *
 * The helper classes ReadPtr and WritePtr uses RAII for thread safe access to
 * a non-volatile reference to the object.
 *
 * The examples listed might make more sense than this description;
 * see VolatilePtrTest::test ()
 *
 * The time overhead of locking an available object is less than 1e-7 s.
 *
 * Author: johan.gustafsson@muchdifferent.com
 */
template<typename T>
class VolatilePtr: boost::noncopyable
{
protected:

    // Need to be instantiated as a subclass.
    VolatilePtr ()
        :
          // NonRecursive is default, but emphasis here that it is wanted
          // because recursive locking could be a sign of an unnecessarily
          // convoluted algorithm.
          lock_ (QReadWriteLock::NonRecursive)
    {}


public:

    typedef boost::shared_ptr<volatile T> Ptr;
    typedef boost::shared_ptr<const volatile T> ConstPtr;
    typedef boost::weak_ptr<volatile T> WeakPtr;

    class LockFailed: public ::LockFailed {};

    ~VolatilePtr () {
        UNUSED(VolatilePtr* p) = (T*)0; // T is required to be a subtype of VolatilePtr
    }


    /**
     * For examples of usage see void VolatilePtrTest::test ().
     *
     * The purpose of ReadPtr is to provide thread safe access to an a const
     * object for a thread during the lifetime of the ReadPtr. This access
     * may be shared by multiple threads that simultaneously use their own
     * ReadPtr to access the same object.
     *
     * The accessors always returns an accessible instance, never null. For
     * this purpose neither ReadPtr nor WritePtr provides any exstensive
     * interfaces to for instance perform unlock and relock.
     *
     * @see void VolatilePtrTest::test ()
     * @see class VolatilePtr
     * @see class WritePtr
     */
    class ReadPtr : boost::noncopyable {
        // Can't implement a copy constructor as ReadPtr wouldn't be able to
        // maintain the lock through a copy. If ReadPtr would release the lock and
        // obtain a new lock the purpose of the class would be defeated.
        // An attempt to lock again for reading may be blocked if another
        // thread has attempted a write lock in between.
    public:

        explicit ReadPtr (const Ptr& p, int timeout_ms=VolatilePtr_lock_timeout_ms)
            :   l_ ((*const_cast<const Ptr*> (&p))->readWriteLock()),
                p_ (p),
                // accessor operators return this non-volatile instance
                t_ (const_cast<const T*> (p.get ()))
        {
            lock(timeout_ms);
        }

        explicit ReadPtr (const volatile Ptr& p, int timeout_ms=VolatilePtr_lock_timeout_ms)
            :   l_ ((*const_cast<const Ptr*> (&p))->readWriteLock()),
                p_ (*const_cast<const Ptr*> (&p)),
                // accessor operators return this non-volatile instance
                t_ (const_cast<const T*> (const_cast<const Ptr*> (&p)->get ()))
        {
            lock(timeout_ms);
        }

        explicit ReadPtr (const volatile ConstPtr& p, int timeout_ms=VolatilePtr_lock_timeout_ms)
            :   l_ ((*const_cast<const ConstPtr*> (&p))->readWriteLock()),
                p_ (*const_cast<const ConstPtr*> (&p)),
                // accessor operators return this non-volatile instance
                t_ (const_cast<const T*> (const_cast<const ConstPtr*> (&p)->get ()))
        {
            lock(timeout_ms);
        }

        explicit ReadPtr (const volatile T* p, int timeout_ms=VolatilePtr_lock_timeout_ms)
            :   l_ (p->readWriteLock()),
                t_ (const_cast<const T*> (p))
        {
            lock(timeout_ms);
        }

        // The copy constructor is not implemented anywhere and ReadPtr is not
        // copyable. But if a there is a public copy constructor the compiler
        // can perform return value optimization in read1 and write1.
        ReadPtr(const ReadPtr&);

        ~ReadPtr() {
            l_->unlock ();
        }

        const T* operator-> () const { return t_; }
        const T& operator* () const { return *t_; }
        const T* get () const { return t_; }

    private:
        // This constructor is not implemented as it's an error to pass a
        // 'const T*' parameter to ReadPTr. If the caller has such a pointer
        // it should use it directly rather than trying to lock it again.
        ReadPtr(T*);
        ReadPtr(const T*);

        void lock(int timeout_ms) {
            if (!l_->tryLockForRead (timeout_ms))
                BOOST_THROW_EXCEPTION(LockFailed()
                                      << typename LockFailed::timeout_value(timeout_ms)
                                      << Backtrace::make (2));
        }

        QReadWriteLock* l_;
        const ConstPtr p_;
        const T* t_;
    };


    /**
     * For examples of usage see void VolatilePtrTest::test ().
     *
     * The purpose of WritePtr is to provide exclusive access to an object for
     * a single thread during the lifetime of the WritePtr.
     *
     * @see void VolatilePtrTest::test ()
     * @see class VolatilePtr
     * @see class ReadPtr
     */
    class WritePtr : boost::noncopyable {
    public:
        explicit WritePtr (const Ptr& p, int timeout_ms=VolatilePtr_lock_timeout_ms)
            :   l_ ((*const_cast<const Ptr*> (&p))->readWriteLock()),
                p_ (*const_cast<const Ptr*> (&p)),
                t_ (const_cast<T*> (const_cast<const Ptr*> (&p)->get ()))
        {
            lock(timeout_ms);
        }

        explicit WritePtr (const volatile Ptr& p, int timeout_ms=VolatilePtr_lock_timeout_ms)
            :   l_ ((*const_cast<const Ptr*> (&p))->readWriteLock()),
                p_ (*const_cast<const Ptr*> (&p)),
                t_ (const_cast<T*> (const_cast<const Ptr*> (&p)->get ()))
        {
            lock(timeout_ms);
        }

        explicit WritePtr (volatile T* p, int timeout_ms=VolatilePtr_lock_timeout_ms)
            :   l_ (p->readWriteLock()),
                t_ (const_cast<T*> (p))
        {
            lock(timeout_ms);
        }

        // See ReadPtr(const ReadPtr&)
        WritePtr(const WritePtr&);

        ~WritePtr() {
            l_->unlock ();
        }

        T* operator-> () const { return t_; }
        T& operator* () const { return *t_; }
        T* get () const { return t_; }

    private:
        // See ReadPtr(const T*)
        WritePtr (T* p);

        void lock(int timeout_ms) {
            if (!l_->tryLockForWrite (timeout_ms))
                BOOST_THROW_EXCEPTION(LockFailed()
                                      << typename LockFailed::timeout_value(timeout_ms)
                                      << Backtrace::make(2));
        }

        QReadWriteLock* l_;
        const Ptr p_;
        T* t_;
    };


    /**
      This would be handy.
      ReadPtr read() volatile const { return ReadPtr(this);}
      WritePtr write() volatile { return WritePtr(this);}
      */
protected:
    /**
     * @brief readWriteLock
     * ok to cast away volatile to access a QReadWriteLock (it's kind of the
     * point of QReadWriteLock that it can be accessed from multiple threads
     * simultaneously)
     * @return the QReadWriteLock* object for this instance.
     */
    QReadWriteLock* readWriteLock() const volatile { return const_cast<QReadWriteLock*>(&lock_); }


private:
    /**
     * @brief lock_
     *
     * For examples of usage see
     * void VolatilePtrTest::test ();
     */
    QReadWriteLock lock_;
};


template<typename T>
typename T::WritePtr write1( const boost::shared_ptr<volatile T>& t) {
    return typename T::WritePtr(t);
}

template<typename T>
typename T::WritePtr write1( volatile const boost::shared_ptr<volatile T>& t) {
    return typename T::WritePtr(t);
}

template<typename T>
typename T::ReadPtr read1( const boost::shared_ptr<volatile T>& t) {
    return typename T::ReadPtr(t);
}

template<typename T>
typename T::ReadPtr read1( volatile const boost::shared_ptr<volatile T>& t) {
    return typename T::ReadPtr(t);
}

class VolatilePtrTest {
public:
    static void test ();
};

#endif // VOLATILELOCK_H
