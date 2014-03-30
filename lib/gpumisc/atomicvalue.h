#ifndef ATOMICVALUE_H
#define ATOMICVALUE_H

#include "volatileptr.h"

/**
 * The AtomicValue class should provide thread-safe access to a value.
 *
 * Type T must be copyable and have a default constructor.
 */
template<typename T>
class AtomicValue
{
public:
    AtomicValue(const T& value = T()) : value_(new T(value)) {}

    void operator=(const T& value) { *WritePtr(value_) = value; }
    operator T() const { return *ReadPtr(value_); }

private:
    typedef typename VolatilePtr<T>::WritePtr WritePtr;
    typedef typename VolatilePtr<T>::ReadPtr ReadPtr;

    VolatilePtr<T> value_;
};


class AtomicValueTest {
public:
    static void test();
};

#endif // ATOMICVALUE_H
