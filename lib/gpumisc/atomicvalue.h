#ifndef ATOMICVALUE_H
#define ATOMICVALUE_H

#include <volatileptr.h>

/**
 * The AtomicValue class should provide thread-safe access to a value.
 *
 * Type T must be copyable and have a default constructor.
 */
template<typename T>
class AtomicValue: public VolatilePtr<AtomicValue<T> >
{
public:
    AtomicValue(const T& value = T()) : value(value) {}

    void operator=(const T& value) volatile { WritePtr(this)->value = value; }
    operator T() const volatile { return ReadPtr(this)->value; }

private:
    typedef typename VolatilePtr<AtomicValue<T> >::WritePtr WritePtr;
    typedef typename VolatilePtr<AtomicValue<T> >::ReadPtr ReadPtr;

    void operator=(T x); // Use the volatile accessor
    operator T() const;  // Use the volatile accessor

    T value;
};


class AtomicValueTest {
public:
    static void test();
};

#endif // ATOMICVALUE_H
