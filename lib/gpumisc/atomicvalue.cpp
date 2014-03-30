#include "atomicvalue.h"
#include "exceptionassert.h"

void AtomicValueTest::
        test()
{
    // It should provide thread-safe access to a value.
    {
        AtomicValue<int> i(5);

        EXCEPTION_ASSERT_EQUALS(i, 5);

        i = 10;

        EXCEPTION_ASSERT_EQUALS(i, 10);
    }
}

