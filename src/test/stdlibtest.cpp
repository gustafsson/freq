#include "stdlibtest.h"
#include "exceptionassert.h"

namespace Test {

void Stdlibtest::
        test()
{
    std::vector<float> r;
    r.reserve(10);
    EXCEPTION_ASSERT_EQUALS(r.size (), 0);
    r.push_back(4);
    EXCEPTION_ASSERT_EQUALS(r.size (), 1);
}

} // namespace Test
