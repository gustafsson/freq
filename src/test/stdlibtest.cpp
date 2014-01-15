#include "stdlibtest.h"
#include "exceptionassert.h"

#include <boost/weak_ptr.hpp>

namespace Test {

void Stdlibtest::
        test()
{
    std::vector<float> r;
    r.reserve(10);
    EXCEPTION_ASSERT_EQUALS(r.size (), 0u);
    r.push_back(4);
    EXCEPTION_ASSERT_EQUALS(r.size (), 1u);

    boost::shared_ptr<float> f(new float(4));
    boost::weak_ptr<float> w=f;
    f.reset();
    boost::shared_ptr<float> f2 = w.lock ();
    EXCEPTION_ASSERT(!f2);
    try {
        boost::shared_ptr<float> f3(w);
        EXCEPTION_ASSERT( false );
    }
    catch (boost::bad_weak_ptr&)
    {}
}

} // namespace Test
