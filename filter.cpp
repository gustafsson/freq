#include "filter.h"
#include <functional>

class apply_filter
{
    Transform_chunk& t;
    bool r;
public:
    apply_filter( Transform_chunk& t):t(t),r(true) {}

    void operator()( pFilter p) {
        r |= (*p)( t );
    }

    operator bool() { return r; }
};


bool FilterChain::operator()( Transform_chunk& t) {
    return std::for_each(begin(), end(), apply_filter( t ));
}
