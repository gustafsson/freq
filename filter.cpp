#include "filter.h"

static class apply_filter
{
    Transform_chunk& t;
    bool r;
public:
    apply_filter( Transform_chunk& t):t(t),r(true) {}

    void operator()( FilterPtr p) {
        r |= (*p)( t );
    }
};

class FilterChain: public Filter, std::list<FilterPtr>
{
public:
    bool operator()( Transform_chunk& t) {
        return foreach(begin(), end(), apply_filter( t ));
    }
};
