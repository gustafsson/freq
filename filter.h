#ifndef FILTER_H
#define FILTER_H

#include <list>
#include <boost/shared_ptr.hpp>

class Transform_chunk;
typedef boost::shared_ptr<class Filter> pFilter;

class Filter
{
public:
    virtual bool operator()( Transform_chunk& ) = 0;
};

class FilterChain: public Filter, std::list<pFilter>
{
public:
    bool operator()( Transform_chunk& );
};

#endif // FILTER_H
