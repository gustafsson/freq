#ifndef FILTER_H
#define FILTER_H

class Transform_chunk;
typedef boost::shared_ptr<class Filter> FilterPtr;

class Filter
{
public:
    virtual bool operator()( Transform_chunk& ) = 0;
};

class FilterChain: public Filter, std::list<FilterPtr>
{
public:
    bool operator()( Transform_chunk& );
};

#endif // FILTER_H
