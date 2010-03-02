#ifndef FILTER_H
#define FILTER_H

#include <list>
#include <boost/shared_ptr.hpp>

class Transform_chunk;
class Transform;
class Waveform_chunk;
typedef boost::shared_ptr<class Filter> pFilter;

class Filter
{
public:
    virtual int type() = 0;
    virtual Filter *filter();
    
    virtual bool operator()( Transform_chunk& ) = 0;
    virtual void range(float& start_time, float& end_time) = 0;

    virtual void invalidateWaveform( const Transform&, Waveform_chunk& );
};

class FilterChain: public Filter, public std::list<pFilter>
{
public:
    static const int chain_filter = 3;
    
    virtual bool operator()( Transform_chunk& );
    virtual void range(float& start_time, float& end_time);

    virtual void invalidateWaveform( const Transform&, Waveform_chunk& );
    
    virtual int type();
};

class EllipsFilter: public Filter
{
public:
    static const int ellips_filter = 1;
    EllipsFilter(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual bool operator()( Transform_chunk& );
    virtual void range(float& start_time, float& end_time);
    virtual int type();

    float _t1, _f1, _t2, _f2;
    
private:
    bool _save_inside;
};

class SquareFilter: public Filter
{
public:
    static const int square_filter = 2;
    
    SquareFilter(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual bool operator()( Transform_chunk& );
    virtual void range(float& start_time, float& end_time);
    virtual int type();
    
    float _t1, _f1, _t2, _f2;
    
private:
    bool _save_inside;
};

#endif // FILTER_H
