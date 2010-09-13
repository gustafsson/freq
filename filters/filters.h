#ifndef FILTERSFILTERS_H
#define FILTERSFILTERS_H

#include "sawe/selection.h"
#include "signal/operation.h"
#include "tfr/cwtfilter.h"

#include <list>
#include <boost/shared_ptr.hpp>


namespace Filters {

    class CwtRenderFilter: public Tfr::CwtFilter
{
public:
    virtual void operator()( Tfr::Chunk& ) { /* TODO render in heightmap! */ }
};

class SelectionFilter: public Tfr::CwtFilter
{
public:
    SelectionFilter( Selection s );

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

    Selection s;

private:
    // TODO Why not copyable?
    SelectionFilter& operator=(const SelectionFilter& );
    SelectionFilter(const SelectionFilter& );
};


class EllipsFilter: public Tfr::CwtFilter
{
public:
    EllipsFilter(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

    float _t1, _f1, _t2, _f2;
    bool _save_inside;

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int version) {
        using boost::serialization::make_nvp;

        ar & make_nvp("Operation", boost::serialization::base_object<Operation>(*this))
           & make_nvp("t1", _t1) & make_nvp("f1", _f1)
           & make_nvp("t2", _t2) & make_nvp("f2", _f2)
           & make_nvp("save_inside", _save_inside);
    }
};


class SquareFilter: public Tfr::CwtFilter
{
public:
    SquareFilter(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

    float _t1, _f1, _t2, _f2;
    bool _save_inside;
};


class MoveFilter: public Tfr::CwtFilter
{
public:
    MoveFilter(float df);

    virtual void operator()( Tfr::Chunk& );

    float _df;
};


class ReassignFilter: public Tfr::CwtFilter
{
public:
    virtual void operator()( Tfr::Chunk& );
};


class TonalizeFilter: public Tfr::CwtFilter
{
public:
    virtual void operator()( Tfr::Chunk& );
};

} // namespace Filters

#endif // FILTERSFILTERS_H
