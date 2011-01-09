#ifndef FILTERS_RECTANGLE_H
#define FILTERS_RECTANGLE_H

#include "tfr/cwtfilter.h"

namespace Filters {

class Rectangle: public Tfr::CwtFilter
{
public:
    Rectangle(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

    float _t1, _f1, _t2, _f2;
    bool _save_inside;

private:
    Rectangle() {} // for deserialization

    Signal::Intervals outside_samples();

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation)
           & make_nvp("t1", _t1) & make_nvp("f1", _f1)
           & make_nvp("t2", _t2) & make_nvp("f2", _f2)
           & make_nvp("save_inside", _save_inside);
    }
};

} // namespace Filters

#endif // FILTERS_RECTANGLE_H
