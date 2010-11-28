#ifndef FILTER_ELLIPSE_H
#define FILTER_ELLIPSE_H

#include "tfr/cwtfilter.h"

namespace Filters {

class Ellipse: public Tfr::CwtFilter
{
public:
    Ellipse(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual void operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

    float _t1, _f1, _t2, _f2;
    bool _save_inside;

private:
    Ellipse() {} // for deserialization

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

//#include <boost/serialization/export.hpp>
//BOOST_CLASS_EXPORT(Filters::Ellipse)

#endif // FILTER_ELLIPSE_H
