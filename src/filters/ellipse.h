#ifndef FILTER_ELLIPSE_H
#define FILTER_ELLIPSE_H

#include "tfr/cwtfilter.h"

namespace Filters {

class Ellipse: public Tfr::CwtFilter
{
public:
    Ellipse(float t1, float f1, float t2, float f2, bool save_inside=false);

    virtual std::string name();
    virtual bool operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

    float _centre_t, _centre_f, _centre_plus_radius_t, _centre_plus_radius_f;
    bool _save_inside;

private:
    Ellipse() {} // for deserialization

    Signal::Intervals outside_samples();

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DeprecatedOperation)
           & make_nvp("t1", _centre_t) & make_nvp("f1", _centre_f)
           & make_nvp("t2", _centre_plus_radius_t) & make_nvp("f2", _centre_plus_radius_f)
           & make_nvp("save_inside", _save_inside);
    }
};

} // namespace Filters

#endif // FILTER_ELLIPSE_H
