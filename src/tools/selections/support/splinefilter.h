#ifndef SPLINEFILTER_H
#define SPLINEFILTER_H

#include "tfr/cwtfilter.h"
#include <vector>

#include <boost/serialization/vector.hpp>

namespace Tools { namespace Selections { namespace Support {

class SplineFilter: public Tfr::CwtFilter
{
public:
    SplineFilter(bool save_inside=true);

    virtual std::string name();

    virtual bool operator()( Tfr::Chunk& );
    virtual Signal::Intervals zeroed_samples();
    virtual Signal::Intervals affected_samples();

	struct SplineVertex
    {
        float t, f; // t in seconds, f in hertz (as opposed to scale in Heightmap::Position)

        friend class boost::serialization::access;
        template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
            using boost::serialization::make_nvp;

            ar & BOOST_SERIALIZATION_NVP(t)
               & BOOST_SERIALIZATION_NVP(f);
        }
    };

    std::vector<SplineVertex> v;
    bool _save_inside;

private:
    Signal::Intervals outside_samples();

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation)
           & BOOST_SERIALIZATION_NVP( v )
           & make_nvp("save_inside", _save_inside);
    }
};

}}} // namespace Tools::Selections::Support

#endif // SPLINEFILTER_H
