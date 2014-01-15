#ifndef SPLINEFILTER_H
#define SPLINEFILTER_H

#include "tfr/cwtfilter.h"
#include <vector>

#include <boost/serialization/vector.hpp>

namespace Tools { namespace Selections { namespace Support {

class SplineFilter: public Tfr::ChunkFilter
{
public:
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


    SplineFilter(bool save_inside=true, std::vector<SplineVertex> v=std::vector<SplineVertex>());

    virtual std::string name();

    void operator()( Tfr::ChunkAndInverse& chunk );
    virtual Signal::Intervals zeroed_samples(double FS);
    virtual Signal::Intervals affected_samples(double FS);

    std::vector<SplineVertex> v;
    bool _save_inside;

private:
    Signal::Intervals outside_samples(double FS);

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & BOOST_SERIALIZATION_NVP( v )
           & make_nvp("save_inside", _save_inside);
    }
};


class SplineFilterDesc: public Tfr::CwtChunkFilterDesc {
public:
    SplineFilterDesc(bool save_inside, std::vector<SplineFilter::SplineVertex> v);

    // ChunkFilterDesc
    Tfr::pChunkFilter       createChunkFilter(Signal::ComputingEngine* engine) const;
    ChunkFilterDesc::Ptr    copy() const;

private:
    bool save_inside;
    std::vector<SplineFilter::SplineVertex> v;
};

}}} // namespace Tools::Selections::Support

#endif // SPLINEFILTER_H
