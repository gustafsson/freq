#ifndef FILTER_ELLIPSE_H
#define FILTER_ELLIPSE_H

#include "tfr/cwtfilter.h"

namespace Filters {

class EllipseKernel: public Tfr::ChunkFilter
{
public:
    EllipseKernel(float t1, float f1, float t2, float f2, bool save_inside=false);

    void operator()( Tfr::ChunkAndInverse& c );

private:
    float _centre_t, _centre_f, _centre_plus_radius_t, _centre_plus_radius_f;
    bool _save_inside;
};


/**
 * @brief The Ellipse class should filter out an ellipse selection from the signal.
 */
class Ellipse: public Tfr::CwtChunkFilterDesc
{
public:
    Ellipse(float t1, float f1, float t2, float f2, bool save_inside=false);

    // ChunkFilterDesc
    Tfr::pChunkFilter       createChunkFilter(Signal::ComputingEngine* engine=0) const;
    ChunkFilterDesc::Ptr    copy() const;

    float _centre_t, _centre_f, _centre_plus_radius_t, _centre_plus_radius_f;
    bool _save_inside;

    std::string name();
    Signal::Intervals zeroed_samples(float FS);
    Signal::Intervals affected_samples(float FS);

private:
    Signal::Intervals outside_samples(float FS);

    Ellipse() {} // for deserialization

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & make_nvp("t1", _centre_t) & make_nvp("f1", _centre_f)
           & make_nvp("t2", _centre_plus_radius_t) & make_nvp("f2", _centre_plus_radius_f)
           & make_nvp("save_inside", _save_inside);
    }

public:
    static void test();
};

} // namespace Filters

#endif // FILTER_ELLIPSE_H
