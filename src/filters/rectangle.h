#ifndef FILTERS_RECTANGLE_H
#define FILTERS_RECTANGLE_H

#include "tfr/cwtfilter.h"

// boost
#include <boost/serialization/nvp.hpp>

namespace Filters {

class RectangleKernel: public Tfr::CwtChunkFilter
{
public:
    RectangleKernel(float t1, float f1, float t2, float f2, bool save_inside=false);

    void subchunk( Tfr::ChunkAndInverse& c );

private:
    float _t1, _f1, _t2, _f2;
    bool _save_inside;
};


/**
 * @brief The Rectangle class should apply a bandpass and time filter between f1,t1 and f2,t2 to a signal.
 */
class Rectangle: public Tfr::CwtChunkFilterDesc
{
public:
    Rectangle(float t1, float f1, float t2, float f2, bool save_inside=false);

    // ChunkFilterDesc
    Tfr::pChunkFilter               createChunkFilter(Signal::ComputingEngine* engine) const;
    Signal::OperationDesc::Extent   extent() const;
    ChunkFilterDesc::Ptr            copy() const;

    float _t1, _f1, _t2, _f2;
    bool _save_inside;

    std::string name();
    Signal::Intervals zeroed_samples(float FS);
    Signal::Intervals affected_samples(float FS);

private:
    Signal::Intervals outside_samples(float FS);

    Rectangle() {} // for deserialization

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & make_nvp("t1", _t1) & make_nvp("f1", _f1)
           & make_nvp("t2", _t2) & make_nvp("f2", _f2)
           & make_nvp("save_inside", _save_inside);
    }

public:
    static void test();
};


} // namespace Filters

#endif // FILTERS_RECTANGLE_H
