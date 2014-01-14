#ifndef FILTERS_RECTANGLE_H
#define FILTERS_RECTANGLE_H

#include "tfr/cwtfilter.h"
#include "filters/selection.h"

// boost
#include <boost/serialization/nvp.hpp>

namespace Filters {

class RectangleKernel: public Tfr::CwtChunkFilter
{
public:
    RectangleKernel(float s1, float f1, float s2, float f2, bool save_inside=false);

    void subchunk( Tfr::ChunkAndInverse& c );

private:
    float _s1, _f1, _s2, _f2;
    bool _save_inside;
};


/**
 * @brief The Rectangle class should apply a bandpass and time filter between f1,t1 and f2,t2 to a signal.
 */
class Rectangle: public Tfr::CwtChunkFilterDesc, public Filters::Selection
{
public:
    Rectangle(float s1, float f1, float s2, float f2, bool save_inside=false);

    // ChunkFilterDesc
    Tfr::pChunkFilter               createChunkFilter(Signal::ComputingEngine* engine) const;
    Signal::OperationDesc::Extent   extent() const;
    ChunkFilterDesc::Ptr            copy() const;

    // Filters::Selection
    bool isInteriorSelected() const override;
    void selectInterior(bool v=true) override;

    float _s1, _f1, _s2, _f2;
    bool _save_inside;

    std::string name();
    Signal::Intervals zeroed_samples();
    Signal::Intervals affected_samples();

private:
    Signal::Intervals outside_samples();

    Rectangle() {} // for deserialization

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & make_nvp("t1", _s1) & make_nvp("f1", _f1)
           & make_nvp("t2", _s2) & make_nvp("f2", _f2)
           & make_nvp("save_inside", _save_inside);
    }

public:
    static void test();
};


} // namespace Filters

#endif // FILTERS_RECTANGLE_H
