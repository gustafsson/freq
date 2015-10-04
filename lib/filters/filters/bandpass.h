#ifndef BANDPASS_H
#define BANDPASS_H

#include "tfr/stftfilter.h"
#include "filters/selection.h"

// boost
#include <boost/serialization/nvp.hpp>

namespace Filters {

class BandpassKernel: public Tfr::ChunkFilter
{
public:
    BandpassKernel(float f1, float f2, bool save_inside=false);

    std::string name();
    void operator()( Tfr::ChunkAndInverse& chunk );

private:
    float _f1, _f2;
    bool _save_inside;
};


/**
 * @brief The Bandpass class should apply a bandpass filter between f1 and f2 to a signal.
 */
class Bandpass: public Tfr::StftFilterDesc, public Filters::Selection
{
public:
    Bandpass(float f1, float f2, bool save_inside=false);

    // ChunkFilterDesc
    Tfr::pChunkFilter    createChunkFilter(Signal::ComputingEngine* engine) const override;
    ChunkFilterDesc::ptr copy() const override;

    // Filters::Selection
    bool isInteriorSelected() const override;
    void selectInterior(bool v=true) override;

    float _f1, _f2;
    bool _save_inside;

private:
    Bandpass(); // for deserialization

    friend class boost::serialization::access;
    template<class archive> void serialize(archive& ar, const unsigned int /*version*/) {
        using boost::serialization::make_nvp;

        ar & make_nvp("f1", _f1) & make_nvp("f2", _f2)
           & make_nvp("save_inside", _save_inside);
    }

public:
    static void test();
};

} // namespace Filters

#endif // BANDPASS_H
