#ifndef HEIGHTMAP_BLOCKCONFIGURATION_H
#define HEIGHTMAP_BLOCKCONFIGURATION_H

#include "tfr/freqaxis.h"

#include <boost/shared_ptr.hpp>

namespace Heightmap {

class Collection;

class BlockConfiguration {
public:
    typedef boost::shared_ptr<BlockConfiguration> Ptr;

    BlockConfiguration(Collection*);
    Collection* collection() const;
    void setCollection(Collection* c);
    unsigned samplesPerBlock() const;
    unsigned scalesPerBlock() const;
    float targetSampleRate() const;
    Tfr::FreqAxis display_scale() const;
    Tfr::FreqAxis transform_scale() const;
    float displayedTimeResolution(float ahz) const;
    float length() const;

private:
    Collection* collection_;
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKCONFIGURATION_H
