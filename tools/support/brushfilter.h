#ifndef BRUSHFILTER_H
#define BRUSHFILTER_H

#include "tfr/cwtfilter.h"
#include <GpuCpuData.h>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <vector_types.h>
#include "heightmap/collection.h"

namespace Tools {
namespace Support {

class BrushFilter : public Tfr::CwtFilter
{
public:
    BrushFilter();

    typedef boost::shared_ptr< GpuCpuData<float> > BrushImageDataP;
    typedef boost::unordered_map<Heightmap::Reference, BrushImageDataP> BrushImages;
    typedef boost::shared_ptr<BrushImages> BrushImagesP;

    /**
      These images will be used when the brush is drawn.
      */
    BrushImagesP images;
};


class MultiplyBrush: public BrushFilter
{
public:
    virtual Signal::Intervals affected_samples();

    virtual void operator()( Tfr::Chunk& );
};

} // namespace Support
} // namespace Tools

#endif // BRUSHFILTER_H
