#ifndef BRUSHFILTER_H
#define BRUSHFILTER_H

#include "tfr/cwtfilter.h"
#include <GpuCpuData.h>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <vector_types.h>

namespace Tools {
namespace Support {

class BrushFilter : public Tfr::CwtFilter
{
public:
    typedef boost::shared_ptr< GpuCpuData<float> > BrushImageDataP;
    struct BrushImage {
        float startTime;
        float endTime;
        float min_hz;
        float max_hz;
        BrushImageDataP data;
    };
    typedef std::vector<BrushImage> BrushImages;
    typedef boost::shared_ptr<BrushImages> BrushImagesP;

    /**
      These images will be used when the brush is drawn.
      */
    BrushImagesP images;
};


class MultiplyBrush: public BrushFilter
{
protected:
    virtual void operator()( Tfr::Chunk& );
};

} // namespace Support
} // namespace Tools

#endif // BRUSHFILTER_H
