#ifndef BRUSHFILTER_H
#define BRUSHFILTER_H

#include "tfr/cwtfilter.h"
#include <GpuCpuData.h>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <vector_types.h>
#include "heightmap/collection.h"

#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/shared_ptr.hpp>

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

    BrushImageDataP getImage(Heightmap::Reference const& ref);
};


class MultiplyBrush: public BrushFilter
{
public:
    virtual Signal::Intervals affected_samples();

    virtual void operator()( Tfr::Chunk& );

private:
    friend class boost::serialization::access;
    template<class archive> void save(archive& ar, const unsigned int /*version*/) const {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
    }
    template<class archive> void load(archive& ar, const unsigned int /*version*/) {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};


} // namespace Support
} // namespace Tools

//#include <boost/serialization/export.hpp>
//BOOST_CLASS_EXPORT(Tools::Support::MultiplyBrush)

#endif // BRUSHFILTER_H
