#ifndef BRUSHFILTER_H
#define BRUSHFILTER_H

#include "tfr/cwtfilter.h"
#include "heightmap/collection.h"

// gpumisc
#include <GpuCpuData.h>
#include <vector_types.h>

// boost
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/binary_object.hpp>

// std
#include <vector>

namespace Tools {
namespace Support {


class BrushFilter : public Tfr::CwtFilter, public boost::noncopyable
{
public:
    BrushFilter();
    ~BrushFilter();

    typedef boost::shared_ptr< GpuCpuData<float> > BrushImageDataP;
    typedef boost::unordered_map<Heightmap::Reference, BrushImageDataP> BrushImages;
    typedef boost::shared_ptr<BrushImages> BrushImagesP;

    /**
      These images will be used when the brush is drawn.
      */
    BrushImagesP images;

    void release_extra_resources();
	void validateRefs(Heightmap::Collection* collection);
    BrushImageDataP getImage(Heightmap::Reference const& ref);

private:
    class BrushFilterSupport* resource_releaser_;
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

		size_t N = images->size();
        ar & BOOST_SERIALIZATION_NVP(N);
        foreach(BrushImages::value_type bv, *images)
        {
			Heightmap::Reference rcopy = bv.first;
			serialize_ref(ar, rcopy);

            cudaExtent sz = bv.second->getNumberOfElements();
            ar & BOOST_SERIALIZATION_NVP(sz.width);
            ar & BOOST_SERIALIZATION_NVP(sz.height);

            boost::serialization::binary_object Data( bv.second->getCpuMemory(), bv.second->getSizeInBytes1D() );
            ar & BOOST_SERIALIZATION_NVP(Data);
        }
    }
    template<class archive> void load(archive& ar, const unsigned int /*version*/) {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);

        unsigned N = 0;
        ar & BOOST_SERIALIZATION_NVP(N);
        for (unsigned i=0; i<N; ++i)
        {
            Heightmap::Reference ref(0);
			serialize_ref(ar, ref);

            cudaExtent sz;
            ar & BOOST_SERIALIZATION_NVP(sz.width);
            ar & BOOST_SERIALIZATION_NVP(sz.height);
            sz.depth = 1;

            BrushImageDataP img(new GpuCpuData<float>(0, sz));
            boost::serialization::binary_object Data( img->getCpuMemory(), img->getSizeInBytes1D() );
            ar & BOOST_SERIALIZATION_NVP(Data);

            (*images)[ ref ] = img;
        }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

	template<class archive> void serialize_ref(archive& ar, Heightmap::Reference& ref) const {
        int lss1 = ref.log2_samples_size[0];
        int lss2 = ref.log2_samples_size[1];
        unsigned bi1 = ref.block_index[0];
        unsigned bi2 = ref.block_index[1];

        ar      & BOOST_SERIALIZATION_NVP(lss1)
                & BOOST_SERIALIZATION_NVP(lss2)
                & BOOST_SERIALIZATION_NVP(bi1)
                & BOOST_SERIALIZATION_NVP(bi2);

        ref.log2_samples_size[0] = lss1;
        ref.log2_samples_size[1] = lss2;
        ref.block_index[0] = bi1;
        ref.block_index[1] = bi2;
    }
};


} // namespace Support
} // namespace Tools

#endif // BRUSHFILTER_H
