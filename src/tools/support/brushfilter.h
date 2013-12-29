#ifndef BRUSHFILTER_H
#define BRUSHFILTER_H

#include "tfr/cwtfilter.h"
#include "heightmap/reference_hash.h"
#include "heightmap/tfrmapping.h"

// boost
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/unordered_map.hpp>

// qt, zlib compression
#include <QByteArray>

namespace Tools {
namespace Support {


class BrushFilter : public Tfr::ChunkFilter, public boost::noncopyable
{
public:
    BrushFilter(Heightmap::BlockLayout, Heightmap::VisualizationParams::ConstPtr);
    ~BrushFilter();

    typedef DataStorage<float>::Ptr BrushImageDataP;
    typedef boost::unordered_map<Heightmap::Reference, BrushImageDataP> BrushImages;
    typedef boost::shared_ptr<BrushImages> BrushImagesP;

    /**
      These images will be used when the brush is drawn.
      */
    BrushImagesP images;

    void release_extra_resources();
    BrushImageDataP getImage(Heightmap::Reference const& ref);


protected:
    /**
     * @brief block_layout_ and visualization_params_ describes how brush images
     * are mapped to the heightmap
     */
    const Heightmap::BlockLayout block_layout_;
    Heightmap::VisualizationParams::ConstPtr visualization_params_;

private:
    class BrushFilterSupport* resource_releaser_;
};


class MultiplyBrush: public BrushFilter
{
public:
    MultiplyBrush(Heightmap::BlockLayout, Heightmap::VisualizationParams::ConstPtr);

    virtual Signal::Intervals affected_samples();

    virtual std::string name();
    void operator()( Tfr::ChunkAndInverse& chunk );

private:
    friend class boost::serialization::access;
    MultiplyBrush():BrushFilter(Heightmap::BlockLayout(0,0, 0), Heightmap::VisualizationParams::ConstPtr())
    { BOOST_ASSERT(false); } // required by serialization, should never be called

    template<class archive> void save(archive& ar, const unsigned int version) const {
        // TODO serialize tfr_mapping

		size_t N = images->size();
        ar & BOOST_SERIALIZATION_NVP(N);
        for (BrushImages::iterator i=images->begin(); i!=images->end(); ++i)
        {
            BrushImages::value_type& bv = *i;

			Heightmap::Reference rcopy = bv.first;
			serialize_ref(ar, rcopy);

            DataStorageSize sz = bv.second->size();
            ar & BOOST_SERIALIZATION_NVP(sz.width);
            ar & BOOST_SERIALIZATION_NVP(sz.height);

            if (version<=0)
            {
                boost::serialization::binary_object Data( bv.second->getCpuMemory(), bv.second->numberOfBytes() );
                ar & BOOST_SERIALIZATION_NVP(Data);
            }
            else
            {
                QByteArray zlibUncompressed = QByteArray::fromRawData( (char*)bv.second->getCpuMemory(), bv.second->numberOfBytes() );
                QByteArray zlibCompressed = qCompress(zlibUncompressed);
                unsigned compressedN = zlibCompressed.size();
                ar & BOOST_SERIALIZATION_NVP(compressedN);
                ar.save_binary( zlibCompressed.constData(), compressedN );
            }
        }
    }

    template<class archive> void load(archive& ar, const unsigned int version) {
        unsigned N = 0;
        ar & BOOST_SERIALIZATION_NVP(N);
        for (unsigned i=0; i<N; ++i)
        {
            Heightmap::Reference ref;
			serialize_ref(ar, ref);

            DataStorageSize sz(0);
            ar & BOOST_SERIALIZATION_NVP(sz.width);
            ar & BOOST_SERIALIZATION_NVP(sz.height);
            sz.depth = 1;

            BrushImageDataP img(new DataStorage<float>(sz));
            if (version<=0)
            {
                boost::serialization::binary_object Data( img->getCpuMemory(), img->numberOfBytes() );
                ar & BOOST_SERIALIZATION_NVP(Data);
            }
            else
            {
                unsigned compressedN = 0;
                ar & BOOST_SERIALIZATION_NVP(compressedN);
                QByteArray zlibCompressed;
                zlibCompressed.resize(compressedN);
                ar.load_binary( zlibCompressed.data(), compressedN );
                QByteArray zlibUncompressed = qUncompress(zlibCompressed);
                EXCEPTION_ASSERT( img->numberOfBytes() == (size_t)zlibUncompressed.size() );
                memcpy(img->getCpuMemory(), zlibUncompressed.constData(), img->numberOfBytes());
            }

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


class MultiplyBrushDesc: public Tfr::CwtFilterDesc {
public:
    MultiplyBrushDesc(Heightmap::BlockLayout bl, Heightmap::VisualizationParams::ConstPtr vp)
        :
          Tfr::CwtFilterDesc(Tfr::pChunkFilter(new MultiplyBrush(bl, vp)))
    {}
};


} // namespace Support
} // namespace Tools

BOOST_CLASS_VERSION(Tools::Support::BrushFilter, 1)

#endif // BRUSHFILTER_H
