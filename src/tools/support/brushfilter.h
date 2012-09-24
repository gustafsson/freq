#ifndef BRUSHFILTER_H
#define BRUSHFILTER_H

#include "tfr/cwtfilter.h"
#include "heightmap/reference_hash.h"

// boost
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/unordered_map.hpp>

// qt, zlib compression
#include <QByteArray>

namespace Tools {
namespace Support {


class BrushFilter : public Tfr::CwtFilter, public boost::noncopyable
{
public:
    BrushFilter();
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

private:
    class BrushFilterSupport* resource_releaser_;
};


class MultiplyBrush: public BrushFilter
{
public:
    virtual Signal::Intervals affected_samples();

    virtual std::string name();
    virtual bool operator()( Tfr::Chunk& );

private:
    friend class boost::serialization::access;
    template<class archive> void save(archive& ar, const unsigned int version) const {
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);

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
        ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Operation);

        unsigned N = 0;
        ar & BOOST_SERIALIZATION_NVP(N);
        for (unsigned i=0; i<N; ++i)
        {
            Heightmap::Reference ref(0);
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
                BOOST_ASSERT( img->numberOfBytes() == (size_t)zlibUncompressed.size() );
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


} // namespace Support
} // namespace Tools

BOOST_CLASS_VERSION(Tools::Support::BrushFilter, 1)

#endif // BRUSHFILTER_H
