#ifndef BLOCK_H
#define BLOCK_H

#include "referenceinfo.h"

// gpumisc
#include "datastorage.h"
#include "volatileptr.h"

#ifndef SAWE_NO_MUTEX
#include <QMutex>
#endif

namespace Heightmap {

    class GlBlock;

    class BlockData: public VolatilePtr<BlockData> {
    public:
        typedef DataStorage<float>::Ptr pData;

        /**
            TODO test this in a multi gpu environment
            For multi-GPU or (just multithreaded) environments, each GPU-thread have
            its own cuda context and data can't be transfered between cuda contexts
            without first going to the cpu. Therefore a 'cpu_copy' is kept in CPU
            memory so that the block data is readily available for merging new
            blocks. Only one GPU may access 'cpu_copy' at once. The OpenGL textures
            are updated from cpu_copy whenever new_data_available is set to true.
        */
        pData cpu_copy;


        /**
          valid_samples describes the intervals of valid samples contained in this block.
          it is relative to the start of the heightmap, not relative to this block.
          The samplerate is the sample rate of the full resolution signal.
          */
        Signal::Intervals valid_samples, non_zero;
    };

    // FEATURE it would probably look awesome if new blocks weren't displayed
    // instantaneously but rather faded in from 0 or from their previous value.
    // This method could be used to slide between the images of two different
    // signals or channels as well. This should be implemented by rendering two or
    // more separate collections in Heightmap::Renderer. It would fetch Blocks by
    // their 'Reference' from the different collections and use a shader to
    // transfer results between them.
    class Block {
    public:
        Block( const Reference&, const TfrMapping& );
        ~Block();

        // TODO move this value to a complementary class
        unsigned frame_number_last_used;
        bool new_data_available;
        bool to_delete;

        // OpenGL data to render
        boost::shared_ptr<GlBlock> glblock;
        BlockData::Ptr block_data() const { return block_data_; }

        const Reference& reference() const { return ref_; }
        const TfrMapping& tfr_mapping() const { return tfr_mapping_; }

        const Signal::Interval& getInterval() const { return block_interval_; }
        const Region& getRegion() const { return region_; }
        float sample_rate() const { return sample_rate_; }

    private:
        BlockData::Ptr block_data_;
        const Reference ref_;
        const TfrMapping tfr_mapping_;

        const Signal::Interval block_interval_;
        const Region region_;
        const float sample_rate_;
    };
    typedef boost::shared_ptr<Block> pBlock;

} // namespace Heightmap

#endif // BLOCK_H
