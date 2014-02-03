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
    };

    // FEATURE it would probably look awesome if new blocks weren't displayed
    // instantaneously but rather faded in from 0 or from their previous value.
    // This method could be used to slide between the images of two different
    // signals or channels as well. This should be implemented by rendering two or
    // more separate collections in Heightmap::Renderer. It would fetch Blocks by
    // their 'Reference' from the different collections and use a shader to
    // transfer results between them.
    /**
     * @brief The Block class should store information and data about a block.
     */
    class Block {
    public:
        Block( Reference, BlockLayout, VisualizationParams::ConstPtr);
        ~Block();

        // TODO move this value to a complementary class
        unsigned frame_number_last_used;

        // OpenGL data to render
        boost::shared_ptr<GlBlock> glblock;
        BlockData::WritePtr block_data();
        void discard_new_data_available();

        // Lock if available but don't wait for it to become available
        // Throws BlockData::LockFailed if data is not available
        BlockData::ReadPtr block_data_const() const {
            return BlockData::ReadPtr(block_data_, NoLockFailed());
        }

        /**
         * @brief update_data updates glblock from block_data
         * @return true if data was updated. false if block_data is currently
         * in use or if glblock is already up-to-date.
         */
        bool update_glblock_data();

        // Shared state
        const VisualizationParams::ConstPtr visualization_params() const { return visualization_params_; }

        // POD properties
        const BlockLayout block_layout() const { return block_layout_; }
        Reference reference() const { return ref_; }
        Signal::Interval getInterval() const { return block_interval_; }
        Region getRegion() const { return region_; }
        float sample_rate() const { return sample_rate_; }

        // Helper
        ReferenceInfo referenceInfo() const { return ReferenceInfo(reference (), block_layout (), visualization_params ()); }

    private:
        BlockData::Ptr block_data_;
        bool new_data_available_;
        const Reference ref_;
        const BlockLayout block_layout_;
        const VisualizationParams::ConstPtr visualization_params_;

        const Signal::Interval block_interval_;
        const Region region_;
        const float sample_rate_;

    public:
        static void test();
    };
    typedef boost::shared_ptr<Block> pBlock;

} // namespace Heightmap

#endif // BLOCK_H
