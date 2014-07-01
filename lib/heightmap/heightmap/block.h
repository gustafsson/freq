#ifndef BLOCK_H
#define BLOCK_H

#include "referenceinfo.h"
#include "render/glblock.h"

// gpumisc
#include "datastorage.h"

#ifndef SAWE_NO_MUTEX
#include <QMutex>
#endif

namespace Heightmap {

    namespace Render {
        class GlBlock;
    }

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
        typedef Render::GlBlock::ptr pGlBlock;

        Block( Reference, BlockLayout, VisualizationParams::const_ptr);
        ~Block();

        // TODO move this value to a complementary class
        unsigned frame_number_last_used;

        // OpenGL data to render
        pGlBlock glblock;

        // Shared state
        const VisualizationParams::const_ptr visualization_params() const { return visualization_params_; }

        // POD properties
        const BlockLayout block_layout() const { return block_layout_; }
        Reference reference() const { return ref_; }
        Signal::Interval getInterval() const { return block_interval_; }
        Region getRegion() const { return region_; }
        float sample_rate() const { return sample_rate_; }

        // Helper
        ReferenceInfo referenceInfo() const { return ReferenceInfo(reference (), block_layout (), visualization_params ()); }

    private:
        const Reference ref_;
        const BlockLayout block_layout_;
        const VisualizationParams::const_ptr visualization_params_;

        const Signal::Interval block_interval_;
        const Region region_;
        const float sample_rate_;

    public:
        static void test();
    };
    typedef boost::shared_ptr<Block> pBlock;

} // namespace Heightmap

#endif // BLOCK_H
