#ifndef BLOCK_H
#define BLOCK_H

#include "referenceinfo.h"

// gpumisc
#include "datastorage.h"

class GlTexture;

namespace Heightmap {
    namespace BlockManagement { class BlockUpdater; }

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
        typedef std::shared_ptr<GlTexture> pGlTexture;

        Block( Reference, BlockLayout, VisualizationParams::const_ptr, Heightmap::BlockManagement::BlockUpdater* updater);
        Block( const Block&)=delete;
        Block& operator=( const Block&)=delete;

        unsigned frame_number_last_used;

        // OpenGL data to render
        pGlTexture texture() const;
        int texture_ota() const;
        void generateMipmap(); // will only generate mipmaps if the min filter is using mipmaps
        void enableOta(bool v);

        Heightmap::BlockManagement::BlockUpdater* updater();

        // POD properties
        const BlockLayout block_layout() const { return block_layout_; }
        Reference reference() const { return ref_; }
        Signal::Interval getInterval() const { return block_interval_; }
        Region getOverlappingRegion() const { return overlapping_region_; }
        Region getVisibleRegion() const { return visible_region_; }
        float sample_rate() const { return sample_rate_; }

        // Helper
        ReferenceInfo referenceInfo() const { return ReferenceInfo(reference (), block_layout (), visualization_params ()); }

        // Shared state
        const VisualizationParams::const_ptr visualization_params() const { return visualization_params_; }

    private:
        const Reference ref_;
        const BlockLayout block_layout_;
        const Signal::Interval block_interval_;
        const Region overlapping_region_, visible_region_;
        const float sample_rate_;

        const VisualizationParams::const_ptr visualization_params_;
        pGlTexture texture_, texture_ota_;
        Heightmap::BlockManagement::BlockUpdater* updater_;

    public:
        static void test();
    };
    typedef boost::shared_ptr<Block> pBlock;

} // namespace Heightmap

#endif // BLOCK_H
