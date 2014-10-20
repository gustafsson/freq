#ifndef BLOCK_H
#define BLOCK_H

#include "referenceinfo.h"

// gpumisc
#include "datastorage.h"

#ifndef SAWE_NO_MUTEX
#include <QMutex>
#endif

class GlTexture;

namespace Heightmap {

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
        Block( Reference, BlockLayout, VisualizationParams::const_ptr);
        Block( const Block&)=delete;
        Block& operator=( const Block&)=delete;

        unsigned frame_number_last_used;

        // OpenGL data to render
        // Probably need to double-buffer this so that drawing FROM this
        // does not collide with drawing TO this texture. But how, all draw
        // calls are asynchronous ...
        pGlTexture texture() const;
        // OpenGL data to use as source when updating with new content
        pGlTexture sourceTexture() const;
        // Call after glFlush to 't'
        void setTexture(pGlTexture t);

        // The block must exist for one whole frame before it can receive
        // updates from another thread. This prevents the texture from being
        // corrupted by having two threads writing to it at the same time.
        bool isTextureReady() const;
        void setTextureReady();

        // POD properties
        const BlockLayout block_layout() const { return block_layout_; }
        Reference reference() const { return ref_; }
        Signal::Interval getInterval() const { return block_interval_; }
        Region getDataRegion() const { return data_region_; }
        Region getRenderRegion() const { return render_region_; }
        float sample_rate() const { return sample_rate_; }

        // Helper
        ReferenceInfo referenceInfo() const { return ReferenceInfo(reference (), block_layout (), visualization_params ()); }

        // Shared state
        const VisualizationParams::const_ptr visualization_params() const { return visualization_params_; }

    private:
        const Reference ref_;
        const BlockLayout block_layout_;
        const Signal::Interval block_interval_;
        const Region data_region_, render_region_;
        const float sample_rate_;

        const VisualizationParams::const_ptr visualization_params_;
        pGlTexture new_texture_;
        pGlTexture texture_;
        pGlTexture texture_hold_; // @see setTextureReady
        bool texture_ready_ = false;

    public:
        static void test();
    };
    typedef boost::shared_ptr<Block> pBlock;

} // namespace Heightmap

#endif // BLOCK_H
