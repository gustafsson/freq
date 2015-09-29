#include "block.h"
#include "GlTexture.h"
#include "tasktimer.h"
#include "log.h"

#include "heightmap/render/blocktextures.h"
#include "heightmap/blockmanagement/blockupdater.h"
#include "heightmap/blockmanagement/mipmapbuilder.h"

//#define BLOCK_INFO
#define BLOCK_INFO if(0)

namespace Heightmap {


Block::
        Block( Reference ref,
               BlockLayout block_layout,
               VisualizationParams::const_ptr visualization_params,
               Heightmap::BlockManagement::BlockUpdater* updater )
:
    frame_number_last_used(0),
    ref_(ref),
    block_layout_(block_layout),
    block_interval_( ReferenceInfo(ref, block_layout, visualization_params).getInterval() ),
    //clipping_region_( RegionFactory(block_layout).getClipping (ref) ),
    overlapping_region_( RegionFactory(block_layout).getOverlapping (ref) ),
    visible_region_( RegionFactory(block_layout).getVisible (ref) ),
    sample_rate_( ReferenceInfo(ref, block_layout, visualization_params).sample_rate() ),
    visualization_params_(visualization_params),
    texture_(Render::BlockTextures::get1()),
    updater_(updater)
{
    if (texture_)
    {
        EXCEPTION_ASSERT_EQUALS(texture_->getWidth (), block_layout.texels_per_row ());
        EXCEPTION_ASSERT_EQUALS(texture_->getHeight (), block_layout.texels_per_column ());
    }
}


Block::pGlTexture Block::
        texture() const
{
    return texture_;
}


Block::pGlTexture Block::
        sourceTexture() const
{
    Block::pGlTexture t = new_texture_;
    return t ? t: texture_;
}


void Block::
        setTexture(Block::pGlTexture t)
{
    // Keep texture_ until it has been flushed
    new_texture_ = t;
#if GL_EXT_debug_label
    if (texture_ != t)
        glLabelObjectEXT(GL_TEXTURE, new_texture_->getOpenGlTextureId (), 0, (boost::format("%s") % getVisibleRegion ()).str().c_str());
#endif
}


void Block::
        showNewTexture(bool use_mipmap)
{
    // release previously replaced texture, see below
    texture_hold_.reset ();

    Block::pGlTexture t;
    t.swap (new_texture_);

    if (t) {
        // hold on to the old texture until the next frame to prevent the other thread from
        // replacing its contents right away
        texture_hold_ = texture_;

        // use the new_texture_
        texture_ = t;
    }

    // check if mipmap settings have changed, or if mipmap is needed for a new texture
    bool has_mipmap = texture_->getMinFilter () == GL_LINEAR_MIPMAP_LINEAR;
    if (has_mipmap != use_mipmap || (use_mipmap && t))
    {
        texture_->bindTexture ();
        if (use_mipmap)
        {
            // recalculate mipmap if reenabled or got new texture
            texture_->setMinFilter (GL_LINEAR_MIPMAP_LINEAR);
//            glGenerateMipmap (GL_TEXTURE_2D);

            this->updater ()->mipmapbuilder ()->buildMipmaps (*texture_, BlockManagement::MipmapBuilder::MipmapOperator_Max, Render::BlockTextures::max_level);
        }
        else
        {
            // disable mipmap
            texture_->setMinFilter (GL_LINEAR);
        }
    }
}


Heightmap::BlockManagement::BlockUpdater* Block::
        updater()
{
    EXCEPTION_ASSERT(updater_);
    return updater_;
}


void Block::
        test()
{
    // It should store information and data about a block.
    {
        // ...
    }
}

} // namespace Heightmap
