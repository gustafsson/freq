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
    EXCEPTION_ASSERT_EQUALS(texture_->getWidth (), block_layout.texels_per_row ());
    EXCEPTION_ASSERT_EQUALS(texture_->getHeight (), block_layout.texels_per_column ());

#if GL_EXT_debug_label
    glLabelObjectEXT(GL_TEXTURE, texture_->getOpenGlTextureId (), 0, (boost::format("%s") % getVisibleRegion ()).str().c_str());
#endif
}


Block::pGlTexture Block::
        texture() const
{
    return texture_;
}


int Block::
        texture_ota() const
{
    return texture_ota_ ? texture_ota_ ->getOpenGlTextureId () : 0;
}


void Block::
        generateMipmap()
{
    // don't generate any mipmap if the min filter doesn't use mipmaps
    switch(texture_->getMinFilter ())
    {
    case GL_NEAREST:
    case GL_LINEAR:
        return;
    }

    this->updater ()->mipmapbuilder ()->generateMipmap (*texture_, BlockManagement::MipmapBuilder::MipmapOperator_Max, Render::BlockTextures::max_level);

    if (texture_ota_)
        this->updater ()->mipmapbuilder ()->generateMipmap (*texture_ota_, *texture_, BlockManagement::MipmapBuilder::MipmapOperator_OTA);
}


void Block::
        enableOta(bool v)
{
    if (v && !texture_ota_)
    {
        // could use a separate Render::BlockTextures for this to conserve memory
        // besides, as only the highest mipmaps are actually used there could be a single texture only used in building
        GLuint tex;
        int w = Render::BlockTextures::getWidth ()/2;
        int h = Render::BlockTextures::getHeight ()/2;
        glGenTextures (1, &tex); // GlTexture becomes responsible for glDeleteTextures
        texture_ota_.reset (new GlTexture(tex, w, h, true));
        Render::BlockTextures::setupTexture(tex, w, h, 1000);
        texture_ota_->setMinFilter(GL_LINEAR_MIPMAP_LINEAR);
    }

    if (!v && texture_ota_)
        texture_ota_.reset ();
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
