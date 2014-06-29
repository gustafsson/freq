#include "clearinterval.h"
#include "clearkernel.h"
#include "resampletexture.h"
#include "heightmap/render/glblock.h"

namespace Heightmap {
namespace Blocks {

ClearInterval::
        ClearInterval(BlockCache::ptr cache)
    :
      cache_(cache)
{
}


std::list<pBlock> ClearInterval::
        discardOutside(Signal::Interval& I)
{
    std::list<pBlock> discarded;

    BlockCache::cache_t C = cache_->clone();
    for (BlockCache::cache_t::value_type itr : C)
    {
        pBlock block(itr.second);
        Signal::Interval blockInterval = block->getInterval();
        Signal::Interval toKeep = I & blockInterval;
        bool remove_entire_block = toKeep == Signal::Interval();
        bool keep_entire_block = toKeep == blockInterval;
        if ( remove_entire_block )
        {
            discarded.push_back (block);
        }
        else if ( keep_entire_block )
        {
        }
        else
        {
            // clear partial block
            if( I.first <= blockInterval.first && I.last < blockInterval.last )
            {
                Region ir = block->getRegion ();
                ResampleTexture rt(block->glblock->glTexture ()->getOpenGlTextureId ());
                ResampleTexture::Area A(ir.a.time, ir.a.scale, ir.b.time, ir.b.scale);
                GlFrameBuffer::ScopeBinding sb = rt.enable(A);
                float t = I.last / block->block_layout ().targetSampleRate();
                A.x1 = t;
                rt.drawColoredArea (A, 0.f);
                (void)sb; // RAII
            }
        }
    }

    return discarded;
}

} // namespace Blocks
} // namespace Heightmap
