#include "blockupdater.h"
#include "heightmap/block.h"
#include "log.h"
#include "fbo2block.h"
#include "heightmap/render/blocktextures.h"
#include "glgroupmarker.h"
#include "mipmapbuilder.h"

#include <unordered_map>

using namespace std;

namespace Heightmap {
namespace BlockManagement {


BlockUpdater::BlockUpdater()
    :
    queue_(new list<pair<Heightmap::pBlock,DrawFunc>>())
{
}


BlockUpdater::~BlockUpdater()
{

}


void BlockUpdater::
    processUpdates(bool isMainThread)
{
    if (!isMainThread) {
        /**
          * MergerTexture needs to prepare the block when it is about to be rendered.
          * Which is a tad late and requires a new FBO and then a swap back to the main FBO.
          *
          * Then after a glFlush the update thread can write to the block, but not before
          * the glFlush as the merged texture might not be ready. updater_->processUpdates
          * should check this and put the update on hold until generateMipmap has been called,
          * indicating that a new frame has begun an thus that there has been a glFlush since
          * MergerTexture.
          */

        EXCEPTION_ASSERTX(false, "Painting on blocks from the update thread is not implemented");
    }

    list<pair<pBlock, DrawFunc>> q;
    queue_->swap(q);

    if (q.empty ())
        return;

    GlGroupMarker gpm("ProcessUpdates");

    if (!fbo2block_)
        fbo2block_.reset (new Fbo2Block);

    // draw multiple updates to a block together
    map<pBlock, list<DrawFunc>> p;
    for (auto i = q.begin (); i != q.end (); i++)
        p[i->first].push_back(move(i->second));

    // release resources bound in function objects
    q_success_.clear ();

    list<pair<pBlock, DrawFunc>> q_failed;
    for (auto i = p.begin (); i != p.end (); i++)
    {
        const pBlock& block = i->first;

        glProjection M;

        auto fbo_mapping = fbo2block_->begin (block->getOverlappingRegion (), block->texture (), block->texture (), M);

        for (auto j = i->second.begin (); j != i->second.end (); j++)
        {
            DrawFunc& draw = *j;

            // try again next frame with draw calls that return false (i.e if a sync object isn'r ready yet)
            // keep successfull draw calls around to prevent opengl resources from being reused (causing sync or corrupted data) before opengl is done with them
            (draw(M) ? q_success_ : q_failed)
               .push_back (pair<pBlock, DrawFunc>(block,move(*j)));
        }

        fbo_mapping.release ();

        block->generateMipmap();
    }

    // reinsert failed draw attempts
    auto w = queue_.write ();
    w->swap (q_failed);

    // if any new updates arrived during processing push any failed draw attempts to the back of the queue
    for (auto& a : q_failed)
        w->push_back (std::move(a));
}


void BlockUpdater::
        queueUpdate(pBlock b, BlockUpdater::DrawFunc && f)
{
    queue_->push_back(pair<pBlock, DrawFunc>(b,move(f)));
}


void BlockUpdater::
        clearQueue()
{
    queue_->clear();
    q_success_.clear ();
    fbo2block_.reset ();
    mipmapbuilder_.reset ();
}


MipmapBuilder* BlockUpdater::
        mipmapbuilder()
{
    if (!mipmapbuilder_)
        mipmapbuilder_.reset(new MipmapBuilder);
    return mipmapbuilder_.get ();
}


} // namespace BlockManagement
} // namespace Heightmap
