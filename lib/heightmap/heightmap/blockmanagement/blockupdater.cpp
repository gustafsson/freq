#include "blockupdater.h"
#include "heightmap/block.h"
#include "log.h"
#include "fbo2block.h"

#include <unordered_map>

//#define INFO
#define INFO if(0)

using namespace std;

namespace Heightmap {
namespace BlockManagement {


BlockUpdater::BlockUpdater()
    :
    queue_(new list<pair<Heightmap::pBlock,DrawFunc>>()),
    fbo2block(new Fbo2Block)
{
}


BlockUpdater::~BlockUpdater()
{
    delete fbo2block;
}


void BlockUpdater::
    processUpdates(bool isMainThread)
{
    list<pair<pBlock, DrawFunc>> q;
    queue_->swap(q);

    if (q.empty ())
        return;

    INFO Log("blockupdater2: updating %d jobs") % q.size ();

    map<Heightmap::pBlock,GlTexture::ptr> textures;
    for (auto i = q.begin (); i != q.end ();)
    {
        const pBlock& block = i->first;
        DrawFunc& draw = i->second;

        INFO Log("blockupdater2: updating %s") % block->getVisibleRegion ();
        glProjection M;
        if (isMainThread)
            textures[block] = block->sourceTexture ();
        else
            textures[block] = Heightmap::Render::BlockTextures::get1 ();

        auto fbo_mapping = fbo2block->begin (block->getOverlappingRegion (), block->sourceTexture (), textures[block], M);

        if (draw(M))
            i = q.erase (i);
        else
            i++;

        fbo_mapping.release ();
    }

    glFlush();

    for (const auto& v : textures)
        v.first->setTexture(v.second);

    // reinsert failed draw attempts
    auto w = queue_.write ();
    w->swap (q);

    // if any new updates arrived during processing push them to the back of the queue
    for (auto a : q)
        w->push_back (std::move(a));

    INFO Log("blockupdater2: queue has %d jobs") % w->size ();
}


void BlockUpdater::
        queueUpdate(pBlock b, BlockUpdater::DrawFunc && f)
{
    queue_->push_back(pair<pBlock, DrawFunc>(b,move(f)));
}

} // namespace BlockManagement
} // namespace Heightmap
