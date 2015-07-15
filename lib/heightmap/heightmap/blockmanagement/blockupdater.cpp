#include "blockupdater.h"
#include "heightmap/block.h"
#include "log.h"
#include "fbo2block.h"

#include <unordered_map>

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

    // draw multiple updates to a block together
    map<pBlock, list<DrawFunc>> p;
    for (auto i = q.begin (); i != q.end (); i++)
        p[i->first].push_back(move(i->second));

    list<pair<pBlock, DrawFunc>> q_failed;
    map<Heightmap::pBlock,GlTexture::ptr> textures;
    for (auto i = p.begin (); i != p.end (); i++)
    {
        const pBlock& block = i->first;

        glProjection M;
        if (isMainThread)
            textures[block] = block->sourceTexture ();
        else
            textures[block] = Heightmap::Render::BlockTextures::get1 ();

        auto fbo_mapping = fbo2block->begin (block->getOverlappingRegion (), block->sourceTexture (), textures[block], M);

        for (auto j = i->second.begin (); j != i->second.end (); j++)
        {
            DrawFunc& draw = *j;

            draw(M);
            if (!draw.get_future().get())
                q_failed.push_back (pair<pBlock, DrawFunc>(block,move(*j)));
        }

        fbo_mapping.release ();
    }

    for (const auto& v : textures)
        v.first->setTexture(v.second);

    // reinsert failed draw attempts
    auto w = queue_.write ();
    w->swap (q_failed);

    // if any new updates arrived during processing push them to the back of the queue
    for (auto& a : q_failed)
        w->push_back (std::move(a));
}


void BlockUpdater::
        queueUpdate(pBlock b, BlockUpdater::DrawFunc && f)
{
    queue_->push_back(pair<pBlock, DrawFunc>(b,move(f)));
}

} // namespace BlockManagement
} // namespace Heightmap
