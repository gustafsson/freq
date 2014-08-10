#include "blockupdater.h"
#include "heightmap/update/tfrblockupdater.h"
#include "tfr/chunk.h"

#include "fbo2block.h"
#include "pbo2texture.h"
#include "source2pbo.h"
#include "texture2fbo.h"

#include "tasktimer.h"
#include "timer.h"
#include "log.h"
#include "neat_math.h"
#include "lazy.h"

#include <thread>
#include <future>
#include <unordered_map>

//#define INFO
#define INFO if(0)

using namespace std;
using namespace JustMisc;

namespace std {
    template<class T>
    struct hash<boost::shared_ptr<T>>
    {
        std::size_t operator()(boost::shared_ptr<T> const& t) const
        {
            return std::hash<T*>()(t.get ());
        }
    };
}

namespace Heightmap {
namespace Update {
namespace OpenGL {


class BlockUpdaterPrivate
{
public:
    Shaders shaders;
    Fbo2Block fbo2block;
};

BlockUpdater::
        BlockUpdater()
    :
//      memcpythread(thread::hardware_concurrency ())
      p(new BlockUpdaterPrivate),
      memcpythread(2, "BlockUpdater")
{
}


BlockUpdater::
        ~BlockUpdater()
{
    sync ();
}


void BlockUpdater::
        processJobs( queue<UpdateQueue::Job>& jobs )
{
    // Select subset to work on, must consume jobs in order
    vector<UpdateQueue::Job> myjobs;
    while (!jobs.empty ())
    {
        UpdateQueue::Job& j = jobs.front ();
        if (dynamic_cast<const TfrBlockUpdater::Job*>(j.updatejob.get ()))
        {
            myjobs.push_back (std::move(j)); // Steal it
            jobs.pop ();
        }
        else
            break;
    }

    // Begin chunk transfer to gpu right away
#ifndef GL_ES_VERSION_2_0
    // PBOs are not supported on OpenGL ES (< 3.0)
    unordered_map<Tfr::pChunk,lazy<Source2Pbo>> source2pbo;
    for (const UpdateQueue::Job& j : myjobs)
    {
        auto job = dynamic_cast<const TfrBlockUpdater::Job*>(j.updatejob.get ());

        Source2Pbo sp(job->chunk);
        memcpythread.addTask (sp.transferData(job->p));

        source2pbo[job->chunk] = move(sp);
    }
#endif

    // Begin transfer of vbo data to gpu
    unordered_map<Texture2Fbo::Params, lazy<Texture2Fbo>> vbos;
    for (const UpdateQueue::Job& j : myjobs)
    {
        auto job = dynamic_cast<const TfrBlockUpdater::Job*>(j.updatejob.get ());

        if (!j.intersecting_blocks.empty ())
        {
            pBlock block = j.intersecting_blocks.front ();
            auto vp = block->visualization_params();

            Texture2Fbo::Params p(job->chunk,
                                  vp->display_scale (),
                                  block->block_layout ());

            // Most vbo's will look the same, only create as many as needed.
            if (vbos.count (p))
                continue;

            vbos[p] = Texture2Fbo(p, job->normalization_factor);
        }
    }

    // Remap block -> chunks (instead of chunk -> blocks) because we want to draw all
    // chunks to each block, instead of each chunk to all blocks.
    //
    // The chunks must be drawn in order, thus a "vector<Tfr::pChunk>" is required
    // to preserve ordering.
    unordered_map<pBlock, vector<Tfr::pChunk>> chunks_per_block;
    for (const UpdateQueue::Job& j : myjobs)
    {
        auto job = dynamic_cast<const TfrBlockUpdater::Job*>(j.updatejob.get ());

        for (pBlock block : j.intersecting_blocks)
            chunks_per_block[block].push_back(job->chunk);
    }

    // Prepare to draw with transferred chunk
    unordered_map<Tfr::pChunk, lazy<Pbo2Texture>> pbo2texture;
#ifdef GL_ES_VERSION_2_0
    for (const UpdateQueue::Job& j : myjobs)
    {
        auto job = dynamic_cast<const TfrBlockUpdater::Job*>(j.updatejob.get ());

        pbo2texture[job->chunk] = Pbo2Texture(p->shaders,
                                            job->chunk,
                                            job->p);
    }
#else
    for (auto& sp : source2pbo)
        pbo2texture[sp.first] = Pbo2Texture(p->shaders,
                                            sp.first,
                                            sp.second->getPboWhenReady());
#endif

    // Draw from all chunks to each block
    for (auto& f : chunks_per_block)
    {
        const pBlock& block = f.first;
        glProjection M;
        auto fbo_mapping = p->fbo2block.begin (block->getRegion (), block->texture (), M);

        for (auto& c : f.second)
        {
            auto vp = block->visualization_params();
            Texture2Fbo::Params p(c, vp->display_scale (), block->block_layout ());

            // If something has changed the vbo is out-of-date, skip this
            if (!vbos.count (p))
            {
                TaskInfo(boost::format("blockupdater: skipping update of block: %s") % block->getRegion ());
                continue;
            }

            auto& vbo = vbos[p];
            int vertex_attrib, tex_attrib;
            auto tex_mapping = pbo2texture[c]->map(
                        vbo->normalization_factor(),
                        vp->amplitude_axis (),
                        M, vertex_attrib, tex_attrib);

            vbo->draw(vertex_attrib, tex_attrib);
            (void)tex_mapping; // suppress warning caused by RAII
        }

        // suppress warning caused by RAII
        (void)fbo_mapping;
    }

#ifndef GL_ES_VERSION_2_0
    source2pbo.clear ();
#endif
    for (UpdateQueue::Job& j : myjobs)
        j.promise.set_value ();
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap


namespace Heightmap {
namespace Update {
namespace OpenGL {

void BlockUpdater::
        test()
{
    // It should update blocks with chunk data
    {

    }
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
