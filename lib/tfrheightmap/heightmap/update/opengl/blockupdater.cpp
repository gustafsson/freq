#include "blockupdater.h"
#include "heightmap/update/tfrblockupdater.h"
#include "tfr/chunk.h"
#include "heightmap/render/blocktextures.h"
#include "heightmap/blockmanagement/blockupdater.h"

#include "pbo2texture.h"
#include "source2pbo.h"
#include "texture2fbo.h"
#include "texturepool.h"

#include "tasktimer.h"
#include "timer.h"
#include "log.h"
#include "neat_math.h"
#include "lazy.h"
#include "GlException.h"

#include <thread>
#include <future>
#include <unordered_map>

//#define INFO
#define INFO if(0)

#ifndef GL_ES_VERSION_2_0
// the current implementation of PBOs doesn't reuse allocated memory
// #define USE_PBO
#endif

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

int gl_max_texture_size() {
    static int v = 0;
    if (0 == v)
        glGetIntegerv (GL_MAX_TEXTURE_SIZE, &v);
    return v;
}

class BlockUpdaterPrivate
{
public:
    Shaders shaders;

    TexturePool texturePool
    {
        gl_max_texture_size(),
        std::min(gl_max_texture_size(),1024),
        TfrBlockUpdater::Job::type == TfrBlockUpdater::Job::Data_F32
            ? TexturePool::Float32
            : TexturePool::Float16
    };
};

BlockUpdater::
        BlockUpdater()
    :
      p(new BlockUpdaterPrivate),
      memcpythread(1, "BlockUpdater")
{
    p->texturePool.resize (2);
}


BlockUpdater::
        ~BlockUpdater()
{
    delete p;
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
            if (myjobs.size () == p->texturePool.size()) {
                processJobs(myjobs);
                myjobs.clear ();
            }

            myjobs.push_back (std::move(j)); // Steal it
            jobs.pop ();
        }
        else
            break;
    }

    if (!myjobs.empty ())
        processJobs (myjobs);
}


void BlockUpdater::
        processJobs( vector<UpdateQueue::Job>& myjobs )
{
#ifdef USE_PBO
    // Begin chunk transfer to gpu right away
    // PBOs are not supported on OpenGL ES (< 3.0)
    unordered_map<Tfr::pChunk,lazy<Source2Pbo>> source2pbo;
    for (const UpdateQueue::Job& j : myjobs)
    {
        auto job = dynamic_cast<const TfrBlockUpdater::Job*>(j.updatejob.get ());

        Source2Pbo sp(job->chunk, job->type==TfrBlockUpdater::Job::Data_F32);
        memcpythread.addTask (sp.transferData(job->p));

        source2pbo[job->chunk] = move(sp);
    }
#endif

    // Begin transfer of vbo data to gpu
    unordered_map<Texture2Fbo::Params, shared_ptr<Texture2Fbo>> vbos_p;
    unordered_map<pBlock, shared_ptr<Texture2Fbo>> vbos;
    for (const UpdateQueue::Job& j : myjobs)
    {
        auto job = dynamic_cast<const TfrBlockUpdater::Job*>(j.updatejob.get ());

        for (pBlock block : j.intersecting_blocks)
        {
            auto vp = block->visualization_params();

            Texture2Fbo::Params p(job->chunk,
                                  vp->display_scale (),
                                  block->block_layout ());

            // Most vbo's will look the same, only create as many as needed.
            if (0==vbos_p.count (p))
                vbos_p[p].reset(new Texture2Fbo(p, job->normalization_factor));

            vbos[block] = vbos_p[p];
        }
    }

    // Prepare to draw with transferred chunk
    unordered_map<Tfr::pChunk, shared_ptr<Pbo2Texture>> pbo2texture;
#ifndef USE_PBO
    for (const UpdateQueue::Job& j : myjobs)
    {
        auto job = dynamic_cast<const TfrBlockUpdater::Job*>(j.updatejob.get ());

        pbo2texture[job->chunk].reset(new Pbo2Texture(p->shaders,
                                              p->texturePool.get1 (),
                                              job->chunk,
                                              job->p,
                                              job->type == TfrBlockUpdater::Job::Data_F32));
    }
#else
    for (auto& sp : source2pbo)
        pbo2texture[sp.first].reset(new Pbo2Texture(p->shaders,
                                            p->texturePool.get1 (),
                                            sp.first,
                                            sp.second->getPboWhenReady(),
                                            sp.second->f32()));
#endif

    glFlush();

    // Draw to each block
    for (const UpdateQueue::Job& j : myjobs)
    {
        auto job = dynamic_cast<const TfrBlockUpdater::Job*>(j.updatejob.get ());

        for (pBlock block : j.intersecting_blocks)
        {
            packaged_task<bool(const glProjection& M)> f(
                    [
                        shader = pbo2texture[job->chunk],
                        vbo = vbos[block],
                        amplitude_axis = block->visualization_params()->amplitude_axis ()
                    ]
                    (const glProjection& M)
                    {
                        int vertex_attrib, tex_attrib;
                        auto tex_mapping = shader->map(
                                    vbo->normalization_factor(),
                                    amplitude_axis,
                                    M, vertex_attrib, tex_attrib);

                        vbo->draw(vertex_attrib, tex_attrib);
                        (void)tex_mapping; // RAII
                        return true;
                    });

            block->updater ()->queueUpdate (block, move(f));

#ifdef PAINT_BLOCKS_FROM_UPDATE_THREAD
            block->updater ()->processUpdates (false);
#endif
        }
    }

#ifdef USE_PBO
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
