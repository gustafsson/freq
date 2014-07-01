#ifndef HEIGHTMAPVBO_H
#define HEIGHTMAPVBO_H

#include "heightmap/blocklayout.h"

// gpumisc
#include "mappedvbo.h"
#include "ThreadChecker.h"
#include "GlTexture.h"
#include "glsyncobjectmutex.h"
#include "shared_state.h"

//#define BLOCK_INDEX_TYPE GL_UNSIGNED_SHORT
//#define BLOCKindexType GLushort
#define BLOCK_INDEX_TYPE GL_UNSIGNED_INT
#define BLOCKindexType GLuint


class shared_state_glmutex: public GlSyncObjectMutex {
public:
    void lock_shared() { lock(); }
    bool try_lock() { return false; }
    bool try_lock_shared() { return false; }
    void unlock_shared() { unlock(); }

    bool try_lock_for(...) { lock(); return true; }
    bool try_lock_shared_for(...) { lock_shared(); return true; }
};


namespace Heightmap {
namespace Render {

class GlBlock
{
public:
    struct shared_state_traits : shared_state_traits_default {
        typedef shared_state_glmutex shared_state_mutex;
        struct enable_implicit_lock : std::false_type {};
    };
    typedef shared_state<GlBlock> ptr;

    GlBlock( GlTexture::ptr tex );

    void            draw( unsigned vbo_size );

    GlTexture::ptr  glTexture();
    void            updateTexture( float*p, int n );
    bool            has_texture() const;
    unsigned        allocated_bytes_per_element() const;

private:
    GlTexture::ptr tex_;
    unsigned _tex_height;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAPVBO_H

