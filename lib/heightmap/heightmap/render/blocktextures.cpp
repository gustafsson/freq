#include "blocktextures.h"
#include "gl.h"
#include "GlException.h"
#include "log.h"
#include "neat_math.h"
#include "tasktimer.h"
#include "shared_state_traits_backtrace.h"
#include "datastorage.h"
#include "shared_state.h"

#include <QOpenGLContextGroup>

//#define INFO
#define INFO if(0)

//#define INFO_DISCARDED
#define INFO_DISCARDED if(0)

namespace Heightmap {
namespace Render {

/**
 * @brief mipmaps should match how the number of mipmap levels being used
 * in heightmap.frag.
 */
#if defined(GL_ES_VERSION_2_0) && !defined(GL_ES_VERSION_3_0)
// slow GPU
const int BlockTextures::max_level=0;
#else
const int BlockTextures::max_level=5;
#endif


class BlockTexturesImpl
{
public:
    struct shared_state_traits: shared_state_traits_backtrace {
        double timeout() { return 10.0; } // might take a long time if allocating a lot of textures
    };

    explicit BlockTexturesImpl(unsigned width, unsigned height);
    BlockTexturesImpl(const BlockTexturesImpl&)=delete;
    BlockTexturesImpl&operator=(const BlockTexturesImpl&)=delete;

    void setCapacityHint(unsigned c);
    void setCapacity (unsigned target_capacity);
    std::vector<GlTexture::ptr> getUnusedTextures(unsigned count) const;
    GlTexture::ptr get1();

    int getCapacity() const;
    int getUseCount() const;
    unsigned getWidth() const { return width_; }
    unsigned getHeight() const { return height_; }

    static void setupTexture(unsigned name, unsigned width, unsigned height);
    static void setupTexture(unsigned name, unsigned width, unsigned height, int max_level);
    static unsigned allocated_bytes_per_element();

private:
    std::vector<GlTexture::ptr> textures;
    const unsigned width_, height_;

public:
    static void test();
};


unsigned g_width=0, g_height=0;
std::map<void*,shared_state<BlockTexturesImpl>> gbti_map;

shared_state<BlockTexturesImpl>& global_block_textures_impl() {
    void* p = QOpenGLContextGroup::currentContextGroup ();
    shared_state<BlockTexturesImpl>& v = gbti_map[p];

    if (!v) v.reset(new BlockTexturesImpl(g_width,g_height));

    return v;
}


bool BlockTextures::
        isInitialized()
{
    return 0!=g_width && 0!=g_height;
}


void BlockTextures::
        init(unsigned width, unsigned height)
{
    EXCEPTION_ASSERT(!isInitialized ());

    g_width = width;
    g_height = height;
}


void BlockTextures::
        destroy()
{
    void* p = QOpenGLContextGroup::currentContextGroup ();
    gbti_map.erase (p);
    g_width = 0;
    g_height = 0;
}


void BlockTextures::
        setCapacityHint(unsigned c)
{
    global_block_textures_impl()->setCapacityHint(c);
}


void BlockTextures::
        gc(bool aggressive)
{
    auto w = global_block_textures_impl().write ();
    int c = w->getUseCount();

    if (aggressive)
        w->setCapacity(c);
    else
        w->setCapacityHint(c);
}


std::vector<GlTexture::ptr> BlockTextures::
        getTextures(unsigned count)
{
    auto w = global_block_textures_impl().write ();
    std::vector<GlTexture::ptr> missing_textures = w->getUnusedTextures(count);

    if (count > missing_textures.size ())
    {
        size_t need_more = count - missing_textures.size ();
        INFO Log("getTextures: %d, had %d of %d readily available, allocating an additional %d textures")
                % count % missing_textures.size () % w->getCapacity () % need_more;
        w->setCapacityHint (w->getCapacity () + need_more);

        auto T = w->getUnusedTextures(need_more);
        EXCEPTION_ASSERT_EQUALS(T.size(), need_more);

        missing_textures.reserve (count);
        for (GlTexture::ptr& t : T)
            missing_textures.push_back (t);
    }

    return missing_textures;
}


GlTexture::ptr BlockTextures::
        get1()
{
    return global_block_textures_impl()->get1();
}


int BlockTextures::
        getCapacity()
{
    return global_block_textures_impl().read ()->getCapacity();
}


int BlockTextures::
        getUseCount()
{
    return global_block_textures_impl().read ()->getUseCount();
}


unsigned BlockTextures::
        getWidth()
{
    return global_block_textures_impl().read ()->getWidth();
}


unsigned BlockTextures::
        getHeight()
{
    return global_block_textures_impl().read ()->getHeight();
}


void BlockTextures::
        setupTexture(unsigned name, unsigned width, unsigned height)
{
    BlockTexturesImpl::setupTexture (name,width,height);
}


void BlockTextures::
        setupTexture(unsigned name, unsigned width, unsigned height, int max_level)
{
    BlockTexturesImpl::setupTexture (name,width,height,max_level);
}


unsigned BlockTextures::
        allocated_bytes_per_element()
{
    return BlockTexturesImpl::allocated_bytes_per_element ();
}


void BlockTextures::
        test()
{
    BlockTexturesImpl::test ();
}


BlockTexturesImpl::BlockTexturesImpl(unsigned width, unsigned height)
    :
      width_(width),
      height_(height)
{
}


void BlockTexturesImpl::
        setCapacityHint (unsigned c)
{
    size_t S = textures.size ();
    unsigned C = 64; // 16 MB
    unsigned lower_bound = c; // need at least this many
    unsigned preferred = align_down(c,C)+C; // create margin textures
    unsigned upper_bound = align_down(c,C)+2*C; // but more than this is unnecessary

    if (lower_bound <= S && S <= upper_bound)
        return;

    INFO Log("hint: %d, textures.size (): %d") % c % textures.size ();
    setCapacity(preferred);
}


void BlockTexturesImpl::
        setCapacity (unsigned target_capacity)
{
    if (target_capacity <= textures.size ())
    {
        std::vector<GlTexture::ptr> pick;
        pick.reserve (target_capacity);

        // Keep all textures that are still in use
        for (const GlTexture::ptr& p : textures)
            if (!p.unique ())
                pick.push_back (p);

        // Keep up to 'target_capacity' textures
        for (const GlTexture::ptr& p : textures)
            if (p.unique () && pick.size ()<target_capacity)
                pick.push_back (p);

        int discarded = textures.size () - pick.size ();
        INFO_DISCARDED Log("BlockTextures: discarding %d textures, was=%d, target=%d") % discarded % textures.size () % target_capacity;

        textures.swap (pick);
        return;
    }

    int new_textures = target_capacity - textures.size ();
    int mipmapfactor = 2;

    std::vector<GLuint> t(new_textures);
    glGenTextures (new_textures, &t[0]);
    INFO Log("BlockTextures: allocating %d new textures from name %d (had %d of which %d were used). %s")
            % new_textures % t[0] % textures.size () % getUseCount()
            % DataStorageVoid::getMemorySizeText (
                textures.size ()*allocated_bytes_per_element()*width_*height_*mipmapfactor);
    textures.reserve (target_capacity);

    for (int i=0; i<new_textures; i++)
    {
        setupTexture (t[i], width_, height_);

        bool adopt = true; // GlTexture does glDeleteTextures
        GlTexture::ptr tp(new GlTexture(t[i], width_, height_, adopt));
        tp->setMinFilter(GL_LINEAR);
        textures.push_back (move(tp));
    }
}


std::vector<GlTexture::ptr> BlockTexturesImpl::
        getUnusedTextures(unsigned count) const
{
    std::vector<GlTexture::ptr> pick;
    pick.reserve (count);

    for (const GlTexture::ptr& p : textures)
        if (p.unique ())
        {
            pick.push_back (p);

            if (pick.size () == count)
                break;
        }

    return pick;
}


int BlockTexturesImpl::
        getCapacity() const
{
    return textures.size ();
}


int BlockTexturesImpl::
        getUseCount() const
{
    int c = 0;
    for (const GlTexture::ptr& p : textures)
        if (!p.unique ())
            c++;
    return c;
}


void BlockTexturesImpl::
        setupTexture(unsigned name, unsigned w, unsigned h)
{
    setupTexture(name, w, h, BlockTextures::max_level);
}


void BlockTexturesImpl::
        setupTexture(unsigned name, unsigned w, unsigned h, int max_level)
{
    glBindTexture(GL_TEXTURE_2D, name);
    // Compatible with GlFrameBuffer

    EXCEPTION_ASSERT_LESS(0u,w);
    EXCEPTION_ASSERT_LESS(0u,h);

    max_level = std::min(max_level,(int)log2(std::max(w,h)));

#if defined(GL_ES_VERSION_2_0)
    // https://www.khronos.org/registry/gles/extensions/EXT/EXT_texture_storage.txt
    #ifdef GL_ES_VERSION_3_0
        GlException_SAFE_CALL( glTexStorage2D ( GL_TEXTURE_2D, max_level+1, GL_R16F, w, h));
    #else
        GlException_SAFE_CALL( glTexStorage2DEXT ( GL_TEXTURE_2D, max_level+1, GL_R16F_EXT, w, h));
    #endif
#else
//    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE); // GL 1.4
    // must create all mipmap levels for mipmapping to be complete
    for (int i=0; i<=max_level; i++)
    {
        int wi = std::max(1u, w>>i);
        int hi = std::max(1u, h>>i);
        GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D, i, GL_R16F, wi, hi, 0, GL_RED, GL_FLOAT, 0) );
        if (wi==1 || hi==1)
            break;
    }

//    GlException_SAFE_CALL( glTexStorage2D (GL_TEXTURE_2D, mipmaplevels, GL_R16F, w, h) ); // GL 4.2

#endif
    //    glGenerateMipmap (GL_TEXTURE_2D); // another way to make sure all mipmaps are allocated, but this allocates more than necessary

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, max_level );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    // change to a mipmapping filter when mipmaps are built later
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
}


unsigned BlockTexturesImpl::
        allocated_bytes_per_element()
{
#ifdef GL_ES_VERSION_2_0
    return 2; // GL_R16F_EXT
#else
    return 2; // GL_R16F
#endif
}


GlTexture::ptr BlockTexturesImpl::
        get1()
{
    auto v = getUnusedTextures(1);
    if (!v.empty ())
        return v[0];

    setCapacityHint (getCapacity ()+1);
    v = getUnusedTextures(1);
    if (!v.empty ())
        return v[0];

    return GlTexture::ptr();
}


} // namespace Render
} // namespace Heightmap

#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget
#include "exceptionassert.h"
#include "trace_perf.h"

namespace Heightmap {
namespace Render {

void BlockTexturesImpl::
        test()
{
    std::string name = "BlockTextures";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should keep track of all OpenGL textures that have been allocated for
    // painting block textures and provide already allocated textures fast.
    {
        GlException_CHECK_ERROR();

        BlockTexturesImpl block_textures(512,512);

        int c = block_textures.getCapacity ();

        EXCEPTION_ASSERT_EQUALS(c,0);

        auto t = block_textures.getUnusedTextures (10);
        EXCEPTION_ASSERT_EQUALS(t.size (),0u);

        block_textures.setCapacityHint (10);

        TRACE_PERF("It should provide already allocated textures fast");
        c = block_textures.getCapacity ();
        EXCEPTION_ASSERT_EQUALS(c,64);

        t = block_textures.getUnusedTextures (55);
        EXCEPTION_ASSERT_EQUALS(t.size (),55u);

        auto t1 = block_textures.getUnusedTextures (11);
        EXCEPTION_ASSERT_EQUALS(t1.size (),9u);

        auto t2 = block_textures.getUnusedTextures (11);
        EXCEPTION_ASSERT_EQUALS(t2.size (),0u);

        t1.resize (3); // shrink from 9 to 3, making 6 textures unused
        t2 = block_textures.getUnusedTextures (11);
        EXCEPTION_ASSERT_EQUALS(t2.size (),6u);

        GlException_CHECK_ERROR();
    }
}

} // namespace Render
} // namespace Heightmap
