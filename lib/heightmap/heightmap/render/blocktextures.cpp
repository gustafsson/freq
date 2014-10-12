#include "blocktextures.h"
#include "gl.h"
#include "GlException.h"
#include "log.h"

//#define INFO
#define INFO if(0)

namespace Heightmap {
namespace Render {

class BlockTexturesImpl
{
public:
    explicit BlockTexturesImpl(unsigned width, unsigned height, unsigned initialCapacity = 0);
    BlockTexturesImpl(const BlockTexturesImpl&)=delete;
    BlockTexturesImpl&operator=(const BlockTexturesImpl&)=delete;

    void setCapacityHint(unsigned c);
    std::vector<GlTexture::ptr> getUnusedTextures(unsigned count) const;
    GlTexture::ptr get1();

    int getCapacity() const;
    int getUseCount() const;
    unsigned getWidth() const { return width_; }
    unsigned getHeight() const { return height_; }

    static void setupTexture(unsigned name, unsigned width, unsigned height);
    static unsigned allocated_bytes_per_element();

private:
    std::vector<GlTexture::ptr> textures;
    const unsigned width_, height_;

    void setCapacity (unsigned target_capacity);

public:
    static void test();
};


shared_state<BlockTexturesImpl> global_block_textures_impl;


bool BlockTextures::
        isInitialized()
{
    return (bool)global_block_textures_impl;
}


void BlockTextures::
        init(unsigned width, unsigned height, unsigned initialCapacity)
{
    EXCEPTION_ASSERT(!isInitialized ());

    global_block_textures_impl.reset(new BlockTexturesImpl(width,height,initialCapacity));
}


void BlockTextures::
        destroy()
{
    global_block_textures_impl.reset();
}


void BlockTextures::
        setCapacityHint(unsigned c)
{
    global_block_textures_impl->setCapacityHint(c);
}


void BlockTextures::
        gc()
{
    auto w = global_block_textures_impl.write ();
    w->setCapacityHint(2*w->getUseCount());
}


std::vector<GlTexture::ptr> BlockTextures::
        getTextures(unsigned count)
{
    auto w = global_block_textures_impl.write ();
    std::vector<GlTexture::ptr> missing_textures = w->getUnusedTextures(count);

    if (count > missing_textures.size ())
    {
        size_t need_more = count - missing_textures.size ();
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
    return global_block_textures_impl->get1();
}


int BlockTextures::
        getCapacity()
{
    return global_block_textures_impl.read ()->getCapacity();
}


unsigned BlockTextures::
        getWidth()
{
    return global_block_textures_impl.read ()->getWidth();
}


unsigned BlockTextures::
        getHeight()
{
    return global_block_textures_impl.read ()->getHeight();
}

void BlockTextures::
        setupTexture(unsigned name, unsigned width, unsigned height)
{
    BlockTexturesImpl::setupTexture (name,width,height);
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


BlockTexturesImpl::BlockTexturesImpl(unsigned width, unsigned height, unsigned initialCapacity)
    :
      width_(width),
      height_(height)
{
    if (initialCapacity>0)
        setCapacity(initialCapacity);
}


void BlockTexturesImpl::
        setCapacityHint (unsigned c)
{
    if (c < textures.size () && textures.size () < 3*c)
    {
        // ok
        return;
    }

    // not ok, adjust
    setCapacity(c*2);
}


void BlockTexturesImpl::
        setCapacity (unsigned target_capacity)
{
    if (target_capacity < textures.size ())
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
        INFO Log("BlockTextures: discarding %d textures, was=%d, target=%d") % discarded % textures.size () % target_capacity;

        textures.swap (pick);
        return;
    }

    int new_textures = target_capacity - textures.size ();
    INFO Log("BlockTextures: allocating %d new textures (had %d)") % new_textures % textures.size ();
    GLuint t[new_textures];
    glGenTextures (new_textures, t);
    textures.reserve (target_capacity);

    for (int i=0; i<new_textures; i++)
    {
        setupTexture (t[i], width_, height_);

        bool adopt = true; // GlTexture does glDeleteTextures
        textures.push_back (GlTexture::ptr(new GlTexture(t[i], width_, height_, adopt)));
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
    glBindTexture(GL_TEXTURE_2D, name);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    // Compatible with GlFrameBuffer
#ifdef GL_ES_VERSION_2_0
    // https://www.khronos.org/registry/gles/extensions/EXT/EXT_texture_storage.txt
    GlException_SAFE_CALL( glTexStorage2DEXT ( GL_TEXTURE_2D, 1, GL_R16F_EXT, w, h));
#else
    GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, w, h, 0, GL_RED, GL_FLOAT, 0) );
#endif

    glBindTexture(GL_TEXTURE_2D, 0);
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

    setCapacity (getCapacity ()+1);
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
        EXCEPTION_ASSERT_EQUALS(c,20);

        t = block_textures.getUnusedTextures (11);
        EXCEPTION_ASSERT_EQUALS(t.size (),11u);

        auto t1 = block_textures.getUnusedTextures (11);
        EXCEPTION_ASSERT_EQUALS(t1.size (),9u);

        auto t2 = block_textures.getUnusedTextures (11);
        EXCEPTION_ASSERT_EQUALS(t2.size (),0u);

        t.resize (5);

        t2 = block_textures.getUnusedTextures (11);
        EXCEPTION_ASSERT_EQUALS(t2.size (),6u);

        GlException_CHECK_ERROR();
    }
}

} // namespace Render
} // namespace Heightmap
