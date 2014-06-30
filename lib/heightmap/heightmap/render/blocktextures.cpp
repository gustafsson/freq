#include "blocktextures.h"
#include "gl.h"
#include "GlException.h"
#include "log.h"

//#define INFO
#define INFO if(0)

namespace Heightmap {
namespace Render {

BlockTextures::BlockTextures(BlockLayout bl)
    :
      block_layout(bl)
{
}


void BlockTextures::
        setCapacityHint (unsigned c)
{
    if (c < textures.size () && textures.size () < 3*c)
    {
        // ok
        return;
    }

    // not ok, adjust
    unsigned target_capacity = c*2;
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
        INFO Log("BlockTextures: discarding %d textures") % discarded;

        textures.swap (pick);
        return;
    }

    int new_textures = target_capacity - textures.size ();
    INFO Log("BlockTextures: allocating %d new textures") % new_textures;
    GLuint t[new_textures];
    glGenTextures (new_textures, t);
    textures.reserve (target_capacity);
    int w = block_layout.texels_per_row ();
    int h = block_layout.texels_per_column ();

    for (int i=0; i<new_textures; i++)
    {
        setupTexture (t[i], w, h);

        textures.push_back (GlTexture::ptr(new GlTexture(t[i])));
    }
}


std::vector<GlTexture::ptr> BlockTextures::
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


int BlockTextures::
        getCapacity() const
{
    return textures.size ();
}


void BlockTextures::
        setupTexture(unsigned name, unsigned w, unsigned h)
{
    glBindTexture(GL_TEXTURE_2D, name);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    // Not compatible with GlFrameBuffer
    //static bool hasTextureFloat = 0 != strstr( (const char*)glGetString(GL_EXTENSIONS), "GL_ARB_texture_float" );
    //GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D,0,hasTextureFloat?GL_LUMINANCE32F_ARB:GL_LUMINANCE,w, h,0, hasTextureFloat?GL_LUMINANCE:GL_RED, GL_FLOAT, 0) );

    // Compatible with GlFrameBuffer
    //GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, 0) );
    GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, w, h, 0, GL_RED, GL_FLOAT, 0) );
    //GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RED, GL_FLOAT, 0) );

    glBindTexture(GL_TEXTURE_2D, 0);
}


} // namespace Render
} // namespace Heightmap

#include <QApplication>
#include <QGLWidget>
#include "exceptionassert.h"
#include "trace_perf.h"

namespace Heightmap {
namespace Render {

void BlockTextures::
        test()
{
    std::string name = "BlockTextures";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // The BlockTextures class should keep track of all OpenGL textures that
    // have been allocated for painting GlBlocks and provide already allocated
    // textures fast
    {
        GlException_CHECK_ERROR();

        BlockLayout block_layout(512,512,1);
        BlockTextures block_textures(block_layout);

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
