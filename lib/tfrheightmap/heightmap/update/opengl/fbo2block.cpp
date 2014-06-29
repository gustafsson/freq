#include "fbo2block.h"
#include "heightmap/render/glblock.h"
#include "heightmap/uncaughtexception.h"
#include "GlException.h"

#include "gl.h"

namespace Heightmap {
namespace Update {
namespace OpenGL {

void grabToTexture(GlTexture::ptr t)
{
    glBindTexture(GL_TEXTURE_2D, t->getOpenGlTextureId ());
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, t->getWidth (), t->getHeight ());
    glBindTexture(GL_TEXTURE_2D, 0);
}


Fbo2Block::Fbo2Block (pBlock block)
    :
      block(block),
      glblock(block->glblock)
{
    // Draw straight onto glblock
    fbo.reset (new GlFrameBuffer(glblock->glTexture ()->getOpenGlTextureId ()));


//    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
//    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
//                           GL_TEXTURE_2D, tex1, 0);
//    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
//                           GL_TEXTURE_2D, tex2, 0);
//    glDrawBuffer(GL_COLOR_ATTACHMENT1);
//    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
//                      GL_COLOR_BUFFER_BIT, GL_NEAREST);

    // Draw to fbo, copy to glblock
    // Create new texture
/*    unsigned tex_;
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);

    static bool hasTextureFloat = 0 != strstr( (const char*)glGetString(GL_EXTENSIONS), "GL_ARB_texture_float" );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    int w = block->block_layout().texels_per_row();
    int h =  block->block_layout().texels_per_column ();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED,
                 w, h, 0,
                 GL_RED, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    fbo.reset (new GlFrameBuffer(tex_));

    glCopyImageSubData();*/

    {
//        glCopyImageSubData​(glblock->glTexture ()->getOpenGlTextureId (), GL_TEXTURE_2D​,
//                                    0​, 0​, 0​, 0​,
//                                    fbo_->getGlTexture()​, GL_TEXTURE_2D,
//                                    0​, 0​, 0, 0​,
//                                    w, h, 1);
//        unsigned vbo=0;
//        glGenBuffers (1, &vbo);
//        struct vertex_format {
//            float x, y, u, v;
//        };

//        float vertices[] = {
//            0, 0, 0, 0,
//            0, 1, 0, 1,
//            1, 0, 1, 0,
//            1, 1, 1, 1,
//        };

//        glBindBuffer(GL_ARRAY_BUFFER, vbo);
//        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STREAM_DRAW);

//        glEnableClientState(GL_VERTEX_ARRAY);
//        glEnableClientState(GL_TEXTURE_COORD_ARRAY);

//        glVertexPointer(2, GL_FLOAT, sizeof(vertex_format), 0);
//        glTexCoordPointer(2, GL_FLOAT, sizeof(vertex_format), (float*)0 + 2);

//        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4); // Paint new contents over it

//        glDeleteBuffers (1, &vbo);
    }

    // Copy from texture to own fbo
//    {
//        GlFrameBuffer fbo(block->glblock->glTexture ()->getOpenGlTextureId ());
//        fbo_->bindFrameBuffer ();
//        grabToTexture(glblock->glTexture());
//        fbo_->unbindFrameBuffer ();
//    }
}


Fbo2Block::~Fbo2Block()
{
    if (!fbo)
        return;

    try {
//        fbo->bindFrameBuffer ();
//        grabToTexture(block->glblock->glTexture());
//        grabToTexture(block->glblock->glVertTexture());
//        fbo->unbindFrameBuffer ();

        //fbo->bindFrameBuffer ();
        //grabToTexture(glblock->glVertTexture());
        //fbo->unbindFrameBuffer ();
//        glFinish ();
//        block->glblock = glblock;
    } catch (...) {
        Heightmap::UncaughtException::handle_exception(boost::current_exception());
    }
}

GlFrameBuffer::ScopeBinding Fbo2Block::
        begin ()
{
    GlException_CHECK_ERROR ();

    GlFrameBuffer::ScopeBinding fboBinding = fbo->getScopeBinding ();

    Region br = block->getRegion ();
    BlockLayout block_layout = block->block_layout ();

    // Juggle texture coordinates so that border texels are centered on the border
    float dt = br.time (), ds = br.scale ();
    br.a.time -= 0.5*dt / block_layout.texels_per_row ();
    br.b.time += 0.5*dt / block_layout.texels_per_row ();
    br.a.scale -= 0.5*ds / block_layout.texels_per_column ();
    br.b.scale += 0.5*ds / block_layout.texels_per_column ();

    // Setup matrices
    glViewport (0, 0, block_layout.texels_per_row (), block_layout.texels_per_column ());
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    glOrtho (br.a.time, br.b.time, br.a.scale, br.b.scale, -10,10);
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity ();

    GlException_CHECK_ERROR ();

    // Disable unwanted capabilities when resampling a texture
    glDisable (GL_DEPTH_TEST);
    glDisable (GL_BLEND);
    glDisable (GL_CULL_FACE);

    return fboBinding;
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
