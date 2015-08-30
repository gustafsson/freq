#include "drawwatermark.h"

#include "gl.h"

// Gpumisc
#include "GlTexture.h"
#include "glPushContext.h"
#include "GlException.h"
#include "tasktimer.h"
#include "Statistics.h"
#include "vbo.h"

// Qt
#include <QImage>

namespace Tools {
namespace Support {

boost::shared_ptr<GlTexture> DrawWatermark::img;
boost::shared_ptr<Vbo> DrawWatermark::postexvbo;

void DrawWatermark::
        loadImage()
{
    if (!img)
    {
        TaskTimer tt("%s", __FUNCTION__);

        GlException_CHECK_ERROR();

        QImage data(":/icons/muchdifferent.png");
        tt.info("format = %d", data.format());
        QImage::Format expected_ftm = QImage::Format_ARGB32;
        if (data.format() != expected_ftm)
        {
            tt.info("changing format to %d", expected_ftm);
            data = data.convertToFormat(expected_ftm);
        }


        {
            std::vector<unsigned char> swizzled(data.byteCount());

            QRgb*ptr = (QRgb*)data.bits();
            for (int y=0; y<data.height(); ++y) for (int x=0; x<data.width(); ++x)
            {
                unsigned o = y*data.width() + x;
                const QRgb& p = ptr[o];
                swizzled[4*o + 0] = qRed(p);
                swizzled[4*o + 1] = qGreen(p);
                swizzled[4*o + 2] = qBlue(p);
                swizzled[4*o + 3] = qAlpha(p);
            }

            img.reset(new GlTexture(data.width(), data.height(), GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, &swizzled[0]));
        }


        postexvbo.reset(new Vbo( 2*4*2*sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW ));

        GlState::glBindBuffer(postexvbo->vbo_type(), *postexvbo);
        float *p = (float *) glMapBuffer(postexvbo->vbo_type(), GL_WRITE_ONLY);

        for (int y=0; y<2; ++y) for (int x=0; x<2; ++x)
        {
            *p++ = x;
            *p++ = 1-y;
//            *p++ = 0.f;
//            *p++ = 1.f;

            *p++ = x*img->getWidth();
            *p++ = y*img->getHeight();
//            *p++ = 0.f;
//            *p++ = 1.f;
        }

        glUnmapBuffer(postexvbo->vbo_type());
        GlState::glBindBuffer(postexvbo->vbo_type(), 0);
    }
}


void DrawWatermark::
        drawWatermark(int viewport_width, int viewport_height)
{
    loadImage();

#ifdef LEGACY_OPENGL
    glPushAttribContext push_attribs;

    glPushMatrixContext push_proj( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0, viewport_width, 0, viewport_height, -1, 1);

    GlState::glDisable(GL_DEPTH_TEST);

    glPushMatrixContext push_model( GL_MODELVIEW );

    glLoadIdentity();

    GlState::glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    GlState::glDisable(GL_COLOR_MATERIAL);
    glColor4f(1,1,1,1);

    {
        GlTexture::ScopeBinding bindTexture = img->getScopeBinding();

        GlState::glBindBuffer(GL_ARRAY_BUFFER, *postexvbo);
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glTexCoordPointer(2, GL_FLOAT, sizeof(float)*4, 0);
        glVertexPointer(2, GL_FLOAT, sizeof(float)*4, (float*)0 + 2);

        GlState::glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        glDisableClientState(GL_VERTEX_ARRAY);
        GlState::glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    GlState::glEnable (GL_DEPTH_TEST);
    GlState::glDisable (GL_BLEND);

    GlException_CHECK_ERROR();
#else
    EXCEPTION_ASSERTX(false, "requires LEGACY_OPENGL");
#endif // LEGACY_OPENGL
}

} // namespace Support
} // namespace Tools
