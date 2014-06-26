#include "wave2fbo.h"
#include "gl.h"
#include "GlException.h"
#include "cpumemorystorage.h"
#include "tasktimer.h"

namespace Heightmap {
namespace Update {
namespace OpenGL {

Wave2Fbo::
        Wave2Fbo(Signal::pMonoBuffer b)
    :
      b_(b),
      N(b_->number_of_samples ())
{
    struct vertex_format_xy {
        float x, y;
    };

    float t0 = b_->start ();
    float ifs = 1.0f/b_->sample_rate ();
    float t1 = t0 + N*ifs;

    std::vector<vertex_format_xy> d(N + 4);
    float* p = CpuMemoryStorage::ReadOnly<1>(b_->waveform_data()).ptr ();
    for (int i=0; i<N; i++)
        d[i] = vertex_format_xy{ t0 + i*ifs, 0.5f + 0.5f*p[i] };

    d[N+0] = vertex_format_xy{ t0, 0 };
    d[N+1] = vertex_format_xy{ t1, 0 };
    d[N+2] = vertex_format_xy{ t0, 1 };
    d[N+3] = vertex_format_xy{ t1, 1 };

    GlException_SAFE_CALL( glGenBuffers (1, &vbo_) ); // Generate 1 buffer
    GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, vbo_) );
    GlException_SAFE_CALL( glBufferData (GL_ARRAY_BUFFER, sizeof(vertex_format_xy)*d.size (), &d[0], GL_STATIC_DRAW) );
    GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, 0) );
}


Wave2Fbo::
        ~Wave2Fbo()
{
    if (vbo_)
        glDeleteBuffers (1, &vbo_);
}


void Wave2Fbo::
        draw()
{
    TaskTimer tt(boost::format("Wave2Fbo::draw %s") % b_->getInterval ());

    GlException_CHECK_ERROR();
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, 0);

    // Draw clear rectangle
    glColor4f(0.f, 0.f, 0.f, 1.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, N, 4);

    // Draw waveform
    glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(.5f, 0.f, 0.f, 0.001f);
    glDrawArrays(GL_LINE_STRIP, 0, N);
    glDisable (GL_BLEND);

    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GlException_CHECK_ERROR();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
