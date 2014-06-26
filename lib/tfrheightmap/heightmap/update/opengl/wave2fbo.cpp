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
      b_(b)
{}


void Wave2Fbo::
        draw()
{
//    TaskTimer tt(boost::format("Wave2Fbo::draw %s") % b_->getInterval ());

    struct vertex_format {
        float x, y;
    };

    float t0 = b_->start ();
    int N = b_->number_of_samples ();
    float ifs = 1.0f/b_->sample_rate ();
    float t1 = t0 + N*ifs;

    std::vector<vertex_format> d(N);
    float* p = CpuMemoryStorage::ReadOnly<1>(b_->waveform_data()).ptr ();
    for (int i=0; i<N; i++)
        d[i] = vertex_format{ t0 + i*ifs, 0.5f + 0.5f*p[i] };

    GlException_CHECK_ERROR();

    // Draw clear rectangle
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor4f(0.f, 0.f, 0.f, 1.0f);
    std::vector<vertex_format> c(4);
    c[0] = vertex_format{ t0, 0 };
    c[1] = vertex_format{ t1, 0 };
    c[2] = vertex_format{ t0, 1 };
    c[3] = vertex_format{ t1, 1 };
    glVertexPointer(2, GL_FLOAT, 0, &c[0]);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Draw waveform
    glEnable (GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(.5f, 0.f, 0.f, 0.001f);
    glVertexPointer(2, GL_FLOAT, 0, &d[0]);
    glDrawArrays(GL_LINE_STRIP, 0, N);
    glDisable (GL_BLEND);
    glDisableClientState(GL_VERTEX_ARRAY);

    GlException_CHECK_ERROR();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
