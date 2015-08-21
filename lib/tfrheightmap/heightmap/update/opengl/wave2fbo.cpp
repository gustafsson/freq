#include "wave2fbo.h"
#include "glstate.h"
#include "GlException.h"
#include "cpumemorystorage.h"
#include "tasktimer.h"
#include "log.h"
#include "heightmap/render/shaderresource.h"
#include "glgroupmarker.h"

#include <QOpenGLShaderProgram>

using namespace std;

namespace Heightmap {
namespace Update {
namespace OpenGL {

Wave2Fbo::
        Wave2Fbo()
    :
      //N(128*1024) // 1 MB
      N_(8*1024) // 64 KB
{
}


function<bool(const glProjection& glprojection)> Wave2Fbo::
        prep(Signal::pMonoBuffer b)
{
    GlGroupMarker gpm("Wave2Fbo");

    if (!program_) {
        program_ = ShaderResource::loadGLSLProgramSource (
                                           R"vertexshader(
                                               attribute highp vec4 vertices;
                                               uniform highp mat4 ModelViewProjectionMatrix;
                                               void main() {
                                                   gl_Position = ModelViewProjectionMatrix*vertices;
                                               }
                                           )vertexshader",
                                           R"fragmentshader(
                                               uniform lowp vec4 rgba;
                                               void main() {
                                                   gl_FragColor = rgba*100.0;
                                               }
                                           )fragmentshader");

        uniModelViewProjectionMatrix_ = program_->uniformLocation("ModelViewProjectionMatrix");
        uniRgba_ = program_->uniformLocation("rgba");
    }

    if (!program_->isLinked ())
        return [](const glProjection& glprojection){return true;};

    GlException_CHECK_ERROR();

    int S = b->number_of_samples ();

    NewVbo gotVbo = getVbo();
    shared_ptr<Vbo> first_vbo = move(gotVbo.second);

    glBindBuffer(GL_ARRAY_BUFFER, *first_vbo);
    vertex_format_xy* d;
    if (gotVbo.first)
    {
        // map entire buffer when it's first allocated to prevent warnings that parts of the buffer to contain uninitialized buffer data
        // OpenGL ES doesn't have glMapBuffer, but it does have glMapBufferRange
        d = (vertex_format_xy*)glMapBufferRange(GL_ARRAY_BUFFER, 0, N_*sizeof(vertex_format_xy), GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_WRITE_BIT);
    }
    else
    {
        d = (vertex_format_xy*)glMapBufferRange(GL_ARRAY_BUFFER, 0, min(N_,4+S)*sizeof(vertex_format_xy), GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT);
    }

    // Prepare clear rectangle
    *d++ = vertex_format_xy{ 0, -1 };
    *d++ = vertex_format_xy{ float(S-1), -1 };
    *d++ = vertex_format_xy{ 0, 1 };
    *d++ = vertex_format_xy{ float(S-1), 1 };
    int first_j = 4;

    float* p = CpuMemoryStorage::ReadOnly<1>(b->waveform_data()).ptr ();

    int i=0;
    for (; i<S && first_j<N_; ++i, ++first_j)
        *d++ = vertex_format_xy{ float(i), p[i] };

    glUnmapBuffer(GL_ARRAY_BUFFER);

    // glMapBufferRange might cause implicit synchronization (wait for
    // previous drawArrays to finish) if it updates the actual buffer
    // right away instead of enqueing the data for update later.
    // Allocating a new buffer is better, it's generic and fast.
    // And OpenGL will free the previous buffer later.
    // https://www.opengl.org/wiki/Buffer_Object_Streaming
    //glBufferData (GL_ARRAY_BUFFER, N * sizeof(vertex_format_xy), NULL, GL_STREAM_DRAW);
    vector<pair<shared_ptr<Vbo>,int>> vbos;

    for (; i<S;)
    {
        --i;
        int j=0;
        NewVbo gotvbo = getVbo();
        shared_ptr<Vbo> vbo = move(gotvbo.second);
        Log("wave2fbo: full, restarting %d") % int(*vbo);

        glBindBuffer(GL_ARRAY_BUFFER, *vbo);
        if (gotVbo.first) // see gotVbo.first above
            d = (vertex_format_xy*)glMapBufferRange(GL_ARRAY_BUFFER, 0, N_*sizeof(vertex_format_xy), GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_WRITE_BIT);
        else
            d = (vertex_format_xy*)glMapBufferRange(GL_ARRAY_BUFFER, 0, min(N_,S-i)*sizeof(vertex_format_xy), GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT);

        for (; i<S && j<N_; ++i, ++j)
            *d++ = vertex_format_xy{ float(i), p[i] };

        glUnmapBuffer(GL_ARRAY_BUFFER);

        vbos.push_back (pair<shared_ptr<Vbo>,int>(vbo,j));
    }

    std::shared_ptr<QOpenGLShaderProgram> program_ = this->program_;
    auto uniModelViewProjectionMatrix = this->uniModelViewProjectionMatrix_;
    auto uniRgba = this->uniRgba_;

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GlException_CHECK_ERROR();

    return [program_,b,uniModelViewProjectionMatrix,uniRgba,
            first_vbo,first_j,vbos]
    (const glProjection& P)
    {
        program_->bind();
        GlState::glEnableVertexAttribArray (0);

        matrixd modelview = P.modelview;
        modelview *= matrixd::translate (b->start (), 0.5, 0);
        modelview *= matrixd::scale (1.0/b->sample_rate (), 0.5, 1);
        glUniformMatrix4fv (uniModelViewProjectionMatrix, 1, false, GLmatrixf(P.projection*modelview).v ());

        GlException_CHECK_ERROR();

        // Draw clear rectangle
        program_->setUniformValue(uniRgba, QVector4D(0.0,0.0,0.0,1.0));
        glBindBuffer(GL_ARRAY_BUFFER, *first_vbo);
        glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 0, 0);
        GlState::glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);

        // Draw waveform
        glLineWidth(1);
        program_->setUniformValue(uniRgba, QVector4D(0.25,0.0,0.0,1.0));
        GlState::glDrawArrays(GL_LINE_STRIP, 4, first_j-4);

        for (auto& v : vbos) {
            glBindBuffer(GL_ARRAY_BUFFER, *v.first);
            glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 0, 0);
            glDrawArrays(GL_LINE_STRIP, 0, v.second);
        }

        GlState::glDisableVertexAttribArray (0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        GlException_CHECK_ERROR();

        return true;
    };
}


Wave2Fbo::NewVbo Wave2Fbo::getVbo ()
{
    shared_ptr<Vbo> r;
    size_t used = 0;
    for (auto& v : vbos_) {
        if (v.unique())
            r = v;
        else
            used++;
    }

    if (r)
    {
        size_t maxsize = used*2+2;
        if (maxsize < vbos_.size ())
        {
            // doesn't matter if dropping vbos in use here as their shared_ptr will release them later
            vbos_.resize (maxsize);
        }
        return NewVbo(false,r);
    }

    r.reset(new Vbo(N_ * sizeof(vertex_format_xy), GL_ARRAY_BUFFER, GL_STREAM_DRAW, NULL));
#if GL_EXT_debug_label
    GlException_SAFE_CALL( glLabelObjectEXT(GL_BUFFER_OBJECT_EXT, *r, 0, "Wave2Fbo") );
#endif
    vbos_.push_back (r);
    return NewVbo(true,r);
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
