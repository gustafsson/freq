#include "source2pbo.h"
#include "GlException.h"

namespace Heightmap {
namespace Update {
namespace OpenGL {

Source2Pbo::Source2Pbo(
        Tfr::pChunk chunk
        )
    :
        chunk_(chunk->transform_data),
        n(chunk->nScales () * chunk->nSamples ()),
        mapped_chunk_data_(0),
        chunk_pbo_(0)
{
    EXCEPTION_ASSERT (chunk);

    setupPbo ();

    EXCEPTION_ASSERT (chunk_pbo_);
}


Source2Pbo::~Source2Pbo()
{
    if (mapped_chunk_data_)
    {
        TaskInfo("~Source2Pbo is waiting for data_transfer before releasing gl resources");
        finishTransfer();
    }

    if (chunk_pbo_)
        glDeleteBuffers (1, &chunk_pbo_);
}

std::packaged_task<void()> Source2Pbo::
        transferData(float *p)
{
    // http://www.seas.upenn.edu/~pcozzi/OpenGLInsights/OpenGLInsights-AsynchronousBufferTransfers.pdf

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_);
    mapped_chunk_data_ = (float*)glMapBuffer (GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    float *c = mapped_chunk_data_;
    int n = this->n;
    auto t = std::packaged_task<void()>([c, p, n](){
//        Timer t;
        memcpy(c, p, n*sizeof(float));
//        TaskInfo("memcpy %s with %s/s", DataStorageVoid::getMemorySizeText(n*sizeof(float)).c_str (), DataStorageVoid::getMemorySizeText(n*sizeof(float) / t.elapsed ()).c_str ());
    });

    data_transfer = t.get_future();
    return t;
}


void Source2Pbo::
        finishTransfer()
{
    if (mapped_chunk_data_)
    {
        if (!data_transfer.valid ())
        {
            TaskTimer tt("data_transfer.wait () %d", n);
            data_transfer.wait ();
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_);
        glUnmapBuffer (GL_PIXEL_UNPACK_BUFFER);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        mapped_chunk_data_ = 0;
    }
}


void Source2Pbo::
        setupPbo ()
{
    glGenBuffers (1, &chunk_pbo_); // Generate 1 buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(float)*n, 0, GL_STATIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    GlException_CHECK_ERROR();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
