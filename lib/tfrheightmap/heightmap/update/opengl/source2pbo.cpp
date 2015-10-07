#include "source2pbo.h"
#include "GlException.h"
#include "tasktimer.h"
#include "gl.h"
#include "log.h"
#include "glstate.h"

#ifdef LEGACY_OPENGL

//#define LOG_TRANSFER_RATE
#define LOG_TRANSFER_RATE if(0)

namespace Heightmap {
namespace Update {
namespace OpenGL {

Source2Pbo::Source2Pbo(
        Tfr::pChunk chunk, bool f32
        )
    :
        chunk_(chunk->transform_data),
        n(DataAccessPosition_t(chunk->transform_data->numberOfElements () * (f32?sizeof(float):sizeof(int16_t)))),
        f32_(f32),
        mapped_chunk_data_(0),
        chunk_pbo_(0)
{
    EXCEPTION_ASSERT (chunk);

    setupPbo ();

    EXCEPTION_ASSERT (chunk_pbo_);
}


Source2Pbo::~Source2Pbo()
{
    if (!QOpenGLContext::currentContext ()) {
        Log ("%s: destruction without gl context leaks pbo %d") % __FILE__ % (unsigned)chunk_pbo_;
        return;
    }

    if (mapped_chunk_data_)
    {
        Log("~Source2Pbo: is waiting for data_transfer before releasing gl resources");
        finishTransfer();
    }

    if (chunk_pbo_)
        GlState::glDeleteBuffers (1, &chunk_pbo_);
}


std::packaged_task<void()> Source2Pbo::
        transferData(void *p)
{
    // http://www.seas.upenn.edu/~pcozzi/OpenGLInsights/OpenGLInsights-AsynchronousBufferTransfers.pdf

    Timer t;
    GlState::glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_);
    mapped_chunk_data_ = (void*)glMapBuffer (GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    GlState::glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (t.elapsed () > 0.001)
        Log("Source2Pbo: It took %s to map chunk_pbo") % TaskTimer::timeToString (t.elapsed ());

    EXCEPTION_ASSERTX(mapped_chunk_data_, boost::format("Source2Pbo: failed glMapBuffer %s") % DataStorageVoid::getMemorySizeText (n));

    void *c = mapped_chunk_data_;
    int n = this->n;
    auto task = std::packaged_task<void()>([c, p, n](){
        Timer t;
        memcpy(c, p, n);
        LOG_TRANSFER_RATE Log("Source2Pbo: memcpy %s with %s/s") % DataStorageVoid::getMemorySizeText(n*sizeof(float)) % DataStorageVoid::getMemorySizeText(n*sizeof(float) / t.elapsed ());
    });

    data_transfer = task.get_future();
    return task;
}


void Source2Pbo::
        finishTransfer()
{
    if (mapped_chunk_data_)
    {
        if (data_transfer.wait_for(std::chrono::milliseconds(10)) != std::future_status::ready)
        {
            TaskTimer tt(boost::format("Source2pbo: waiting to transfer %s") % DataStorageVoid::getMemorySizeText (n,0));
            data_transfer.wait ();
        }

        GlState::glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_);
        glUnmapBuffer (GL_PIXEL_UNPACK_BUFFER);
        GlState::glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        mapped_chunk_data_ = 0;
    }
}


void Source2Pbo::
        setupPbo ()
{
    glGenBuffers (1, &chunk_pbo_); // Generate 1 buffer
    GlState::glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, n, 0, GL_STREAM_DRAW);
    GlState::glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    GlException_CHECK_ERROR();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
#endif
