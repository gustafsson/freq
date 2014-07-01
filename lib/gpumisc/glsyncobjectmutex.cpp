#include "glsyncobjectmutex.h"
#include "gl.h"
#include "tasktimer.h"
#include "log.h"

#include <mutex>
#include <thread>

class GlSyncObjectMutexPrivate {
public:
    GLsync s = 0;
    std::thread::id id;
    std::mutex m;
    bool clientsync = false;
};

GlSyncObjectMutex::
        GlSyncObjectMutex(bool clientsync)
    :
      p(new GlSyncObjectMutexPrivate)
{
    p->id = std::this_thread::get_id ();
    p->clientsync = clientsync;
}


GlSyncObjectMutex::
        ~GlSyncObjectMutex()
{
    if (p->s)
        glDeleteSync (p->s);
    delete p;
}


void GlSyncObjectMutex::
        lock()
{
    p->m.lock ();

    if (p->s)
    {
        if (std::this_thread::get_id () != p->id)
        {
            GLint result = GL_UNSIGNALED;
            glGetSynciv(p->s, GL_SYNC_STATUS, sizeof(GLint), NULL, &result);
            if (GL_UNSIGNALED == result)
            {
                TaskTimer tt("GlSyncObjectMutex: waiting");

                //glFlush (); // Must make sure that the sync object is in the queue
                //glFlush must be called prior to glWaitSync according to the spec. But
                //the sync object is created in another thread, which will flush.
                if (p->clientsync)
                    glClientWaitSync (p->s, 0, GL_TIMEOUT_IGNORED);
                else
                    glWaitSync (p->s, 0, GL_TIMEOUT_IGNORED);
            }
        }

        Log("lock: p->s = %s") % p->s;
        glDeleteSync (p->s);
        p->s = 0;
    }
}


void GlSyncObjectMutex::
        unlock()
{
    p->s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    if (std::this_thread::get_id () != p->id)
        p->id = std::this_thread::get_id ();
    Log("unlock: p->s = %s") % p->s;
    p->m.unlock ();
}


#include <QApplication>
#include <QGLWidget>
#include "exceptionassert.h"
#include "barrier.h"
#include "GlException.h"

using namespace std;

void GlSyncObjectMutex::
        test()
{
    std::string name = "GlSyncObjectMutex";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget p;
    QGLWidget w(0, &p);
    w.makeCurrent ();
    std::thread t;


    // It should provide a mutex mechanism for OpenGL resources.
    try {
        unsigned vbo;
        GlException_SAFE_CALL( glGenBuffers (1, &vbo) );

        unsigned texture;
        GlException_SAFE_CALL( glGenTextures (1, &texture) );

        GlSyncObjectMutex m(true);
        spinning_barrier barrier(2);

        // Most predictable benefit for large data sets, small datasets "might" work anyways
        int width = 1024;
        int height = 1024;
        const float N = 4*width*height;
        vector<float> result(N);
        vector<float> texture_init(N);
        vector<float> texture_update(N);

        auto resetVbosAndTexture = [&]()
        {
            for (unsigned i=0; i<N; i++)
            {
                result[i] = 1 + i;
                texture_init[i] = 2 + i;
                texture_update[i] = 3 + i;
            }

            GlException_SAFE_CALL( glBindBuffer (GL_ARRAY_BUFFER, vbo) );
            GlException_SAFE_CALL( glBufferData (GL_ARRAY_BUFFER, sizeof(float)*N, &texture_update[0], GL_DYNAMIC_COPY) );
            GlException_SAFE_CALL( glBindBuffer (GL_ARRAY_BUFFER, 0) );

            GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, texture) );
            GlException_SAFE_CALL( glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, &texture_init[0]) );
            GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, 0) );
        };

        auto copyFromVboToTextureAsync = [&]()
        {
            GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, texture) );
            GlException_SAFE_CALL( glBindBuffer (GL_PIXEL_UNPACK_BUFFER, vbo) );
            GlException_SAFE_CALL( glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, 0) );
            GlException_SAFE_CALL( glBindBuffer (GL_PIXEL_UNPACK_BUFFER, 0) );
            GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, 0) );
        };

        auto copyFromTextureToResult = [&]()
        {
            GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, texture) );
            GlException_SAFE_CALL( glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, &result[0]) );
            GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, 0) );
        };

        // Start a separate thread, and read data from this thread.
        auto copyFromVboToTextureThread = [&](bool dolock)
        {
            QGLWidget w2(0, &w);
            w2.makeCurrent ();

            if (dolock)
            {
                std::lock_guard<GlSyncObjectMutex> l(m);
                barrier.wait ();
                copyFromVboToTextureAsync();
                (void)l; // RAII
            }
            else
            {
                copyFromVboToTextureAsync();
                barrier.wait ();
            }

            glFlush(); // Hand over all OpenGL commands to the gpu before destroying thread
        };


        // Take 0. verify setup
        {
            resetVbosAndTexture();
            copyFromTextureToResult();
            EXCEPTION_ASSERT_EQUALS( 0, memcmp(&result[0], &texture_init[0], N*sizeof(float)));
        }

        // Take 1. single threaded
        {
            resetVbosAndTexture();

            // No lock needed if the transfers happen on the same thread, they
            // will queue up in order and glGetTexImage won't return until the
            // transfer is finished.
            copyFromVboToTextureAsync();
            copyFromTextureToResult();
            EXCEPTION_ASSERT_EQUALS( 0, memcmp(&result[0], &texture_update[0], N*sizeof(float)));
        }


        // Take 2. multi-threaded, synchronized on cpu
        {
            resetVbosAndTexture();
            copyFromTextureToResult();
            EXCEPTION_ASSERT_EQUALS( 0, memcmp(&result[0], &texture_init[0], N*sizeof(float)));

            t = std::thread(copyFromVboToTextureThread, false);

            barrier.wait ();
            copyFromTextureToResult();
            t.join ();

            // should not have gotten all data from the other thread even though
            // the gl commands are issued (thanks to barrier.wait())
            EXCEPTION_ASSERT_NOTEQUALS( result[0], texture_update[0]);
            EXCEPTION_ASSERT_NOTEQUALS( result[N-1], texture_update[N-1]);
            EXCEPTION_ASSERT_EQUALS( result[0], texture_init[0]);
            EXCEPTION_ASSERT_EQUALS( result[N-1], texture_init[N-1]);
        }


        // Take 3. multi-threaded, synchronized on gpu
        {
            resetVbosAndTexture();
            copyFromTextureToResult();
            EXCEPTION_ASSERT_EQUALS( 0, memcmp(&result[0], &texture_init[0], N*sizeof(float)));

            t = std::thread(copyFromVboToTextureThread, true);

            barrier.wait ();

            {
                // Read samples
                std::lock_guard<GlSyncObjectMutex> l(m);
                copyFromTextureToResult();
                (void)l; // RAII
            }

            t.join ();

            // m.lock will make sure not only that glBufferData has returned but
            // also that the OpenGL command has been completely processed

            // This time the transfer has finished
            EXCEPTION_ASSERT_EQUALS( result[0], texture_update[0]);
            EXCEPTION_ASSERT_EQUALS( result[N-1], texture_update[N-1]);
        }
    } catch (...) {
        if (t.joinable ())
            t.join ();
        throw;
    }
}
