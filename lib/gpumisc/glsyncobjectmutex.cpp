#include "glsyncobjectmutex.h"
#include "gl.h"

#if defined(LEGACY_OPENGL) && !defined(_WIN32)

#include "tasktimer.h"
#include "log.h"
#include "GlException.h"
#include "glstate.h"

#include <mutex>
#include <thread>

using namespace std;

class GlSyncObjectMutexPrivate {
public:
    GLsync s = 0;
    thread::id id;
    mutex m;
    bool clientsync = false;
};

GlSyncObjectMutex::
        GlSyncObjectMutex(bool clientsync)
    :
      p(new GlSyncObjectMutexPrivate)
{
    p->id = this_thread::get_id ();
    p->clientsync = clientsync;
}


GlSyncObjectMutex::
        ~GlSyncObjectMutex()
{
    if (!QOpenGLContext::currentContext ()) {
        Log ("%s: destruction without gl context leaks sync object %d") % __FILE__ % p->s;
        return;
    }

    if (p->s)
        glDeleteSync (p->s);
    delete p;
}


void GlSyncObjectMutex::
        lock()
{
    if (!p->m.try_lock ()) {
        Log("GlSyncObjectMutex: waiting for cpu lock");
        p->m.lock ();
    }

    if (p->s)
    {
        if (this_thread::get_id () != p->id)
        {
            GLint result = GL_UNSIGNALED;
            glGetSynciv(p->s, GL_SYNC_STATUS, sizeof(GLint), NULL, &result);
            if (GL_UNSIGNALED == result)
            {
                if (p->clientsync)
                {
                    TaskTimer tt("GlSyncObjectMutex: waiting for gpu sync");
                    // could use GL_SYNC_FLUSH_COMMANDS_BIT
                    glClientWaitSync (p->s, 0, GL_TIMEOUT_IGNORED);
                }
                else
                {
                    // glWaitSync returns right away, but prevents the driver from adding anything
                    // to the queue before this sync object is signalled.
                    //
                    // glFlush() must be called prior to glWaitSync (according to the spec) to ensure
                    // that the sync object is in the queue. But this sync object is created in
                    // another thread (assumed different context) which will flush its own queue.
                    glWaitSync (p->s, 0, GL_TIMEOUT_IGNORED);
                }
            }
        }

        glDeleteSync (p->s);
        p->s = 0;
    }
}


void GlSyncObjectMutex::
        unlock()
{
    p->s = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    if (this_thread::get_id () != p->id)
        p->id = this_thread::get_id ();
    p->m.unlock ();
}


#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget
#include "exceptionassert.h"
#include "barrier.h"

void GlSyncObjectMutex::
        test()
{
    string name = "GlSyncObjectMutex";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();
    thread t;

    // It should provide a mutex mechanism for OpenGL resources.
    try {
        unsigned vbo;
        GlException_SAFE_CALL( glGenBuffers (1, &vbo) );

        unsigned texture;
        GlException_SAFE_CALL( glGenTextures (1, &texture) );

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

            GlException_SAFE_CALL( GlState::glBindBuffer (GL_ARRAY_BUFFER, vbo) );
            GlException_SAFE_CALL( glBufferData (GL_ARRAY_BUFFER, sizeof(float)*N, &texture_update[0], GL_DYNAMIC_COPY) );
            GlException_SAFE_CALL( GlState::glBindBuffer (GL_ARRAY_BUFFER, 0) );

            GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, texture) );
            GlException_SAFE_CALL( glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, &texture_init[0]) );
            // read with glGetTexImage to create an OpenGL client read state of the texture
            GlException_SAFE_CALL( glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, &result[0]) );
        };

        auto copyFromVboToTextureAsync = [&]()
        {
            GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, texture) );
            GlException_SAFE_CALL( GlState::glBindBuffer (GL_PIXEL_UNPACK_BUFFER, vbo) );
            GlException_SAFE_CALL( glTexSubImage2D  (GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, 0) );
            GlException_SAFE_CALL( GlState::glBindBuffer (GL_PIXEL_UNPACK_BUFFER, 0) );
        };

        auto copyFromTextureToResult = [&]()
        {
            GlException_SAFE_CALL( glBindTexture (GL_TEXTURE_2D, texture) );
            GlException_SAFE_CALL( glGetTexImage (GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, &result[0]) );
        };


        // Run OpenGL commands in a separate thread
        auto openglThread = [&](function<void()> f)
        {
            QGLWidget w2(0, &w);
            w2.makeCurrent ();

            f();

            // Finish all OpenGL commands on the gpu before destroying thread
            glFinish();
        };


        // Take 1. single threaded
        {
            resetVbosAndTexture();
            copyFromVboToTextureAsync();
            copyFromTextureToResult();

            EXCEPTION_ASSERT_EQUALS( 0, memcmp(&result[0], &texture_update[0], N*sizeof(float)));
        }


        // Take 2. multi-threaded, synchronized on cpu
        {
            resetVbosAndTexture();

            t = thread(openglThread, [&](){
                copyFromVboToTextureAsync();
                barrier.wait ();
                glFlush ();
                barrier.wait ();
            });

            barrier.wait ();
            copyFromTextureToResult();

            // should not have gotten all data from the other thread even though
            // the gl commands are issued (thanks to barrier.wait())
            EXCEPTION_ASSERT_NOTEQUALS( result[0], texture_update[0]);
            EXCEPTION_ASSERT_NOTEQUALS( result[N-1], texture_update[N-1]);
            EXCEPTION_ASSERT_EQUALS( result[0], texture_init[0]);
            EXCEPTION_ASSERT_EQUALS( result[N-1], texture_init[N-1]);
            EXCEPTION_ASSERT_EQUALS( 0, memcmp(&result[0], &texture_init[0], N*sizeof(float)));

            barrier.wait ();
            copyFromTextureToResult();
            t.join ();

            // Even after a glFlush() the texture state was not updated because this context
            // has used the texture more recently and discarded the update
            EXCEPTION_ASSERT_EQUALS( 0, memcmp(&result[0], &texture_init[0], N*sizeof(float)));
            EXCEPTION_ASSERT_NOTEQUALS( 0, memcmp(&result[0], &texture_update[0], N*sizeof(float)));
        }


        // Take 3. multi-threaded, synchronized cpu and gpu with glFlush after write
        {
            resetVbosAndTexture();

            t = thread(openglThread, [&](){
                copyFromVboToTextureAsync();
                glFlush ();
                barrier.wait ();
            });

            barrier.wait ();
            copyFromTextureToResult();
            t.join ();

            // the texture update is flushed to the gpu by the other thread
            // before this thread uses the texture

            // This time the transfer has finished
            EXCEPTION_ASSERT_EQUALS( result[0], texture_update[0]);
            EXCEPTION_ASSERT_EQUALS( result[N-1], texture_update[N-1]);
            EXCEPTION_ASSERT_EQUALS( 0, memcmp(&result[0], &texture_update[0], N*sizeof(float)));
        }


        // Take 4. multi-threaded, synchronized cpu and gpu with glClientWait before read, without glFlush
        {
            // unclear if/when GlSyncObjectMutex m(clientsync=false) is useful
            GlSyncObjectMutex m(true);

            resetVbosAndTexture();

            t = thread(openglThread, [&](){
                lock_guard<GlSyncObjectMutex> l(m);
                barrier.wait ();
                copyFromVboToTextureAsync();
                (void)l; // RAII, mark a synchronization point in the command queue
            });

            barrier.wait ();

            {
                lock_guard<GlSyncObjectMutex> l(m);
                copyFromTextureToResult();
                (void)l; // RAII
            }

            t.join ();

            // GlSyncObjectMutex will make sure that when the texture is
            // modified in different threads they first wait for the other
            // update is finished

            // This time the transfer has finished
            EXCEPTION_ASSERT_EQUALS( result[0], texture_update[0]);
            EXCEPTION_ASSERT_EQUALS( result[N-1], texture_update[N-1]);
            EXCEPTION_ASSERT_EQUALS( 0, memcmp(&result[0], &texture_update[0], N*sizeof(float)));
        }

        GlException_SAFE_CALL( GlState::glDeleteBuffers (1, &vbo) );
        GlException_SAFE_CALL( glDeleteTextures (1, &texture) );
    } catch (...) {
        if (t.joinable ())
            t.join ();
        throw;
    }
}

#else
int LEGACY_OPENGL_GlSyncObjectMutext;
#endif // LEGACY_OPENGL
