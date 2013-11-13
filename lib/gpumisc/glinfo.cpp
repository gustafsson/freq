#include "glinfo.h"

#include <sstream>
#include <QGLWidget>

using namespace std;

string glinfo::
        pretty_format(const QGLFormat& f)
{
    QGLFormat::OpenGLVersionFlags flag = f.openGLVersionFlags();

    stringstream s;
    s << "accum=" << f.accum() << endl
      << "accumBufferSize=" << f.accumBufferSize() << endl
      << "alpha=" << f.alpha() << endl
      << "alphaBufferSize=" << f.alphaBufferSize() << endl
      << "blueBufferSize=" << f.blueBufferSize() << endl
      << "depth=" << f.depth() << endl
      << "depthBufferSize=" << f.depthBufferSize() << endl
      << "directRendering=" << f.directRendering() << endl
      << "doubleBuffer=" << f.doubleBuffer() << endl
      << "greenBufferSize=" << f.greenBufferSize() << endl
      << "hasOverlay=" << f.hasOverlay() << endl
      << "redBufferSize=" << f.redBufferSize() << endl
      << "rgba=" << f.rgba() << endl
      << "sampleBuffers=" << f.sampleBuffers() << endl
      << "samples=" << f.samples() << endl
      << "stencil=" << f.stencil() << endl
      << "stencilBufferSize=" << f.stencilBufferSize() << endl
      << "stereo=" << f.stereo() << endl
      << "swapInterval=" << f.swapInterval() << endl
      << "" << endl
      << "hasOpenGL=" << f.hasOpenGL() << endl
      << "hasOpenGLOverlays=" << f.hasOpenGLOverlays() << endl
      << "OpenGL_Version_None=" << (QGLFormat::OpenGL_Version_None == flag) << endl
      << "OpenGL_Version_1_1=" << (QGLFormat::OpenGL_Version_1_1 & flag) << endl
      << "OpenGL_Version_1_2=" << (QGLFormat::OpenGL_Version_1_2 & flag) << endl
      << "OpenGL_Version_1_3=" << (QGLFormat::OpenGL_Version_1_3 & flag) << endl
      << "OpenGL_Version_1_4=" << (QGLFormat::OpenGL_Version_1_4 & flag) << endl
      << "OpenGL_Version_1_5=" << (QGLFormat::OpenGL_Version_1_5 & flag) << endl
      << "OpenGL_Version_2_0=" << (QGLFormat::OpenGL_Version_2_0 & flag) << endl
      << "OpenGL_Version_2_1=" << (QGLFormat::OpenGL_Version_2_1 & flag) << endl
      << "OpenGL_Version_3_0=" << (QGLFormat::OpenGL_Version_3_0 & flag) << endl
      << "OpenGL_ES_CommonLite_Version_1_0=" << (QGLFormat::OpenGL_ES_CommonLite_Version_1_0 & flag) << endl
      << "OpenGL_ES_Common_Version_1_0=" << (QGLFormat::OpenGL_ES_Common_Version_1_0 & flag) << endl
      << "OpenGL_ES_CommonLite_Version_1_1=" << (QGLFormat::OpenGL_ES_CommonLite_Version_1_1 & flag) << endl
      << "OpenGL_ES_Common_Version_1_1=" << (QGLFormat::OpenGL_ES_Common_Version_1_1 & flag) << endl
      << "OpenGL_ES_Version_2_0=" << (QGLFormat::OpenGL_ES_Version_2_0 & flag) << endl;

    return s.str();
}


string glinfo::
        pretty_format(const QGLWidget& w)
{
    stringstream s;

    s << "doubleBuffer=" <<  w.doubleBuffer() << endl
      << "isSharing=" <<  w.isSharing() << endl
      << "isValid=" <<  w.isValid() << endl
      << pretty_format( w.format() );

    return s.str();
}


string glinfo::
        driver_info()
{
    stringstream ss;

    ss << "vendor: " << string((char*)glGetString(GL_VENDOR)) << endl
       << "renderer: " << glGetString(GL_RENDERER) << endl
       << "version: " << glGetString(GL_VERSION) << endl
       << "shading language: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl
       << "extensions/capabilities/caps: " << glGetString(GL_EXTENSIONS) << endl;

    return ss.str();
}


#include <QApplication>
#include "exceptionassert.h"

void glinfo::
        test()
{
    // It should provide a human readable text string of a Qt managed open gl render context.
    {
        int argc = 0;
        char* argv = 0;
        QApplication a(argc,&argv); // takes 0.4 s if this is the first instantiation of QApplication
        QGLWidget w;
        w.makeCurrent ();
        string ws = glinfo::pretty_format (w);
        string fs = glinfo::pretty_format (w.format ());
        string ds = glinfo::driver_info ();

        EXCEPTION_ASSERT(ws.find ("doubleBuffer") != std::string::npos);
        EXCEPTION_ASSERT(fs.find ("depthBufferSize") != std::string::npos);
        EXCEPTION_ASSERT(ws.find (fs) != std::string::npos);
        EXCEPTION_ASSERT(ds.find ("GL_ARB_depth_buffer_float") != std::string::npos);
    }
}
