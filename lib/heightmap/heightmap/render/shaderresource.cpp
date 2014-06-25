#include "shaderresource.h"
#include "GlException.h"

#include "exceptionassert.h"
#include "gl.h"
#include "demangle.h"
#include "tasktimer.h"

// Qt
#include <QResource>

//#define TIME_COMPILESHADER
#define TIME_COMPILESHADER if(0)

using namespace std;

namespace Heightmap {


// Helpers based on Cuda SDK sample, ocean FFT
// TODO check license terms of the Cuda SDK

// Attach shader to a program
string attachShader(GLuint prg, GLenum type, const char *name)
{
    stringstream result;

    TIME_COMPILESHADER TaskTimer tt("Compiling shader %s", name);
    try {
        GLuint shader;
        FILE * fp=0;
        int size, compiled;
        char * src;

        shader = glCreateShader(type);

        QResource qr(name);
        EXCEPTION_ASSERTX( qr.isValid(), string("Couldn't find shader resource ") + name);
        EXCEPTION_ASSERTX( 0 != qr.size(), string("Shader resource empty ") + name);

        size = qr.size();
        src = (char*)qr.data();
        glShaderSource(shader, 1, (const char**)&src, (const GLint*)&size);
        glCompileShader(shader);
        glGetShaderiv(shader, GL_COMPILE_STATUS, (GLint*)&compiled);

        if (fp) free(src);

        char shaderInfoLog[2048];
        glGetShaderInfoLog(shader, sizeof(shaderInfoLog), 0, shaderInfoLog);

        bool showShaderLog = !compiled;
#ifdef _DEBUG
        QString qshaderInfoLog(shaderInfoLog);
        showShaderLog |= 0 != qshaderInfoLog.contains("fail", Qt::CaseInsensitive);
        showShaderLog |= 0 != qshaderInfoLog.contains("warning", Qt::CaseInsensitive);
        showShaderLog |= strlen(shaderInfoLog)>0;
#endif

        if (showShaderLog)
        {
            result << "Failed to compile shader '" << name << "'"  << endl
                   << shaderInfoLog << endl;
        }

        if (compiled)
        {
            glAttachShader(prg, shader);
        }

        glDeleteShader(shader);

    } catch (const exception &x) {
        TIME_COMPILESHADER TaskInfo("Failed, throwing %s", vartype(x).c_str());
        throw;
    }

    return result.str();
}


// Create shader program from vertex shader and fragment shader files
unsigned ShaderResource::
        loadGLSLProgram(const char *vertFileName, const char *fragFileName)
{
    GLint linked;
    GLuint program;
    stringstream resultLog;

    program = glCreateProgram();
    try {
        if (strlen(vertFileName))
            resultLog << attachShader(program, GL_VERTEX_SHADER, vertFileName);
        if (strlen(fragFileName))
            resultLog << attachShader(program, GL_FRAGMENT_SHADER, fragFileName);

        glLinkProgram(program);
        glGetProgramiv(program, GL_LINK_STATUS, &linked);

        char programInfoLog[2048];
        glGetProgramInfoLog(program, sizeof(programInfoLog), 0, programInfoLog);
        if (!linked)
            TaskTimer tt("Failed to link vertex shader \"%s\" with fragment shader \"%s\"\n%s",
                         vertFileName, fragFileName, programInfoLog);

        bool showProgramLog = !linked;
#ifdef _DEBUG
        QString qprogramInfoLog(programInfoLog);
        showProgramLog |= 0 != qprogramInfoLog.contains("fail", Qt::CaseInsensitive);
        showProgramLog |= 0 != qprogramInfoLog.contains("warning", Qt::CaseInsensitive);
        showProgramLog |= strlen(programInfoLog)>0;
#endif

        if (showProgramLog)
        {
            stringstream log;
            log     << "Failed to link fragment shader (" << fragFileName << ") "
                    << "with vertex shader (" << vertFileName << ")" << endl
                    << programInfoLog << endl
                    << resultLog.str();

            TaskInfo("Couldn't properly setup graphics\n%s", log.str().c_str());

            EXCEPTION_ASSERTX(false, log.str ());
        }

        glUseProgram(program);

        GLenum glError = glGetError();
        if (GL_NO_ERROR != glError)
        {
            TaskInfo("glUseProgram failed %s", gluErrorString(glError));
            program = 0;
        }

        glUseProgram( 0 );

    } catch (...) {
        glDeleteProgram(program);
        throw;
    }
    return program;
}


} // namespace Heightmap
