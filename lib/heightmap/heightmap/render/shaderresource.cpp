#include "shaderresource.h"

#include "exceptionassert.h"
#include "log.h"
#include "tasktimer.h"

// Qt
#include <QOpenGLShaderProgram>
#include <QResource>

#define TIME_COMPILESHADER
//#define TIME_COMPILESHADER if(0)

using namespace std;

namespace Heightmap {

ShaderPtr ShaderResource::
        loadGLSLProgram(const char *vertFileName, const char *fragFileName, const char* vertTop, const char* fragTop)
{
    QString vertShader, fragShader;
    if (vertFileName!=0 && *vertFileName!=0)
        vertShader = (const char*)QResource(vertFileName).data ();
    if (fragFileName!=0 && *fragFileName!=0)
        fragShader = (const char*)QResource(fragFileName).data ();

    return loadGLSLProgramSource(vertShader, fragShader, vertTop, fragTop);
}

ShaderPtr ShaderResource::
    loadGLSLProgramSource(QString vertShader, QString fragShader, const char* vertTop, const char* fragTop)
{
#ifdef GL_ES_VERSION_2_0
    if (fragShader.contains ("fwidth") || fragShader.contains ("dFdx") || fragShader.contains ("dFdy"))
    {
        fragShader = "#extension GL_OES_standard_derivatives : enable\n" + fragShader;
    }
#endif

#if !defined(LEGACY_OPENGL) && !defined(GL_ES_VERSION_2_0)
    if (vertShader.contains (QRegExp("\\btexture\\s*[;=]")))
    {
        EXCEPTION_ASSERTX(false,
                boost::format("ShaderResource: vertex shader contains illegal identifier ('texture\\s*[;=]')\n%s")
                        % vertShader.toStdString ());
    }

    if (fragShader.contains (QRegExp("\\btexture\\s*[;=]")) || fragShader.contains (QRegExp("\\bout_FragColor\\b")))
    {
        EXCEPTION_ASSERTX(false,
                boost::format("ShaderResource: fragment shader contains illegal identifier  ('texture\\s*[;=]' or out_FragColor)\n%s")
                        % fragShader.toStdString ());
    }

    vertShader.replace (QRegExp("\\battribute\\b"),"in");
    vertShader.replace (QRegExp("\\bvarying\\b"),"out");
    vertShader.replace (QRegExp("\\btexture2D\\b"),"texture");
    if (vertTop) {
        vertShader = "\n" + vertShader;
        vertShader = vertTop + vertShader;
    }
    vertShader = "#version 150\n" + vertShader;

    fragShader.replace (QRegExp("\\bvarying\\b"),"in");
    fragShader.replace (QRegExp("\\bgl_FragColor\\b"),"out_FragColor");
    fragShader.replace (QRegExp("\\btexture2D\\b"),"texture");
    if (fragTop) {
        fragShader = "\n" + fragShader;
        fragShader = fragTop + fragShader;
    }
    fragShader = "out vec4 out_FragColor;\n" + fragShader;
    fragShader = "#version 150\n" + fragShader;
#endif

    QOpenGLShaderProgram* program = new QOpenGLShaderProgram();
    if (!vertShader.isEmpty ())
        program->addShaderFromSourceCode (QOpenGLShader::Vertex, vertShader);
    if (!fragShader.isEmpty ())
        program->addShaderFromSourceCode (QOpenGLShader::Fragment, fragShader);
    program->link();

    QString log = program->log ();
    EXCEPTION_ASSERTX(program->isLinked (),
                      boost::format("ShaderResource:\n%s\n\n--- Vertex shader ---\n%s\n\n--- Fragment shader ---\n%s")
                              % log.toStdString ()
                              % vertShader.toStdString ()
                              % fragShader.toStdString ());

    if (log.contains("fail", Qt::CaseInsensitive) ||
            log.contains("warn", Qt::CaseInsensitive))
    {
        Log("ShaderResource:\n%s\n\n--- Vertex shader ---\n%s\n\n--- Fragment shader ---\n%s")
                % log.toStdString ()
                % vertShader.toStdString ()
                % fragShader.toStdString ();
    }

    return ShaderPtr(program);
}

} // namespace Heightmap
