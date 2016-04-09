#ifndef HEIGHTMAP_SHADERRESOURCE_H
#define HEIGHTMAP_SHADERRESOURCE_H

#include <QString>
#include <QOpenGLShaderProgram>
#include <memory>

namespace Heightmap {

class OpenGLShaderProgramGlState : public QOpenGLShaderProgram
{
public:
    ~OpenGLShaderProgramGlState();
};


typedef std::unique_ptr<OpenGLShaderProgramGlState> ShaderPtr;


class ShaderResource
{
public:
    static ShaderPtr loadGLSLProgram(const char *vertFileName="", const char *fragFileName="", const char* vertTop=0, const char* fragTop=0);
    static ShaderPtr loadGLSLProgramSource(QString vertShader, QString fragShader, const char* vertTop=0, const char* fragTop=0);
};

} // namespace Heightmap

#endif // HEIGHTMAP_SHADERRESOURCE_H
