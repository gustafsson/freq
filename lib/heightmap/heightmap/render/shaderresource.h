#ifndef HEIGHTMAP_SHADERRESOURCE_H
#define HEIGHTMAP_SHADERRESOURCE_H

#include <QString>

class QOpenGLShaderProgram;
namespace Heightmap {

typedef std::unique_ptr<QOpenGLShaderProgram> ShaderPtr;
class ShaderResource
{
public:
    static ShaderPtr loadGLSLProgram(const char *vertFileName="", const char *fragFileName="");
    static ShaderPtr loadGLSLProgramSource(QString vertShader, QString fragShader);
};

} // namespace Heightmap

#endif // HEIGHTMAP_SHADERRESOURCE_H
