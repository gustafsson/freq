#ifndef HEIGHTMAP_SHADERRESOURCE_H
#define HEIGHTMAP_SHADERRESOURCE_H

namespace Heightmap {

class ShaderResource
{
public:
    static GLuint loadGLSLProgram(const char *vertFileName="", const char *fragFileName="");
};

} // namespace Heightmap

#endif // HEIGHTMAP_SHADERRESOURCE_H
