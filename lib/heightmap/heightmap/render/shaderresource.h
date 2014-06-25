#ifndef HEIGHTMAP_SHADERRESOURCE_H
#define HEIGHTMAP_SHADERRESOURCE_H

namespace Heightmap {

class ShaderResource
{
public:
    static unsigned loadGLSLProgram(const char *vertFileName="", const char *fragFileName="");
};

} // namespace Heightmap

#endif // HEIGHTMAP_SHADERRESOURCE_H
