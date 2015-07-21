#ifndef HEIGHTMAP_UPDATE_OPENGL_TEXTUREPOOL_H
#define HEIGHTMAP_UPDATE_OPENGL_TEXTUREPOOL_H

#include "GlTexture.h"
#include "vbo.h"
#include <vector>

namespace Heightmap {
namespace Update {
namespace OpenGL {

/**
 * @brief The TexturePool class should keep frequently used textures allocated
 * to avoid the overhead of allocating the same amount of memory over and over
 * again.
 *
 * Similar to class Heightmap::Render::BlockTextures but less complex, and not
 * global. Will use Heightmap::Render::BlockTextures::setupTexture for setup.
 */
class TexturePool
{
public:
    enum FloatSize {
        Float16 = 16,
        Float32 = 32
    };

    TexturePool(int width, int height, FloatSize format);
    TexturePool(const TexturePool&)=delete;
    TexturePool& operator=(const TexturePool&)=delete;
    TexturePool(TexturePool&&)=default;
    TexturePool& operator=(TexturePool&&)=default;

    size_t size();
    void resize(size_t);

    // get1 will increase the size of the pool if no textures were available
    GlTexture::ptr get1();

    int width() const { return width_; }
    int height() const { return height_; }
    FloatSize format() { return format_; }

private:
    int width_, height_;
    FloatSize format_;
    std::vector<GlTexture::ptr> pool;

    GlTexture::ptr newTexture();
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_TEXTUREPOOL_H
