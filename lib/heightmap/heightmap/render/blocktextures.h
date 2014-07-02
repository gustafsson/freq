#ifndef HEIGHTMAP_RENDER_BLOCKTEXTURES_H
#define HEIGHTMAP_RENDER_BLOCKTEXTURES_H

#include "GlTexture.h"
#include "shared_state.h"
#include <memory>
#include <vector>

namespace Heightmap {
namespace Render {

/**
 * @brief The BlockTextures class should keep track of all OpenGL textures that
 * have been allocated for painting block textures and provide already
 * allocated textures fast.
 *
 * It should provide already allocated textures fast.
 */
class BlockTextures
{
public:
    typedef shared_state<BlockTextures> ptr;

    explicit BlockTextures(unsigned width, unsigned height, unsigned initialCapacity = 0);

    /**
     * @brief setCapacityHint
     * The actual capacity will be above 'c' and below '3*c'.
     * When the capacity is out of range the new capacity will be set to '2*c'.
     *
     * @param c
     */
    void setCapacityHint(unsigned c);

    /**
     * @brief getUnusedTexture
     * @return up to 'count' textures if there are any unused ones.
     *
     * To free up unused textures, delete the previously obtained copies.
     */
    std::vector<GlTexture::ptr> getUnusedTextures(unsigned count) const;

    /**
     * @brief get1
     * @return
     */
    GlTexture::ptr get1();

    /**
     * @brief getCapacity
     * @return
     */
    int getCapacity() const;
    unsigned getWidth() const { return width_; }
    unsigned getHeight() const { return height_; }

    /**
     * @brief setupTexture
     * @param name
     * @param width
     * @param height
     */
    static void setupTexture(unsigned name, unsigned width, unsigned height);

    /**
     * @brief allocated_bytes_per_element
     */
    static unsigned allocated_bytes_per_element();

private:
    std::vector<GlTexture::ptr> textures;
    const unsigned width_, height_;

    void setCapacity (unsigned target_capacity);
public:
    static void test();
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_BLOCKTEXTURES_H
