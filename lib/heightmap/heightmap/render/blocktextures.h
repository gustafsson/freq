#ifndef HEIGHTMAP_RENDER_BLOCKTEXTURES_H
#define HEIGHTMAP_RENDER_BLOCKTEXTURES_H

#include "GlTexture.h"
#include "shared_state.h"
#include <memory>
#include <vector>

namespace Heightmap {
namespace Render {

/**
 * This is a singleton.
 *
 * @brief The BlockTextures class should keep track of all OpenGL textures that
 * have been allocated for painting block textures and provide already
 * allocated textures fast.
 *
 * It should provide already allocated textures fast.
 *
 * The memory is a shared global resource. Thus the instance of BlockTextures
 * is also a shared and global singleton.
 */
class BlockTextures
{
public:
    static bool isInitialized();
    static void init(unsigned width, unsigned height, unsigned initialCapacity = 0);
    static void destroy();

    /**
     * @brief Scoped provides a scoped initialization, useful in test environments.
     */
    struct Scoped {
        Scoped(unsigned width, unsigned height, unsigned initialCapacity = 0) {
            init(width,height,initialCapacity);
        }

        ~Scoped() { destroy(); }
    };

    static void testInit(unsigned width, unsigned height, unsigned initialCapacity = 0);

    /**
     * @brief setCapacityHint
     * The actual capacity will be above 'c' and below '3*c'.
     * When the capacity is out of range the new capacity will be set to '2*c'.
     *
     * @param c
     */
    static void setCapacityHint(unsigned c);

    /**
     * @brief gc, calls setCapacityHint('number of currently used textures')
     */
    static void gc();

    /**
     * @brief getTextures
     * @returns 'count' textures
     *
     * To free up unused textures, delete the previously obtained copies.
     */
    static std::vector<GlTexture::ptr> getTextures(unsigned count);

    /**
     * @brief get1
     * @return
     */
    static GlTexture::ptr get1();

    /**
     * @brief getCapacity
     * @return
     */
    static int getCapacity();
    static unsigned getWidth();
    static unsigned getHeight();

    /**
     * @brief setupTexture
     * @param name
     * @param width
     * @param height
     */
    static void setupTexture(unsigned name, unsigned width, unsigned height, bool mipmaps=true);

    /**
     * @brief allocated_bytes_per_element
     */
    static unsigned allocated_bytes_per_element();

public:
    static void test();

private:
    BlockTextures(); // don't instantiate
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_BLOCKTEXTURES_H
