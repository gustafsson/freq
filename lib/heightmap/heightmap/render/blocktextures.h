#ifndef HEIGHTMAP_RENDER_BLOCKTEXTURES_H
#define HEIGHTMAP_RENDER_BLOCKTEXTURES_H

#include "heightmap/blocklayout.h"
#include "GlTexture.h"
#include "shared_state.h"
#include <memory>
#include <vector>

namespace Heightmap {
namespace Render {

/**
 * @brief The BlockTextures class should keep track of all OpenGL textures that
 * have been allocated for painting GlBlocks.
 *
 * It should provide already allocated textures fast.
 */
class BlockTextures
{
public:
    typedef shared_state<BlockTextures> ptr;

    BlockTextures(BlockLayout l);

    /**
     * @brief setCapacityHint
     * The actual capacity will be above 'c' and below '3*c'.
     * When the capacity is out of range
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
     * @brief getCapacity
     * @return
     */
    int getCapacity() const;

private:
    std::vector<GlTexture::ptr> textures;
    BlockLayout block_layout;

public:
    static void test();
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_BLOCKTEXTURES_H
