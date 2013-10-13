#ifndef HEIGHTMAP_RENDER_RENDERHEIGHTMAP_H
#define HEIGHTMAP_RENDER_RENDERHEIGHTMAP_H

#include "heightmap/blockcache.h"
#include "heightmap/render/frustumclip.h"
#include "glprojection.h"
#include "renderblock.h"
#include "renderinfo.h"

#include <boost/unordered_set.hpp>

namespace Heightmap {
namespace Render {

/**
 * @brief The RenderHeightmap class should find references within a frustum.
 */
class RenderHeightmap
{
public:
    RenderHeightmap(RenderInfo* render_info);


private:
    RenderInfo* render_info;


public:
    static void test();
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERHEIGHTMAP_H
