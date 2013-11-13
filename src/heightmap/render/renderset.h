#ifndef HEIGHTMAP_RENDER_RENDERSET_H
#define HEIGHTMAP_RENDER_RENDERSET_H

#include "heightmap/reference.h"
#include "heightmap/render/renderinfo.h"

#include <boost/unordered_set.hpp>

namespace Heightmap {
namespace Render {

/**
 * @brief The RenderSet class should compute which references that cover the
 * visible heightmap with sufficient resolution, as judged by RenderInfoI.
 */
class RenderSet
{
public:
    typedef boost::unordered_set<Reference> references_t;

    RenderSet(RenderInfoI* render_info, float L);


    /**
     * @brief computeRenderSet
     * @param entireHeightmap
     * @return
     */
    references_t    computeRenderSet( Reference entireHeightmap );


    /**
     * @brief computeRefAt
     * @param p
     * @param entireHeightmap
     * @return
     *
     * Could return a set from the previously computed render set.
     */
    Reference       computeRefAt( Heightmap::Position p, Reference entireHeightmap ) const;

private:
    RenderInfoI*    render_info;
    float           L;

    references_t    computeChildrenRenderSet( Reference ref );

public:
    static void test();
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERSET_H
