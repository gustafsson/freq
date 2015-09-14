#ifndef HEIGHTMAP_RENDER_FRUSTUMCLIP_H
#define HEIGHTMAP_RENDER_FRUSTUMCLIP_H

#include "glprojection.h"
#include <vector>

namespace Heightmap {
namespace Render {

class FrustumClip
{
public:
    FrustumClip(const glProjecter& gl_projection, float border_width=0, float border_height=0);

    const vectord& getCamera() const { return camera; }

    std::vector<vectord> clipFrustum( vectord corner[4], vectord* closest_i=0 ) const;
    std::vector<vectord> clipFrustum( std::vector<vectord> l, vectord* closest_i=0 ) const;
    std::vector<vectord> visibleXZ();

private:
    vectord camera;
    tvector<4,double> right, left, top, bottom, far, near;

    void update(const glProjecter& gl_projection, double w, double h);
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_FRUSTUMCLIP_H
