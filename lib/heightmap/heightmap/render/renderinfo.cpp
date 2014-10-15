#include "renderinfo.h"
#include "heightmap/reference.h"
#include "heightmap/position.h"
#include "heightmap/referenceinfo.h"
#include "heightmap/render/frustumclip.h"
#include "log.h"

namespace Heightmap {
namespace Render {


RenderInfo::
        RenderInfo(const glProjection* gl_projection, BlockLayout bl, VisualizationParams::const_ptr vp, float redundancy)
    :
      gl_projection(gl_projection),
      bl(bl),
      vp(vp),
      redundancy(redundancy)
{
}


RenderInfo::LevelOfDetal RenderInfo::
        testLod( Reference ref ) const
{
    float timePixels, scalePixels;
    Region r = RegionFactory ( bl )(ref);
    if (!computePixelsPerUnit( r, timePixels, scalePixels ))
        return Lod_Invalid;

    if(0) if (-10==ref.log2_samples_size[0] && -8==ref.log2_samples_size[1]) {
        fprintf(stdout, "Ref (%d,%d)\t%g\t%g\n", ref.block_index[0], ref.block_index[1], timePixels,scalePixels);
        fflush(stdout);
    }

    GLdouble needBetterF = scalePixels / (redundancy*bl.texels_per_column ()),
             needBetterT = timePixels / (redundancy*bl.texels_per_row ());

    if (!ReferenceInfo(ref.top(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighS) &&
        !ReferenceInfo(ref.bottom(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighS))
        needBetterF = 0;

    if (!ReferenceInfo(ref.left(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighT))
        needBetterT = 0;

    if ( needBetterF > needBetterT && needBetterF > 1 )
        return Lod_NeedBetterF;

    else if ( needBetterT > 1 )
        return Lod_NeedBetterT;

    else
        return Lod_Ok;
}


bool RenderInfo::
        boundsCheck(Reference ref, ReferenceInfo::BoundsCheck bc) const
{
    return ReferenceInfo(ref, bl, vp).boundsCheck(bc);
}


Region RenderInfo::
        region(Reference ref) const
{
    return RegionFactory(bl)(ref);
}


/**
  @arg r Region to study. Only the point in 'r' closest to the camera will be considered.
  @arg timePixels Resolution in pixels per 'r.time()' time units
  @arg scalePixels Resolution in pixels per 'r.scale()' scale unit
  */
bool RenderInfo::
        computePixelsPerUnit( Region r, float& timePixels, float& scalePixels ) const
{
    const Position p[2] = { r.a, r.b };

    Render::FrustumClip frustum_clip(*gl_projection);
    double y[]={0, frustum_clip.getCamera()[1]*.5};
    for (unsigned i=0; i<sizeof(y)/sizeof(y[0]); ++i)
    {
        vectord corner[]=
        {
            vectord( p[0].time, y[i], p[0].scale),
            vectord( p[0].time, y[i], p[1].scale),
            vectord( p[1].time, y[i], p[1].scale),
            vectord( p[1].time, y[i], p[0].scale)
        };

        // the resolution needed from the whole block is determinded by the
        // resolution needed from the point of the block that is closest to
        // the camera, start by finding that point
        vectord closest_i;
        std::vector<vectord> clippedCorners = frustum_clip.clipFrustum(corner, &closest_i); // about 10 us
        if (clippedCorners.empty ())
            continue;

        // use a small delta to estimate the resolution at closest_i
        vectord::T deltaTime = 0.001*r.time (),
                   deltaScale = 0.001*r.scale ();
        vectord timePoint = closest_i + vectord(deltaTime,0,0);
        vectord scalePoint = closest_i + vectord(0,0,deltaScale);

        // time/scalepixels is approximately the number of pixels in ref along the time/scale axis
        vectord::T pixelsPerTime = gl_projection->computePixelDistance (closest_i, timePoint) / deltaTime;
        vectord::T pixelsPerScale = gl_projection->computePixelDistance (closest_i, scalePoint) / deltaScale;
        timePixels = pixelsPerTime * r.time ();
        scalePixels = pixelsPerScale * r.scale ();

        // fail if 'r' doesn't even cover a pixel
        if (scalePixels < 0.5 || timePixels < 0.5)
            continue;

        return true;
    }

    return false;
}


} // namespace Render
} // namespace Heightmap
