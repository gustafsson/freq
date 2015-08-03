#include "renderinfo.h"
#include "heightmap/reference.h"
#include "heightmap/position.h"
#include "heightmap/referenceinfo.h"
#include "heightmap/render/frustumclip.h"
#include "log.h"

namespace Heightmap {
namespace Render {


LevelOfDetail::
        LevelOfDetail(bool valid)
    :
      LevelOfDetail(valid?1:-1, valid?1:-1, true, true)
{}


LevelOfDetail::
        LevelOfDetail(double pixels_per_data_point_t, double pixels_per_data_point_s,
              bool max_t, bool max_s)
    :
      pixels_per_data_point_t(pixels_per_data_point_t),
      pixels_per_data_point_s(pixels_per_data_point_s),
      max_t(max_t),
      max_s(max_s)
{
}


bool LevelOfDetail::
        valid() const
{
    return pixels_per_data_point_t>=0 && pixels_per_data_point_s>=0;
}


RenderInfo::
        RenderInfo(const glProjection* gl_projection, BlockLayout bl, VisualizationParams::const_ptr vp, float redundancy)
    :
      gl_projection(gl_projection),
      bl(bl),
      vp(vp),
      redundancy(redundancy)
{
}


LevelOfDetail RenderInfo::
        testLod( Reference ref ) const
{
    double timePixels, scalePixels;
    Region r = RegionFactory ( bl ).getVisible (ref);
    if (!computePixelsPerUnit( r, timePixels, scalePixels ))
        return false;

    double needBetterT = timePixels / (redundancy*bl.texels_per_row ()),
           needBetterS = scalePixels / (redundancy*bl.texels_per_column ());

    bool max_s =
            !ReferenceInfo(ref.top(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighS) &&
            !ReferenceInfo(ref.bottom(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighS);

    bool max_t =
            !ReferenceInfo(ref.left(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighT);

    return LevelOfDetail(needBetterT, needBetterS, max_t, max_s);
}


bool RenderInfo::
        boundsCheck(Reference ref, ReferenceInfo::BoundsCheck bc) const
{
    return ReferenceInfo(ref, bl, vp).boundsCheck(bc);
}


Region RenderInfo::
        visible_region(Reference ref) const
{
    return RegionFactory(bl).getVisible (ref);
}


/**
  @arg r Region to study. Only the point in 'r' closest to the camera will be considered.
  @arg timePixels Resolution in pixels per 'r.time()' time units
  @arg scalePixels Resolution in pixels per 'r.scale()' scale unit
  */
bool RenderInfo::
        computePixelsPerUnit( Region r, double& timePixels, double& scalePixels ) const
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
