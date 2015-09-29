#include "renderinfo.h"
#include "heightmap/reference.h"
#include "heightmap/position.h"
#include "heightmap/referenceinfo.h"
#include "heightmap/render/frustumclip.h"
#include "log.h"

namespace Heightmap {
namespace Render {

template<typename T>
auto length2d(tvector<3,T> a) -> T {
    return std::sqrt(a[0]*a[0]+a[1]*a[1]);
}

CornerResolution::CornerResolution(float x00, float x01, float x10, float x11, float y00, float y01, float y10, float y11)
    :
      // the mipmap level is log2(texels/pixel)
      // the lowest mipmap level is 0. i.e log2f(1). A level below 0 would mean magnification, but that doesn't apply.
      x00(std::max(0.5f, x00)),
      x01(std::max(0.5f, x01)),
      x10(std::max(0.5f, x10)),
      x11(std::max(0.5f, x11)),
      y00(std::max(0.5f, y00)),
      y01(std::max(0.5f, y01)),
      y10(std::max(0.5f, y10)),
      y11(std::max(0.5f, y11))
{
}


RenderInfo::
        RenderInfo(const glProjecter* gl_projecter, BlockLayout bl, VisualizationParams::const_ptr vp, float redundancy)
    :
      gl_projecter(gl_projecter),
      frustum_clip(*gl_projecter),
      bl(bl),
      vp(vp),
      redundancy(redundancy)
{
}


LevelOfDetail RenderInfo::
        testLod( Reference ref ) const
{
    double pixelsPerTimeTexel, pixelsPerScaleTexel;
    Region r = RegionFactory ( bl ).getVisible (ref);
    if (!computePixelsPerUnit( r, pixelsPerTimeTexel, pixelsPerScaleTexel ))
        return false;

    double needBetterT = pixelsPerTimeTexel / redundancy,
           needBetterS = pixelsPerScaleTexel / redundancy;

    bool max_s =
            !ReferenceInfo(ref.top(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighS) &&
            !ReferenceInfo(ref.bottom(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighS);

    bool max_t =
            !ReferenceInfo(ref.left(), bl, vp).boundsCheck(ReferenceInfo::BoundsCheck_HighT);

    return LevelOfDetail(needBetterT, needBetterS, max_t, max_s);
}


CornerResolution RenderInfo::
        cornerResolution (Reference ref) const
{
    Region r = RegionFactory ( bl ).getVisible (ref);
    const Position p[2] = { r.a, r.b };

    vectord corner[]=
    {
        vectord( p[0].time, 0, p[0].scale),
        vectord( p[0].time, 0, p[1].scale),
        vectord( p[1].time, 0, p[0].scale),
        vectord( p[1].time, 0, p[1].scale)
    };

    // use a delta of one texel along both time and scale to estimate the resolution
    vectord::T dt = r.time ()/bl.texels_per_row (),
               ds = r.scale ()/bl.texels_per_column ();

    float l = 1.f; // distance in texels

    // xNN is pixels per texel
    vectord screen[]=
    {
        gl_projecter->project (corner[0]),
        gl_projecter->project (corner[1]),
        gl_projecter->project (corner[2]),
        gl_projecter->project (corner[3])
    };
    float x00 = length2d(screen[0] - gl_projecter->project (corner[0] + vectord(dt,0,0)));
    float x01 = length2d(screen[1] - gl_projecter->project (corner[1] + vectord(dt,0,0)));
    float x10 = length2d(screen[2] - gl_projecter->project (corner[2] + vectord(-dt,0,0)));
    float x11 = length2d(screen[3] - gl_projecter->project (corner[3] + vectord(-dt,0,0)));
    float y00 = length2d(screen[0] - gl_projecter->project (corner[0] + vectord(0,0,ds)));
    float y01 = length2d(screen[1] - gl_projecter->project (corner[1] + vectord(0,0,-ds)));
    float y10 = length2d(screen[2] - gl_projecter->project (corner[2] + vectord(0,0,ds)));
    float y11 = length2d(screen[3] - gl_projecter->project (corner[3] + vectord(0,0,-ds)));

    // ignore if p == 0
    if (x00 == 0) x00 = l;
    if (x01 == 0) x01 = l;
    if (x10 == 0) x10 = l;
    if (x11 == 0) x11 = l;
    if (y00 == 0) y00 = l;
    if (y01 == 0) y01 = l;
    if (y10 == 0) y10 = l;
    if (y11 == 0) y11 = l;

    // CornerResolution is built from texels per pixel in each corner
    return CornerResolution(l/x00, l/x01, l/x10, l/x11, l/y00, l/y01, l/y10, l/y11);
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
  @arg timePixels Resolution in pixels per time texel
  @arg scalePixels Resolution in pixels per scale texel
  */
bool RenderInfo::
        computePixelsPerUnit( Region r, double& pixelsPerTimeTexel, double& pixelsPerScaleTexel ) const
{
    vectord corner[]=
    {
        vectord( r.a.time, 0, r.a.scale),
        vectord( r.a.time, 0, r.b.scale),
        vectord( r.b.time, 0, r.b.scale),
        vectord( r.b.time, 0, r.a.scale)
    };

    // the resolution needed from the whole block is determinded by the
    // resolution needed from the point of the block that is closest to
    // the camera, start by finding that point
    vectord closest_i;
    std::vector<vectord> clippedCorners = frustum_clip.clipFrustum(corner, &closest_i); // about 10 us
    if (clippedCorners.empty ())
        return false;

    // use a delta of one texel to estimate the resolution at closest_i
    vectord::T deltaTime = r.time () / bl.texels_per_row (),
               deltaScale = r.scale () / bl.texels_per_column ();
    vectord timePoint = closest_i + vectord(deltaTime,0,0),
            scalePoint = closest_i + vectord(0,0,deltaScale);

    vectord screen_closest = gl_projecter->project( closest_i ),
            screen_t = gl_projecter->project( timePoint ),
            screen_s = gl_projecter->project( scalePoint );

    pixelsPerTimeTexel = length2d(screen_closest-screen_t);
    pixelsPerScaleTexel = length2d(screen_closest-screen_s);

    return true;
}


} // namespace Render
} // namespace Heightmap
