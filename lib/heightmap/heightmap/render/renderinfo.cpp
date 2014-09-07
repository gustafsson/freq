#include "renderinfo.h"
#include "heightmap/reference.h"
#include "heightmap/position.h"
#include "heightmap/referenceinfo.h"
#include "heightmap/render/frustumclip.h"

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
  @arg ref See timePixels and scalePixels
  @arg timePixels Estimated longest line of pixels along time axis within ref measured in pixels
  @arg scalePixels Estimated longest line of pixels along scale axis within ref measured in pixels
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

        vectord closest_i;
        std::vector<vectord> clippedCorners = frustum_clip.clipFrustum(corner, &closest_i); // about 10 us
        if (clippedCorners.empty ())
            continue;

        vectord timePoint = closest_i;
        vectord scalePoint = closest_i;
        for (vectord v : clippedCorners)
        {
            if (fabsf(v[0]-closest_i[0]) > fabsf(timePoint[0]-closest_i[0]))
                timePoint = v;
            if (fabsf(v[2]-closest_i[2]) > fabsf(scalePoint[2]-closest_i[2]))
                scalePoint = v;
        }

        timePoint[2] = closest_i[2];
        scalePoint[0] = closest_i[0];

        timePoint = closest_i + (timePoint-closest_i)*0.01f;
        scalePoint = closest_i + (scalePoint-closest_i)*0.01f;

        vectord::T timeLength = fabsf (closest_i[0] - timePoint[0]);
        vectord::T scaleLength = fabsf (closest_i[2] - scalePoint[2]);

        if (timeLength==0 || scaleLength==0)
            continue;

        // time/scalepixels is approximately the number of pixels in ref along the time/scale axis
        vectord::T pixelsPerTime = gl_projection->computePixelDistance (closest_i, timePoint) / timeLength;
        vectord::T pixelsPerScale = gl_projection->computePixelDistance (closest_i, scalePoint) / scaleLength;
        timePixels = pixelsPerTime * r.time ();
        scalePixels = pixelsPerScale * r.scale ();

        return true;
    }

    return false;
}


} // namespace Render
} // namespace Heightmap
