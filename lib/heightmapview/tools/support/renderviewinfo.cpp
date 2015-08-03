#include "renderviewinfo.h"
#include "gltextureread.h"
#include "neat_math.h"
#include "gluproject_ios.h"
#include "heightmap/render/renderinfo.h"
#include "heightmap/render/renderset.h"
#include "tools/renderview.h"
#include "heightmap/collection.h"
#include "gluunproject.h"

namespace Tools {
namespace Support {

RenderViewInfo::RenderViewInfo(const Tools::RenderModel* model)
    :
      model(model),
      gl_projection(*model->gl_projection.read ())
{
}


QRectF RenderViewInfo::
        rect()
{
    // can't translate from bottom-relative offset (viewport) to top-relative offset (widget), assume offset is 0
    int device_height = model->render_settings.device_pixel_height;
    int y_offset = device_height - gl_projection.viewport[1] - gl_projection.viewport[3];
    return QRectF(gl_projection.viewport[0], y_offset, gl_projection.viewport[2], gl_projection.viewport[3]);
}


float RenderViewInfo::
        getHeightmapValue( Heightmap::Position pos, Heightmap::Reference* ref, float* pick_local_max, bool fetch_interpolation, bool* is_valid_value )
{
    if (is_valid_value)
        *is_valid_value = true;

    if (pos.time < 0 || pos.scale < 0 || pos.scale >= 1 || pos.time > model->tfr_mapping().read()->length())
        return 0;

    if (is_valid_value)
        *is_valid_value = false;

    Heightmap::Reference myref;
    if (!ref)
    {
        ref = &myref;
        ref->block_index[0] = (unsigned)-1;
    }
    if (ref->block_index[0] == (unsigned)-1)
    {
        *ref = findRefAtCurrentZoomLevel( pos );
    }

    Heightmap::RegionFactory rr(model->tfr_mapping ().read ()->block_layout ());
    Heightmap::Region r = rr.getVisible (*ref);

    ref->block_index[0] = pos.time / r.time();
    ref->block_index[1] = pos.scale / r.scale();
    r = rr.getVisible (*ref);

    Heightmap::Collection::ptr collection = model->collections()[0];
    Heightmap::pBlock block = collection.raw ()->getBlock( *ref );
    Heightmap::ReferenceInfo ri(block->referenceInfo ());
    if (!block)
        return 0;
    if (is_valid_value)
    {
        Signal::IntervalType s = pos.time * ri.sample_rate();
        Signal::Intervals I = Signal::Intervals(s, s+1);
        I &= ri.getInterval ();
        if (I.empty())
        {
            *is_valid_value = true;
        } else
            return 0;
    }

    DataStorage<float>::ptr blockData = GlTextureRead(*block->texture ()).readFloat();

    float* data = blockData->getCpuMemory();
    Heightmap::BlockLayout block_layout = model->tfr_mapping ().read ()->block_layout();
    unsigned w = block_layout.texels_per_row ();
    unsigned h = block_layout.texels_per_column ();
    unsigned x0 = (pos.time-r.a.time)/r.time()*(w-1) + .5f;
    float    yf = (pos.scale-r.a.scale)/r.scale()*(h-1);
    unsigned y0 = yf + .5f;

    EXCEPTION_ASSERT( x0 < w );
    EXCEPTION_ASSERT( y0 < h );
    float v;

    if (!pick_local_max && !fetch_interpolation)
    {
        v = data[ x0 + y0*w ];
    }
    else
    {
        unsigned yb = y0;
        if (0==yb) yb++;
        if (h==yb+1) yb--;
        float v1 = data[ x0 + (yb-1)*w ];
        float v2 = data[ x0 + yb*w ];
        float v3 = data[ x0 + (yb+1)*w ];

        // v1 = a*(-1)^2 + b*(-1) + c
        // v2 = a*(0)^2 + b*(0) + c
        // v3 = a*(1)^2 + b*(1) + c
        float k = 0.5f*v1 - v2 + 0.5f*v3;
        float p = -0.5f*v1 + 0.5f*v3;
        float q = v2;

        float m0;
        if (fetch_interpolation)
        {
            m0 = yf - yb;
        }
        else
        {
            // fetch max
            m0 = -p/(2*k);
        }

        if (m0 > -2 && m0 < 2)
        {
            v = k*m0*m0 + p*m0 + q;
            if (pick_local_max)
                *pick_local_max = r.a.scale + r.scale()*(y0 + m0)/(h-1);
        }
        else
        {
            v = v2;
            if (pick_local_max)
                *pick_local_max = r.a.scale + r.scale()*(y0)/(h-1);
        }

        float testmax;
        float testv = quad_interpol( y0, data + x0, h, w, &testmax );
        float testlocalmax = r.a.scale + r.scale()*(testmax)/(h-1);

        EXCEPTION_ASSERT( testv == v );
        if (pick_local_max)
            EXCEPTION_ASSERT( testlocalmax == *pick_local_max );
    }

    return v;
}


Heightmap::Reference RenderViewInfo::
        findRefAtCurrentZoomLevel(Heightmap::Position p)
{
    return findRefAtCurrentZoomLevel(p, &gl_projection);
}


Heightmap::Reference RenderViewInfo::
        findRefAtCurrentZoomLevel(Heightmap::Position p, const glProjection* gl_projection)
{
    // "collection" isn't needed to compute this, but its convenient
    auto collection = model->collections()[0];
    Heightmap::Reference entireHeightmap = collection.read ()->entireHeightmap();
    Heightmap::BlockLayout bl = collection.read ()->block_layout();
    Heightmap::VisualizationParams::const_ptr vp = collection.read ()->visualization_params();
    Heightmap::Render::RenderInfo ri(gl_projection, bl, vp, model->render_settings.redundancy);
    Heightmap::Reference r = Heightmap::Render::RenderSet(&ri, 0).computeRefAt (p, entireHeightmap);
    return r;
}


QPointF RenderViewInfo::
        getScreenPos( Heightmap::Position pos, double* dist, bool use_heightmap_value )
{
    double objY = 0;
    const auto c = *model->camera.read ();
    float last_ysize = model->render_settings.last_ysize;
    if ((1 != c.orthoview || c.r[0]!=90) && use_heightmap_value)
        objY = getHeightmapValue(pos) * model->render_settings.y_scale * last_ysize;

    vectord win = gl_projection.gluProject (vectord(pos.time, objY, pos.scale));
    vectord::T winX = win[0], winY = win[1]; // winZ = win[2];

    if (dist)
    {
        GLint const* const& vp = gl_projection.viewport.v;
        float z0 = .1, z1=.2;
        vectord projectionPlane = gl_projection.gluUnProject ( vectord( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z0) );
        vectord projectionNormal = gl_projection.gluUnProject( vectord( vp[0] + vp[2]/2, vp[1] + vp[3]/2, z1) ) - projectionPlane;

        vectord p;
        p[0] = pos.time;
        p[1] = 0;//objY;
        p[2] = pos.scale;

        vectord d = p-projectionPlane;
        projectionNormal[0] *= c.xscale;
        projectionNormal[2] *= last_ysize;
        d[0] *= c.xscale;
        d[2] *= last_ysize;

        projectionNormal = projectionNormal.Normalized();

        *dist = d%projectionNormal;
    }

    int r = model->render_settings.dpifactor;
    return QPointF( winX, rect().height() - 1 - winY ) /= r;
}


QPointF RenderViewInfo::
        getWidgetPos( Heightmap::Position pos, double* dist, bool use_heightmap_value )
{
    QPointF pt = getScreenPos(pos, dist, use_heightmap_value);
    pt -= rect().topLeft();
    return pt;
}


Heightmap::Position RenderViewInfo::
        getHeightmapPos( QPointF widget_pos, bool useRenderViewContext )
{
    const auto c = *model->camera.read ();
    if (1 == c.orthoview)
        return getPlanePos(widget_pos, 0, useRenderViewContext);

    TaskTimer tt("RenderViewInfo::getHeightmapPos(%g, %g) Newton raphson", widget_pos.x(), widget_pos.y());
    widget_pos *= model->render_settings.dpifactor;

    QPointF pos;
    pos.setX( widget_pos.x() + rect().left() );
    pos.setY( model->render_settings.device_pixel_height - 1 - widget_pos.y() - rect().top() );

    const vectord::T* m = gl_projection.modelview.v (), *proj = gl_projection.projection.v ();
    const GLint* vp = gl_projection.viewport.v;
#ifndef LEGACY_OPENGL
    EXCEPTION_ASSERTX(useRenderViewContext, "glGetDoublev(*MATRIX) is only supported on legacy OpenGL");
#else
    vectord::T other_m[16], other_proj[16];
    GLint other_vp[4];
    if (!useRenderViewContext)
    {
        glGetDoublev(GL_MODELVIEW_MATRIX, other_m);
        glGetDoublev(GL_PROJECTION_MATRIX, other_proj);
        glGetIntegerv(GL_VIEWPORT, other_vp);
        m = other_m;
        proj = other_proj;
        vp = other_vp;
    }
#endif

    vectord win = gluProject(c.q, m, proj, vp);

    vectord::T objX1, objY1, objZ1;
    gluUnProject( pos.x(), pos.y(), win[2],
                m, proj, vp,
                &objX1, &objY1, &objZ1);

    vectord::T objX2, objY2, objZ2;
    gluUnProject( pos.x(), pos.y(), win[2]*0.999999,
                m, proj, vp,
                &objX2, &objY2, &objZ2);

    Heightmap::Position p;
    double y = 0;

    double prevs = 1;
    // Newton raphson with naive damping
    for (int i=0; i<50; ++i)
    {
        double s = (y-objY1)/(objY2-objY1);
        double k = 1./8;
        s = (1-k)*prevs + k*s;

        Heightmap::Position q;
        q.time = objX1 + s * (objX2-objX1);
        q.scale = objZ1 + s * (objZ2-objZ1);

        Heightmap::Position d(
                (q.time-p.time)*c.xscale,
                (q.scale-p.scale)*c.zscale);

        QPointF r = getWidgetPos( q, 0 );
        d.time = r.x() - widget_pos.x();
        d.scale = r.y() - widget_pos.y();
        float e = d.time*d.time + d.scale*d.scale;
        p = q;
        if (e < 0.4) // || prevs > s)
            break;
        //if (e < 1e-5 )
        //    break;

        y = getHeightmapValue(p) * model->render_settings.y_scale * 4 * model->render_settings.last_ysize;
        tt.info("(%g, %g) %g is Screen(%g, %g), s = %g", p.time, p.scale, y, r.x(), r.y(), s);
        prevs = s;
    }

    y = getHeightmapValue(p) * model->render_settings.y_scale * 4 * model->render_settings.last_ysize;
    TaskInfo("Screen(%g, %g) projects at Heightmap(%g, %g, %g)", widget_pos.x(), widget_pos.y(), p.time, p.scale, y);
    QPointF r = getWidgetPos( p, 0 );
    TaskInfo("Heightmap(%g, %g) projects at Screen(%g, %g)", p.time, p.scale, r.x(), r.y() );
    return p;
}

Heightmap::Position RenderViewInfo::
        getPlanePos( QPointF widget_pos, bool* success, bool useRenderViewContext )
{
    const auto c = *model->camera.read ();

    widget_pos *= model->render_settings.dpifactor;

    QPointF pos;
    pos.setX( widget_pos.x() + rect().left() );
    pos.setY( model->render_settings.device_pixel_height - 1 - widget_pos.y() - rect().top() );

    const vectord::T* m = gl_projection.modelview.v (), *proj = gl_projection.projection.v ();
    const GLint* vp = gl_projection.viewport.v;
#ifndef LEGACY_OPENGL
    EXCEPTION_ASSERTX(useRenderViewContext, "glGetDoublev(*MATRIX) is only supported on legacy OpenGL");
#else
    vectord::T other_m[16], other_proj[16];
    GLint other_vp[4];
    if (!useRenderViewContext)
    {
        glGetDoublev(GL_MODELVIEW_MATRIX, other_m);
        glGetDoublev(GL_PROJECTION_MATRIX, other_proj);
        glGetIntegerv(GL_VIEWPORT, other_vp);
        m = other_m;
        proj = other_proj;
        vp = other_vp;
    }
#endif

    vectord win = gluProject(c.q, m, proj, vp);
    vectord::T objX1, objY1, objZ1;
    gluUnProject( pos.x(), pos.y(), win[2],
                m, proj, vp,
                &objX1, &objY1, &objZ1);

    vectord::T objX2, objY2, objZ2;
    gluUnProject( pos.x(), pos.y(), win[2]*0.999999,
                m, proj, vp,
                &objX2, &objY2, &objZ2);

    double ylevel = 0;
    double s = (ylevel-objY1)/(objY2-objY1);

    if (0==objY2-objY1)
        s = 0;

    Heightmap::Position p;
    p.time = objX1 + s * (objX2-objX1);
    p.scale = objZ1 + s * (objZ2-objZ1);

    if (success) *success=true;

    float minAngle = 3;
    if(success)
    {
        if( s < 0)
            if (success) *success=false;

        float L = sqrt((objX1-objX2)*(objX1-objX2)*c.xscale*c.xscale
                       +(objY1-objY2)*(objY1-objY2)
                       +(objZ1-objZ2)*(objZ1-objZ2)*c.zscale*c.zscale);
        if (objY1-objY2 < sin(minAngle *(M_PI/180)) * L )
            if (success) *success=false;
    }

    return p;
}


QPointF RenderViewInfo::
        widget_coordinates( QPointF window_coordinates )
{
    return window_coordinates - rect().topLeft();
}


QPointF RenderViewInfo::
        window_coordinates( QPointF widget_coordinates )
{
    return widget_coordinates + rect().topLeft();
}


float RenderViewInfo::
        length()
{
    return model->tfr_mapping().read ()->length();
}

} // namespace Support
} // namespace Tools
