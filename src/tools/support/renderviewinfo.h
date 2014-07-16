#ifndef TOOLS_SUPPORT_RENDERVIEWINFO_H
#define TOOLS_SUPPORT_RENDERVIEWINFO_H

#include "heightmap/position.h"
#include "heightmap/reference.h"

#include <QPointF>

namespace Tools {
class RenderView;
class RenderModel;

namespace Support {

class RenderViewInfo
{
public:
    RenderViewInfo(Tools::RenderView* renderview);

    QPointF getScreenPos( Heightmap::Position pos, double* dist, bool use_heightmap_value = true );
    QPointF getWidgetPos( Heightmap::Position pos, double* dist, bool use_heightmap_value = true );
    Heightmap::Position getHeightmapPos( QPointF widget_coordinates, bool useRenderViewContext = true );
    Heightmap::Position getPlanePos( QPointF widget_coordinates, bool* success = 0, bool useRenderViewContext = true );
    QPointF widget_coordinates( QPointF window_coordinates );
    QPointF window_coordinates( QPointF widget_coordinates );
    float getHeightmapValue( Heightmap::Position pos, Heightmap::Reference* ref = 0, float* find_local_max = 0, bool fetch_interpolation = false, bool* is_valid_value = 0 );

    /**
      You might want to use Heightmap::Reference::containsPoint(p) to se
      if the returned reference actually is a valid reference for the point
      given. It will not be valid if 'p' lies outside the spectrogram.
      */
    Heightmap::Reference findRefAtCurrentZoomLevel(Heightmap::Position p);

    float length();

private:
    Tools::RenderView* view;
    Tools::RenderModel* model;
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_RENDERVIEWINFO_H
