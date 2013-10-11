#include "rendersettings.h"
#include "sawe/configuration.h"

namespace Heightmap {

RenderSettings::RenderSettings()
    :   draw_piano(false),
        draw_hz(true),
        draw_t(true),
        draw_cursor_marker(false),
        draw_axis_at0(0),
        camera(0,0,0),
        draw_contour_plot(false),
        color_mode( ColorMode_Rainbow ),
        fixed_color( 1,0,0,1 ),
        clear_color( 1,1,1,0 ),
        y_scale( 1 ),
        last_ysize( 1 ),
        last_axes_length( 0 ),
        drawn_blocks(0),
        left_handed_axes(true),
        vertex_texture(true),
        draw_flat(true),
        drawcrosseswhen0( Sawe::Configuration::version().empty() )
{
}

} // namespace Heightmap
