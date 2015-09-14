#include "rendersettings.h"

namespace Heightmap {
namespace Render {

RenderSettings::RenderSettings()
    :   draw_piano(false),
        draw_hz(true),
        draw_t(true),
        draw_cursor_marker(false),
        draw_axis_at0(0),
        draw_contour_plot(false),
        color_mode( ColorMode_Rainbow ),
        fixed_color( 1,0,0,1 ),
        clear_color( 1,1,1,1 ),
        y_scale( 1 ),
        y_offset( 0 ),
        y_normalize( 0 ),
        log_scale( 0 ),
        last_ysize( 1 ),
        last_axes_length( 0 ),
        drawn_blocks(0),
        left_handed_axes(true),
        shadow_shader(true),
        draw_flat(true),
        drawcrosseswhen0( false ),
        dpifactor( 1 ),
        axes_border(false),

        /*
         reasoning about the default redundancy value.
         The thing about Sonic AWE is a good visualization. In this there is value
         booth in smooth navigation and high resolution. As the navigation is fast
         on new computers even with high resolution we set this value to give most
         people a good first impression. For people with older computers it's
         possible to suggest that they lower the resolution for faster navigation.

         This could be done through a dropdownnotification if plain rendering
         takes too long.
         */
        redundancy(1.0f) // 1 means every pixel gets at least one texel (and vertex), 10 means every 10th pixel gets its own vertex, default=2
{
}

} // namespace Render
} // namespace Heightmap
