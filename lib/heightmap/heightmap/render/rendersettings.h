#ifndef HEIGHTMAP_RENDERSETTINGS_H
#define HEIGHTMAP_RENDERSETTINGS_H

// gpumisc
#include "GLvector.h"
#include "TAni.h"

namespace Heightmap {
namespace Render {

class RenderSettings
{
public:
    enum ColorMode {
        ColorMode_Rainbow,
        ColorMode_Grayscale,
        ColorMode_BlackGrayscale,
        ColorMode_FixedColor,
        ColorMode_GreenRed,
        ColorMode_GreenWhite,
        ColorMode_Green,
        ColorMode_WhiteBlackGray
    };

    RenderSettings();

    bool draw_piano;
    bool draw_hz;
    bool draw_t;
    bool draw_cursor_marker;
    int draw_axis_at0;
    vectord cursor;

    bool draw_contour_plot;
    ColorMode color_mode;
    tvector<4, float> fixed_color;
    tvector<4, float> clear_color;
    float y_scale;
    float y_offset;

    /**
     * @brief y_normalize
     * y_scale and y_offset still apply even if y_normalize=true, but the value
     * on which they are applied is affected.
     */
    float y_normalize;
    TAni<> log_scale;
    float last_ysize;
    float last_axes_length;
    unsigned drawn_blocks;
    bool left_handed_axes;
    bool shadow_shader;
    bool draw_flat;
    bool drawcrosseswhen0;
    double dpifactor;
    int device_pixel_height;
    bool axes_border;
    float redundancy;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDERSETTINGS_H
