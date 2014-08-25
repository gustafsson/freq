#include "channelcolors.h"

#include <QColor>

namespace Tools {
namespace Support {

std::vector<tvector<4> > ChannelColors::
        compute(unsigned N)
{
    std::vector<tvector<4> > channel_colors( N );

    // Set colors
    float R = 0, G = 0, B = 0;
    for (unsigned i=0; i<N; ++i)
    {
        QColor c = QColor::fromHsvF( i/(float)N, 1, 1 );
        channel_colors[i] = tvector<4>(c.redF(), c.greenF(), c.blueF(), c.alphaF());
        R += channel_colors[i][0];
        G += channel_colors[i][1];
        B += channel_colors[i][2];
    }

    // R, G and B sum up to the same constant = N/2 if N > 1
    for (unsigned i=0; i<N; ++i)
        channel_colors[i] = channel_colors[i] * (N/2.f);

    if (1==N) // There is a grayscale mode to use for this
        channel_colors[0] = tvector<4>(0,0,0,1);

    return channel_colors;
}

} // namespace Support
} // namespace Tools
