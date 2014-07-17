#ifndef TOOLS_SUPPORT_CHANNELCOLORS_H
#define TOOLS_SUPPORT_CHANNELCOLORS_H

#include <vector>
#include "tvector.h"

namespace Tools {
namespace Support {

class ChannelColors
{
public:
    static std::vector<tvector<4> > compute(unsigned N);
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_CHANNELCOLORS_H
