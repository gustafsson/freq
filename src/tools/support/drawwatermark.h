#ifndef DRAWWATERMARK_H
#define DRAWWATERMARK_H

#include <boost/shared_ptr.hpp>

class GlTexture;
class Vbo;

namespace Tools {
namespace Support {

class DrawWatermark
{
public:
    static void drawWatermark(int viewport_width, int viewport_height);

private:
    static void loadImage();
    static boost::shared_ptr<GlTexture> img;
    static boost::shared_ptr<Vbo> postexvbo;
};

} // namespace Support
} // namespace Tools

#endif // DRAWWATERMARK_H
