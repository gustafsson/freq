#ifndef TOOLS_SUPPORT_DRAWIMAGE_H
#define TOOLS_SUPPORT_DRAWIMAGE_H

#include <QPointF>
#include <QSize>
#include <boost/shared_ptr.hpp>

#include "GlTexture.h"
#include "vbo.h"

namespace Tools {
namespace Support {

class DrawImage
{
public:
    DrawImage(QString imagePath, QPointF pos = QPointF() );

    QPointF pos();
    void move(QPointF); // set position

    QSize size() const;

    void drawImage(int viewport_width, int viewport_height) const;
    void directDraw() const;
private:
    static boost::shared_ptr<GlTexture> loadImage(std::string imagePath);

    void setupVbo();

    boost::shared_ptr<GlTexture> img_;
    boost::shared_ptr<Vbo> postexvbo_;

    QPointF pos_;
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_DRAWIMAGE_H
