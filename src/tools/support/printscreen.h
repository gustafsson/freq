#ifndef TOOLS_SUPPORT_PRINTSCREEN_H
#define TOOLS_SUPPORT_PRINTSCREEN_H

#include "sawe/sawedll.h"

class QWidget;
class QGLWidget;
class QImage;
class QPixmap;
namespace Sawe { class Project; }

namespace Tools {
namespace Support {

class SaweDll PrintScreen
{
public:
    PrintScreen(Sawe::Project* p);

    QImage saveImage();
    QPixmap saveWindowImage();

    QImage saveImage(QGLWidget *glwidget);
    QPixmap saveWindowImage(QWidget* mainwindow, QGLWidget *glwidget);

private:
    Sawe::Project* p;
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_PRINTSCREEN_H
