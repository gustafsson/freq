#include "printscreen.h"

#include "sawe/project.h"

#include <QGLWidget>

namespace Tools {
namespace Support {

PrintScreen::
        PrintScreen(Sawe::Project* p)
    :
      p(p)
{
}


QImage PrintScreen::
        saveImage()
{
    return saveImage(p->tools().render_view()->glwidget);
}


QPixmap PrintScreen::
        saveWindowImage()
{
    return saveWindowImage(p->mainWindowWidget(), p->tools().render_view()->glwidget);
}


QImage PrintScreen::
        saveImage(QGLWidget *glwidget)
{
    TaskTimer ti("%s", __FUNCTION__);

    glwidget->swapBuffers();
    return glwidget->grabFrameBuffer();
}


QPixmap PrintScreen::
        saveWindowImage(QWidget* mainwindow, QGLWidget *glwidget)
{
    TaskTimer ti("%s", __FUNCTION__);

    QPixmap pixmap(mainwindow->size());
    QGL::setPreferredPaintEngine(QPaintEngine::OpenGL);
    QPainter painter(&pixmap);

    // update layout by calling render
    mainwindow->activateWindow();
    mainwindow->raise();
    mainwindow->render(&painter);

    // draw OpenGL window
    QPoint p2 = glwidget->mapTo( mainwindow, QPoint() );
    glwidget->swapBuffers();
    QImage glimage = glwidget->grabFrameBuffer();
    painter.drawImage(p2, glimage);

#ifdef Q_OS_LINUX
    // draw Qt widgets that are on top of the opengl window
    mainwindow->render(&painter);
#endif

    return pixmap;
}


} // namespace Support
} // namespace Tools
