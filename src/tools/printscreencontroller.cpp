#include "printscreencontroller.h"

#include "sawe/project.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "support/printscreen.h"

#include <QImage>
#include <QFileDialog>

namespace Tools {

PrintScreenController::
        PrintScreenController(Sawe::Project* p)
:
    project_(p)
{
    Ui::MainWindow* items = p->mainWindow ()->getItems ();
    connect(items->menuTools->addAction ("Save screen dump"), SIGNAL(triggered()), SLOT(takePrintScreen()));
}


void PrintScreenController::
        takePrintScreen()
{
    // Supported file formats
    // http://doc.qt.digia.com/qt/qimage.html#reading-and-writing-image-files
    QString filter = "Image (*.png *.bmp *.jpg *.jpeg *.ppm *.tiff *.xbm *.xpm)";

    QString qfilename;
    do
    {
        qfilename = QFileDialog::getSaveFileName(project_->mainWindow(), "Save print screen image", qfilename, filter);
    } while (!qfilename.isEmpty() && QDir(qfilename).exists()); // try again if a directory was selected

    if (qfilename.isEmpty ())
        return; // abort

    Support::PrintScreen(project_).saveImage ().save(qfilename);
}

} // namespace Tools
