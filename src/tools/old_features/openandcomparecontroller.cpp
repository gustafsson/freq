#if 0
#include "openandcomparecontroller.h"
#include "signal/operation-basic.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "adapters/audiofile.h"

// std
#include <string>

// Qt
#include <QFileDialog>

using namespace std;

namespace Tools {

OpenAndCompareController::
        OpenAndCompareController(Sawe::Project* project)
            :
            _project(project)
{
    setupGui();
}


OpenAndCompareController::
        ~OpenAndCompareController()
{
}

void OpenAndCompareController::
        setupGui()
{
    Ui::SaweMainWindow* main = _project->mainWindow();
    Ui::MainWindow* ui = main->getItems();

#if !defined(TARGET_reader)
    connect(ui->actionOpen_and_compare, SIGNAL(triggered()), this, SLOT(slotOpenAndCompare()));
#else
    ui->actionOpen_and_compare->setEnabled( false );
#endif
}


void OpenAndCompareController::
        slotOpenAndCompare()
{
#if !defined(TARGET_reader)
    try
    {
        string filter = Adapters::Audiofile::getFileFormatsQtFilter( false ).c_str();
        filter = "All files (" + filter + ");;";
        filter += Adapters::Audiofile::getFileFormatsQtFilter( true ).c_str();

        QString qfilename = QFileDialog::getOpenFileName(NULL, "Open file", "", QString::fromLocal8Bit(filter.c_str()));
        if (0 == qfilename.length()) {
            // User pressed cancel
            return;
        }
        string audio_file = qfilename.toLocal8Bit().data();

        Signal::pOperation secondAudio( new Adapters::Audiofile( QDir::current().relativeFilePath( audio_file.c_str() ).toStdString()) );

        Signal::pOperation sumChannels( new Signal::OperationAddChannels(
                _project->head->head_source(), secondAudio));

        //_project->tools().selection_model.set_current_selection(Signal::pOperation());
        _project->head->appendOperation(sumChannels);
    }
    catch (const std::exception& x)
    {
        TaskInfo("%s: %s", __FUNCTION__, x.what());
    }
#endif
}

} // namespace Tools
#endif
