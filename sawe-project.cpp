#include "sawe-project.h"
#include "sawe-application.h"
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>
#include "signal-audiofile.h"
#include "signal-microphonerecorder.h"
#include "sawe-timelinewidget.h"

using namespace std;

namespace Sawe {

Project::
        Project( Signal::pSource head_source )
:   head_source(head_source)
{
}

Project::
        ~Project()
{
    TaskTimer tt("~Project");
    displayWidget()->setTimeline( Signal::pSink() );
    _timelineWidgetCallback.reset();
    _timelineWidget.reset();
    _displayWidget.reset();
    _mainWindow.reset();
}

pProject Project::
        open(std::string project_file_or_audio_file )
{
    string filename; filename.swap( project_file_or_audio_file );

    if (!filename.empty() && !QFile::exists(filename.c_str()))
    {
        QMessageBox::warning( 0,
                     QString("Can't find file"),
                     QString("File ") + QString::fromStdString(filename) );
        filename.clear();
    }

    if (0 == filename.length()) {
        string filter = Signal::getFileFormatsQtFilter( false ).c_str();
        filter = "All files (*.sonicawe " + filter + ");;";
        filter += "SONICAWE - Sonic AWE project (*.sonicawe);;";
        filter += Signal::getFileFormatsQtFilter( true ).c_str();

        QString qfilemame = QFileDialog::getOpenFileName(0, "Open file", NULL, QString::fromStdString(filter));
        if (0 == qfilemame.length()) {
            // User pressed cancel
            return pProject();
        }
        filename = qfilemame.toStdString();
    }

    string err;
    for (int i=0; i<2; i++) try { switch(i) {
        case 0: return Project::openProject( filename );
        case 1: return Project::openAudio( filename );
    }}
    catch (const exception& x) {
        if (!err.empty())
            err += '\n';
        err += x.what();
    }

    QMessageBox::warning( 0,
                 QString("Can't open file"),
                 QString::fromStdString(err) );
    return pProject();
}


pProject Project::
        createRecording(int record_device)
{
    Signal::pSource s( new Signal::MicrophoneRecorder(record_device) );
    return pProject( new Project( s ));
}


void Project::
        save(std::string /*project_file*/)
{
    // TODO implement
    throw std::runtime_error("TODO implement Project::save");
}


boost::shared_ptr<MainWindow> Project::
        mainWindow()
{
    createMainWindow();
    return _mainWindow;
}


DisplayWidget* Project::
        displayWidget()
{
    createMainWindow();
    return dynamic_cast<DisplayWidget*>(_displayWidget.get());
}


void Project::
        createMainWindow()
{
    if (_mainWindow)
        return;

    string title = Sawe::Application::version_string();
    Signal::Audiofile* af;
    if (0 != (af = dynamic_cast<Signal::Audiofile*>(head_source.get()))) {
        QFileInfo info( QString::fromStdString( af->filename() ));
        title = info.baseName().toStdString() + " - Sonic AWE";
    }

    _mainWindow.reset( new MainWindow( title.c_str()));

    Signal::pWorker wk( new Signal::Worker( head_source ) );
    Heightmap::Collection* sgp( new Heightmap::Collection(wk) );
    Signal::pSink sg( sgp );
    _displayWidget.reset( new DisplayWidget( wk, sg ) );

    _mainWindow->connectLayerWindow( displayWidget() );
    _mainWindow->setCentralWidget( displayWidget() );

    {
        _timelineWidget.reset( new TimelineWidget( _displayWidget ) );
        TaskTimer("_timelineWidget = %p, QGLWidget = %p, Sink = %p",
                  dynamic_cast<TimelineWidget*>(_timelineWidget.get()),
                  dynamic_cast<QGLWidget*>(_timelineWidget.get()),
                  _timelineWidget.get()).suppressTiming();
        _mainWindow->setTimelineWidget( dynamic_cast<QWidget*>(_timelineWidget.get()) );
    }

    //_displayWidgetCallback.reset( new Signal::WorkerCallback( displayWidget()->worker(), _displayWidget ));
    _timelineWidgetCallback.reset( new Signal::WorkerCallback( displayWidget()->worker(), _timelineWidget ));

    displayWidget()->setTimeline( _timelineWidget );
    displayWidget()->show();
    _mainWindow->hide();
    _mainWindow->show();
}


pProject Project::
        openProject(std::string /*project_file*/)
{
    // TODO implement
    throw std::runtime_error("TODO implement Project::openProject");
}


pProject Project::
        openAudio(std::string audio_file)
{
    Signal::pSource s( new Signal::Audiofile( audio_file.c_str() ) );
    return pProject( new Project( s ));
}

} // namespace Sawe
