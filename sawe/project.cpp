#include "sawe/project.h"
#include "sawe/application.h"
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>
#include "signal/audiofile.h"
#include "signal/microphonerecorder.h"
#include "sawe/timelinewidget.h"
#include <QVBoxLayout>
#include <sys/stat.h>

using namespace std;

namespace Sawe {

Project::
        Project( Signal::pOperation head_source )
:   head_source(head_source)
{
}

Project::
        ~Project()
{
    TaskTimer tt("~Project");
    if (_mainWindow)
        displayWidget()->setTimeline( 0 );
    _timelineWidgetCallback.reset();
    // _timelineWidget.reset(); // TODO howto clear QWidgets?
    // _displayWidget.reset();
    _mainWindow.reset();
}

pProject Project::
        open(std::string project_file_or_audio_file )
{
    string filename; filename.swap( project_file_or_audio_file );

    struct stat dummy;
    // QFile::exists doesn't work as expected can't handle unicode names
    if (!filename.empty() && 0!=stat( filename.c_str(),&dummy))
    {
        QMessageBox::warning( 0,
                     QString("Can't find file"),
                     QString("Can't find file '") + QString::fromLocal8Bit(filename.c_str()) + "'");
        filename.clear();
    }

    if (0 == filename.length()) {
        string filter = Signal::Audiofile::getFileFormatsQtFilter( false ).c_str();
        filter = "All files (*.sonicawe " + filter + ");;";
        filter += "SONICAWE - Sonic AWE project (*.sonicawe);;";
        filter += Signal::Audiofile::getFileFormatsQtFilter( true ).c_str();

		QString qfilemame = QFileDialog::getOpenFileName(0, "Open file", NULL, QString::fromLocal8Bit(filter.c_str()));
        if (0 == qfilemame.length()) {
            // User pressed cancel
            return pProject();
        }
        filename = qfilemame.toLocal8Bit().data();
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
				 QString::fromLocal8Bit(err.c_str()) );
    return pProject();
}


pProject Project::
        createRecording(int record_device)
{
    Signal::pOperation s( new Signal::MicrophoneRecorder(record_device) );
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
		QFileInfo info( QString::fromLocal8Bit( af->filename().c_str() ));
        title = string(info.baseName().toLocal8Bit()) + " - Sonic AWE";
    }

    _mainWindow.reset( new MainWindow( title.c_str()));

    Signal::pWorker wk( new Signal::Worker( head_source ) );
    Heightmap::pCollection cl( new Heightmap::Collection(wk) );
    // TODO Qt memory management?
    _displayWidget.reset( new DisplayWidget( wk, cl ) );

    _mainWindow->connectLayerWindow( displayWidget() );
    _mainWindow->setCentralWidget( displayWidget() );

    _mainWindow->setCorner( Qt::BottomLeftCorner, Qt::LeftDockWidgetArea );
    _mainWindow->setCorner( Qt::BottomRightCorner, Qt::RightDockWidgetArea );
    _mainWindow->setCorner( Qt::TopLeftCorner, Qt::LeftDockWidgetArea );
    _mainWindow->setCorner( Qt::TopRightCorner, Qt::RightDockWidgetArea );
    float L = displayWidget()->worker()->source()->length();
    L/=2;
    if (L>5) L = 5;
    displayWidget()->setPosition( L, 0.5f );

    {
        // TODO Qt memory management?
        _timelineWidget.reset( new TimelineWidget( dynamic_cast<QGLWidget*>(_displayWidget.get()) ));
        _mainWindow->setTimelineWidget( dynamic_cast<QGLWidget*>(_timelineWidget.get()) );
    }

    //_displayWidgetCallback.reset( new Signal::WorkerCallback( displayWidget()->worker(), _displayWidget ));
    _timelineWidgetCallback.reset( new Signal::WorkerCallback( displayWidget()->worker(), _timelineWidget ));

    displayWidget()->setTimeline( dynamic_cast<QGLWidget*>( _timelineWidget.get() ));
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
    Signal::pOperation s( new Signal::Audiofile( audio_file.c_str() ) );
    return pProject( new Project( s ));
}

} // namespace Sawe
