#include "sawe/project.h"

#include "sawe/application.h"
#include "adapters/audiofile.h"
#include "adapters/microphonerecorder.h"
#include "tools/toolfactory.h"
#include "ui/mainwindow.h"

// Qt
#include <QtGui/QFileDialog>
#include <QVBoxLayout>
#include <QtGui/QMessageBox>
#include <QSettings>

// Std
#include <sys/stat.h>

using namespace std;

namespace Sawe {

Project::
        Project( Signal::pOperation root, std::string layer_title )
:   worker(Signal::pTarget()),
    layers(this),
    is_modified_(true)
{
    Signal::pChain chain(new Signal::Chain(root));
    chain->name = layer_title;
    layers.addLayer( chain );
    head.reset( new Signal::ChainHead(chain) );
}


Project::
        ~Project()
{
    TaskTimer tt("~Project");

    _tools.reset();

    if (_mainWindow)
        delete _mainWindow;
}


Tools::ToolRepo& Project::
        toolRepo()
{
    if (!areToolsInitialized())
        throw std::logic_error("tools() was called before createMainWindow()");

    return *_tools;
}


Tools::ToolFactory& Project::
        tools()
{
    return *dynamic_cast<Tools::ToolFactory*>(&toolRepo());
}


bool Project::
        areToolsInitialized()
{
    return _tools;
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
        string filter = Adapters::Audiofile::getFileFormatsQtFilter( false ).c_str();
        filter = "All files (*.sonicawe *.sonicawe " + filter + ");;";
        filter += "SONICAWE - Sonic AWE project (*.sonicawe);;";
        filter += Adapters::Audiofile::getFileFormatsQtFilter( true ).c_str();

        QString qfilename = QFileDialog::getOpenFileName(NULL, "Open file", "", QString::fromLocal8Bit(filter.c_str()));
        if (0 == qfilename.length()) {
            // User pressed cancel
            return pProject();
        }
        filename = qfilename.toLocal8Bit().data();
    }

    string err;
    for (int i=0; i<2; i++) try { switch(i) {
        case 0: return Project::openProject( filename );
        case 1: return Project::openAudio( filename );
    }}
    catch (const exception& x) {
        if (!err.empty())
            err += '\n';
        err += "Error: " + vartype(x);
        err += "\nDetails: " + (std::string)x.what();
    }

    QMessageBox::warning( 0, "Can't open file", QString::fromLocal8Bit(err.c_str()) );
    TaskInfo("======================\nCan't open file\n%s\n======================", err.c_str());
    return pProject();
}


pProject Project::
        createRecording(int record_device)
{
    Signal::pOperation s( new Adapters::MicrophoneRecorder(record_device) );
    return pProject( new Project( s, "New recording" ));
}


bool Project::
        isModified()
{
    return is_modified_;
}


void Project::
        setModified()
{
    is_modified_ = true;
}


Ui::SaweMainWindow* Project::
        mainWindow()
{
    createMainWindow();
    return dynamic_cast<Ui::SaweMainWindow*>(_mainWindow.data());
}


std::string Project::
        project_name()
{
    return QFileInfo(QString::fromLocal8Bit( project_filename_.c_str() )).fileName().toStdString();
}


Project::
        Project()
            :
            worker(Signal::pTarget()),
            layers(this),
            is_modified_(true)
{}


void Project::
        createMainWindow()
{
    if (_mainWindow)
        return;

    TaskTimer tt("Project::createMainWindow");
    string title = Sawe::Application::version_string();
    if (!project_name().empty())
        title = project_name() + " - " + title;

    _mainWindow = new Ui::SaweMainWindow( title.c_str(), this );

    {
        TaskTimer tt("new Tools::ToolFactory");
        _tools.reset( new Tools::ToolFactory(this) );
        tt.info("Created tools");
    }

    QSettings settings("REEP", "Sonic AWE");
    _mainWindow->restoreGeometry(settings.value("geometry").toByteArray());
    _mainWindow->restoreState(settings.value("windowState").toByteArray());
}


void Project::
        updateWindowTitle()
{
    _mainWindow->setWindowTitle( (project_name() + " - " + Sawe::Application::version_string()).c_str() );
}


bool Project::
        saveAs()
{
    QString filter = "SONICAWE - Sonic AWE project (*.sonicawe)";

    QString qfilename = QFileDialog::getSaveFileName(mainWindow(), "Save project", "", filter);
    if (0 == qfilename.length()) {
        // User pressed cancel
        return false;
    }

    QString extension = ".sonicawe";
    if (qfilename.length() < extension.length())
        qfilename += extension;
    if (0 != QString::compare(qfilename.mid(qfilename.length() - extension.length()), extension, Qt::CaseInsensitive))
        qfilename += extension;

    project_filename_ = qfilename.toLocal8Bit().data();

    updateWindowTitle();

    return save();
}


pProject Project::
        openAudio(std::string audio_file)
{
    Signal::pOperation s( new Adapters::Audiofile( audio_file.c_str() ) );
    return pProject( new Project( s, QFileInfo( audio_file.c_str() ).fileName().toStdString() ));
}

} // namespace Sawe
