#include "sawe/project.h"

#include "sawe/application.h"
#if !defined(TARGET_reader)
#include "adapters/audiofile.h"
#endif
#include "adapters/microphonerecorder.h"
#include "signal/operationcache.h"
#include "tools/toolfactory.h"
#include "tools/support/operation-composite.h"
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
    is_modified_(false),
    project_title_(layer_title)
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


void Project::
        appendOperation(Signal::pOperation s)
{
    Tools::SelectionModel& m = tools().selection_model;

    if (m.current_selection() && m.current_selection()!=s)
    {
        Signal::pOperation onselectionOnly(new Tools::Support::OperationOnSelection(
                head->head_source(),
                Signal::pOperation( new Signal::OperationCachedSub(
                    m.current_selection_copy( Tools::SelectionModel::SaveInside_TRUE ))),
                Signal::pOperation( new Signal::OperationCachedSub(
                    m.current_selection_copy( Tools::SelectionModel::SaveInside_FALSE ))),
                s
                ));

        s = onselectionOnly;
    }

    this->head->appendOperation( s );

    tools().render_model.renderSignalTarget->findHead( head->chain() )->head_source( head->head_source() );
    tools().playback_model.playbackTarget->findHead( head->chain() )->head_source( head->head_source() );

    setModified();
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
        string filter;
#if !defined(TARGET_reader)
        filter = Adapters::Audiofile::getFileFormatsQtFilter( false ).c_str();
#endif
        filter = "All files (*.sonicawe *.sonicawe " + filter + ");;";
        filter += "SONICAWE - Sonic AWE project (*.sonicawe);;";
#if !defined(TARGET_reader)
        filter += Adapters::Audiofile::getFileFormatsQtFilter( true ).c_str();
#endif

        QString qfilename = QFileDialog::getOpenFileName(NULL, "Open file", "", QString::fromLocal8Bit(filter.c_str()));
        if (0 == qfilename.length()) {
            // User pressed cancel
            return pProject();
        }
        filename = qfilename.toLocal8Bit().data();
    }

    string err;
    pProject p;
    int availableFileTypes = 1;
#if !defined(TARGET_reader)
    availableFileTypes++;
#endif
    for (int i=0; i<availableFileTypes; i++) try
    {
        switch(i) {
            case 0: p = Project::openProject( filename ); break;
#if !defined(TARGET_reader)
            case 1: p = Project::openAudio( filename ); break;
#endif
        }
        break; // successful loading without thrown exception
    }
    catch (const exception& x) {
        if (!err.empty())
            err += '\n';
        err += "Error: " + vartype(x);
        err += "\nDetails: " + (std::string)x.what();
    }

    if (!p)
    {
        QMessageBox::warning( 0, "Can't open file", QString::fromLocal8Bit(err.c_str()) );
        TaskInfo("======================\n"
                 "Can't open file '%s' as project nor audio file\n"
                 "%s\n"
                 "======================",
                 filename.c_str(),
                 err.c_str());
        return pProject();
    }

    addRecentFile( filename );

    return p;
}


void Project::
        addRecentFile( std::string filename )
{
    QSettings settings;
    QStringList recent_files = settings.value("recent files").toStringList();
    QFileInfo fi(QString::fromLocal8Bit( filename.c_str() ));
    fi.makeAbsolute();
    QString qfilename = fi.canonicalFilePath();
    if (!qfilename.isEmpty())
    {
        recent_files.removeAll( qfilename );
        recent_files.push_front( qfilename );
        while (recent_files.size()>8)
            recent_files.pop_back();
        settings.setValue("recent files", recent_files);
    }
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
        project_title()
{
    return project_title_;
}


std::string Project::
        project_filename()
{
    return QFileInfo(QString::fromLocal8Bit( project_filename_.c_str() )).fileName().toStdString();
}



Project::
        Project()
            :
            worker(Signal::pTarget()),
            layers(this),
            is_modified_(false)
{}


void Project::
        createMainWindow()
{
    if (_mainWindow)
        return;

    TaskTimer tt("Project::createMainWindow");
    string title = Sawe::Application::version_string();
    if (!project_title().empty())
        title = project_title() + " - " + title;

    _mainWindow = new Ui::SaweMainWindow( title.c_str(), this );

    {
        TaskTimer tt("new Tools::ToolFactory");
        _tools.reset( new Tools::ToolFactory(this) );
        tt.info("Created tools");
    }

    defaultGeometry = _mainWindow->saveGeometry();
    defaultState = _mainWindow->saveState();

    QSettings settings;
    _mainWindow->restoreGeometry(settings.value("geometry").toByteArray());
    _mainWindow->restoreState(settings.value("windowState").toByteArray());

    // don't start in fullscreen mode
    dynamic_cast<Ui::SaweMainWindow*>(_mainWindow.data())->disableFullscreen();

    _mainWindow->show();

    Sawe::Application::check_license();
    updateWindowTitle();
}


void Project::
        updateWindowTitle()
{
    if (!project_filename_.empty())
        project_title_ = QFileInfo(QString::fromLocal8Bit( project_filename_.c_str() )).fileName().toStdString();
    _mainWindow->setWindowTitle( QString::fromLocal8Bit( (project_title() + " - " + Sawe::Application::version_string()).c_str() ));
}


void Project::
        restoreDefaultLayout()
{
    QSettings settings;
    _mainWindow->restoreGeometry(defaultGeometry);
    _mainWindow->restoreState(defaultState);
    QString value = settings.value("value").toString();
    settings.clear();
    settings.setValue("value", value);
    settings.setValue("geometry", _mainWindow->saveGeometry());
    settings.setValue("windowState", _mainWindow->saveState());
}


#if !defined(TARGET_reader)
bool Project::
        saveAs()
{
    QString filter = "SONICAWE - Sonic AWE project (*.sonicawe)";

    QString qfilename = QFileDialog::getSaveFileName(mainWindow(), "Save project", QString::fromStdString(project_filename_), filter);
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

    bool r = save();

    addRecentFile( project_filename_ );

    return r;
}
#endif

#if !defined(TARGET_reader)
pProject Project::
        openAudio(std::string audio_file)
{
    Adapters::Audiofile*a;
    Signal::pOperation s( a = new Adapters::Audiofile( QDir::current().relativeFilePath( audio_file.c_str() ).toStdString()) );
    return pProject( new Project( s, a->name() ));
}
#endif

} // namespace Sawe
