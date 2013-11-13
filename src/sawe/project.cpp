#include "sawe/project.h"

#include "sawe/application.h"
#include "sawe/openfileerror.h"
#include "sawe/configuration.h"
#if !defined(TARGET_reader)
#include "adapters/audiofile.h"
#include "adapters/csvtimeseries.h"
#endif
#include "adapters/microphonerecorder.h"
#include "adapters/networkrecorder.h"
#include "signal/operationcache.h"
#include "signal/oldoperationwrapper.h"
#include "tools/toolfactory.h"
#include "tools/support/operation-composite.h"
#include "tools/commands/commandinvoker.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "tools/commands/appendoperationdesccommand.h"

// Qt
#include <QFileDialog>
#include <QVBoxLayout>
#include <QMessageBox>
#include <QSettings>
#include <QInputDialog>

// Std
#include <sys/stat.h>

using namespace std;

namespace Sawe {

Project::
        Project( std::string project_title )
:   is_modified_(false),
    is_sawe_project_(false),
    project_title_(project_title)
{
    // class Project has two constructors. Initialize common stuff in createMainWindow instead of here.
}


Project::
        ~Project()
{
    TaskTimer tt("~Project");

    TaskInfo("project_title = %s", project_title().c_str());
    TaskInfo("project_filename = %s", project_filename().c_str());

    this->processing_chain_.reset ();

    {
        TaskInfo ti("releasing tool resources");
        _tools.reset();
    }

    if (_mainWindow)
        delete _mainWindow;

    TaskInfo("Closed project");
}


void Project::
        appendOperation(Signal::pOperation s)
{
    Signal::OperationDesc::Ptr oodw(new Signal::OldOperationDescWrapper(s));
    appendOperation(oodw);
}


void Project::
        appendOperation(Signal::OperationDesc::Ptr s)
{
    Tools::Commands::pCommand c(
                new Tools::Commands::AppendOperationDescCommand(
                    s,
                    this->processing_chain (),
                    default_target())
                );

    commandInvoker()->invokeCommand(c);

    // TODO recompute_extent must be called again if the operation is undone.
    tools().render_model.recompute_extent();

    setModified ();
}


/*
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
*/


Tools::ToolRepo& Project::
        toolRepo()
{
    if (!areToolsInitialized())
        EXCEPTION_ASSERTX(false, "tools() was called before createMainWindow()");

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
    return _tools.get ();
}


pProject Project::
        open(std::string project_file_or_audio_file )
{
    string filename; filename.swap( project_file_or_audio_file );

    // QFile::exists doesn't work as expected with unicode names
    struct stat dummy;
    bool fileExists = 0==stat( filename.c_str(),&dummy);

    if (!filename.empty() && !fileExists)
    {
        QUrl url(filename.c_str());
        if (url.isValid() && !url.scheme().isEmpty())
        {
            std::string scheme = url.scheme().toStdString();
            Signal::pOperation s( new Adapters::NetworkRecorder(url) );

            pProject p( new Project( "New network recording" ));
            p->createMainWindow ();
            p->tools ().addRecording (new Adapters::NetworkRecorder(url));

            return p;
        }

        QMessageBox::warning( 0,
                     QString("Can't find file"),
                     QString("Can't find file '") + QString::fromLocal8Bit(filename.c_str()) + "'");
        filename.clear();
    }

    if (filename.empty()) {
        string filter;
#if !defined(TARGET_reader)
        filter += " " + Adapters::Audiofile::getFileFormatsQtFilter( false );
#endif
#if !defined(TARGET_reader) && !defined(TARGET_hast)
        filter += " " + Adapters::CsvTimeseries::getFileFormatsQtFilter( false );
#endif
        if (Sawe::Configuration::feature("stable")) {
            filter = "All files (" + filter + ");;";
        } else {
            filter = "All files (*.sonicawe *.sonicawe" + filter + ");;";
            filter += "SONICAWE - Sonic AWE project (*.sonicawe)";
        }
#if !defined(TARGET_reader)
        filter += ";;" + Adapters::Audiofile::getFileFormatsQtFilter( true );
#endif
#if !defined(TARGET_reader) && !defined(TARGET_hast)
        filter += ";;" + Adapters::CsvTimeseries::getFileFormatsQtFilter( true );
#endif

        QString qfilename = QFileDialog::getOpenFileName(NULL, "Open file", "", QString::fromLocal8Bit(filter.c_str()));
        if (0 == qfilename.length()) {
            // User pressed cancel
            return pProject();
        }
        filename = qfilename.toLocal8Bit().data();
    }

    string err;
    string openfile_err;
    pProject p;
    if (0!=stat( filename.c_str(),&dummy))
        err = "File '" + filename + "' does not exist";
    else
    {
        int availableFileTypes = 1;
    #if !defined(TARGET_reader)
        availableFileTypes++;
    #endif
    #if !defined(TARGET_reader) && !defined(TARGET_hast)
        availableFileTypes++;
    #endif

        string suffix = QFileInfo(filename.c_str()).completeSuffix().toLower().toStdString();
        int expected = -1;
        if (suffix == "sonicawe") expected = 0;
#if !defined(TARGET_reader)
        if (Adapters::Audiofile::hasExpectedSuffix(suffix)) expected = 1;
#endif
#if !defined(TARGET_reader) && !defined(TARGET_hast)
        if (Adapters::CsvTimeseries::hasExpectedSuffix(suffix)) expected = 2;
#endif

        int i = 0;
        if (expected >= 0)
            i = expected, availableFileTypes=expected+1;

        for (; i<availableFileTypes; i++) try
        {
            switch(i) {
                case 0: p = Project::openProject( filename ); break;
    #if !defined(TARGET_reader)
                case 1: p = Project::openAudio( filename ); break;
    #endif
    #if !defined(TARGET_reader) && !defined(TARGET_hast)
                case 2: p = Project::openCsvTimeseries( filename ); break;
    #endif
            }
            break; // successful loading without thrown exception
        }
        catch (const OpenFileError& x) {
            if (!openfile_err.empty())
                openfile_err += '\n';
            openfile_err += x.what();
        }
        catch (const exception& x) {
            if (!err.empty())
                err += '\n';
            err += boost::diagnostic_information(x);
        }
    }

    if (!p)
    {
        if (!openfile_err.empty())
            err = openfile_err;

        QMessageBox::warning(
                    0,
                    "Can't open file",
                    QString("Can't open file\n%1")
                        .arg (filename.c_str ())
                    );

        TaskInfo(boost::format(
                 "======================\n"
                 "Can't open file '%s'\n"
                 "%s\n"
                 "======================")
                 % filename
                 % err);
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
        createRecording()
{
    int device = QSettings().value("inputdevice", -1).toInt();

    pProject p( new Project( "New recording" ));
    p->createMainWindow ();
    p->tools ().addRecording (new Adapters::MicrophoneRecorder(device));

    return p;
}


bool Project::
        isModified()
{
    return is_modified_;
}


void Project::
        setModified( bool is_modified )
{
    is_modified_ = is_modified;
}


Tools::Commands::CommandInvoker* Project::
        commandInvoker()
{
    return command_invoker_.get();
}


Ui::SaweMainWindow* Project::
        mainWindow()
{
    createMainWindow();
    return dynamic_cast<Ui::SaweMainWindow*>(_mainWindow.data());
}


QWidget* Project::
        mainWindowWidget()
{
    return dynamic_cast<QWidget*>(mainWindow());
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
            //worker(Signal::pTarget()),
            //layers(this),
            is_modified_(false),
            is_sawe_project_(true)
{}


void Project::
        createMainWindow()
{
    if (_mainWindow)
        return;

    TaskTimer tt("Project::createMainWindow");

    processing_chain_ = Signal::Processing::Chain::createDefaultChain ();

    command_invoker_.reset( new Tools::Commands::CommandInvoker(this) );

    string title = Sawe::Configuration::title_string();
    if (!project_title().empty())
        title = project_title() + " - " + title;

    Ui::SaweMainWindow* saweMain = 0;
    _mainWindow = saweMain = new Ui::SaweMainWindow( title.c_str(), this );

    _tools.reset( new Tools::ToolFactory(this) );

    defaultGuiState = saweMain->saveSettings();

    saweMain->restoreSettings( QSettings().value("GuiState").toByteArray() );

    // don't start in fullscreen mode
    saweMain->disableFullscreen();

    _mainWindow->show();

    Sawe::Application::check_license();
    updateWindowTitle();

    is_modified_ = false;
}


void Project::
        updateWindowTitle()
{
    if (!project_filename_.empty())
        project_title_ = QFileInfo(QString::fromLocal8Bit( project_filename_.c_str() )).fileName().toStdString();
    _mainWindow->setWindowTitle( QString::fromLocal8Bit( (project_title() + " - " + Sawe::Configuration::title_string()).c_str() ));
}


QByteArray Project::
        getGuiState() const
{
    Ui::SaweMainWindow* saweMain = dynamic_cast<Ui::SaweMainWindow*>(_mainWindow.data());
    return saweMain->saveSettings();
}


void Project::
        setGuiState(QByteArray guiState)
{
    Ui::SaweMainWindow* saweMain = dynamic_cast<Ui::SaweMainWindow*>(_mainWindow.data());
    saweMain->restoreSettings( guiState );
}


void Project::
        resetLayout()
{
    setGuiState( defaultGuiState );
}


void Project::
        resetView()
{
    tools().render_view()->model->resetSettings();
    Application::global_ptr()->clearCaches();
    tools().render_view()->userinput_update( false );
}


bool Project::
        isSaweProject()
{
    return is_sawe_project_;
}


Signal::Processing::TargetMarker::Ptr Project::
        default_target()
{
    return tools().render_model.target_marker();
}


Signal::OperationDesc::Extent Project::
        extent()
{
    Signal::OperationDesc::Extent x;

    if (areToolsInitialized())
        x = read1(processing_chain_)->extent(this->default_target ());

    if (!x.interval.is_initialized ())
        x.interval = Signal::Interval();
    if (!x.number_of_channels.is_initialized ())
        x.number_of_channels = 0;
    if (!x.sample_rate.is_initialized ())
        x.sample_rate = 1;
    return x;
}


float Project::
        length()
{
    Signal::OperationDesc::Extent x = extent();
    return x.interval.get ().count() / x.sample_rate.get ();
}


#if !defined(TARGET_reader)
bool Project::
        saveAs()
{
    QString filter = "SONICAWE - Sonic AWE project (*.sonicawe)";

    QString qfilename = QString::fromStdString(project_filename_);
    do
    {
        qfilename = QFileDialog::getSaveFileName(mainWindow(), "Save project", qfilename, filter);
    } while (!qfilename.isEmpty() && QDir(qfilename).exists()); // try again if a directory was selected

    if (0 == qfilename.length()) {
        // User pressed cancel
        return false;
    }

    QString extension = ".sonicawe";
    if (qfilename.length() < extension.length())
        qfilename += extension;
    if (0 != QString::compare(qfilename.mid(qfilename.length() - extension.length()), extension, Qt::CaseInsensitive))
        qfilename += extension;

    return saveAs( qfilename.toLocal8Bit().data() );
}


bool Project::
        saveAs(std::string newprojectfilename)
{
    project_filename_ = newprojectfilename;

    updateWindowTitle();

    bool r = save();

    addRecentFile( project_filename_ );

    setModified (false);

    return r;
}
#endif


#if !defined(TARGET_reader)
pProject Project::
        openAudio(std::string audio_file)
{
    std::string path = QDir::current().relativeFilePath( audio_file.c_str() ).toStdString();

    boost::shared_ptr<Adapters::Audiofile> a( new Adapters::Audiofile( path ) );
    Signal::OperationDesc::Ptr d(new Adapters::AudiofileDesc(a));

    pProject p( new Project(a->name() ));
    p->createMainWindow ();
    write1(p->tools ().render_model.tfr_mapping ())->channels(a->num_channels ());
    p->appendOperation (d);
    p->setModified (false);

    return p;
}
#endif

#if !defined(TARGET_reader) && !defined(TARGET_hast)
pProject Project::
        openCsvTimeseries(std::string audio_file)
{
    Adapters::CsvTimeseries*a;
    Signal::pOperation s( a = new Adapters::CsvTimeseries( QDir::current().relativeFilePath( audio_file.c_str() ).toStdString()) );
    double fs = QInputDialog::getDouble (0, "Sample rate",
                                           "Enter the sample rate for the csv data:", 1);
    if (fs<=0)
        fs = 1;

    a->set_sample_rate (fs);
    pProject p( new Project( a->name() ));
    p->createMainWindow ();
    write1(p->tools ().render_model.tfr_mapping ())->channels(a->num_channels ());
    p->appendOperation (s);
    p->mainWindow()->getItems()->actionTransform_info->setChecked( true );
    p->setModified (false);

    return p;
}
#endif

} // namespace Sawe
