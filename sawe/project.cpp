#include "sawe/project.h"
#include "sawe/application.h"
#include "adapters/audiofile.h"
#include "adapters/microphonerecorder.h"
#include "tools/toolfactory.h"
#include "ui/mainwindow.h"

#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>
#include <QVBoxLayout>
#include <sys/stat.h>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <fstream>

using namespace std;

namespace Sawe {

Project::
        Project( Signal::pOperation head_source )
:   worker( head_source )
{
}


Project::
        ~Project()
{
    TaskTimer tt("~Project");
}


Tools::ToolFactory& Project::
        tools()
{
    if (!_tools)
        throw std::logic_error("tools() was called before createMainWindow()");

    return *_tools;
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
        filter = "All files (*.sonicawe " + filter + ");;";
        filter += "SONICAWE - Sonic AWE project (*.sonicawe);;";
        filter += Adapters::Audiofile::getFileFormatsQtFilter( true ).c_str();

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
    Signal::pOperation s( new Adapters::MicrophoneRecorder(record_device) );
    return pProject( new Project( s ));
}


Ui::SaweMainWindow* Project::
        mainWindow()
{
    createMainWindow();
    return dynamic_cast<Ui::SaweMainWindow*>(_mainWindow.data());
}


Project::
        Project()
{}


void Project::
        createMainWindow()
{
    if (_mainWindow)
        return;

    string title = Sawe::Application::version_string();
    Adapters::Audiofile* af;
    if (0 != (af = dynamic_cast<Adapters::Audiofile*>(worker.source().get()))) {
		QFileInfo info( QString::fromLocal8Bit( af->filename().c_str() ));
        title = string(info.baseName().toLocal8Bit()) + " - Sonic AWE";
    }

    _mainWindow.reset( new Ui::SaweMainWindow( title.c_str(), this ));

    _tools.reset( new Tools::ToolFactory(this) );
}


void Project::
        save(std::string project_file)
{
    if (project_file.empty()) {
        string filter = "SONICAWE - Sonic AWE project (*.sonicawe);;";

        QString qfilemame = QFileDialog::getSaveFileName(0, "Save project", NULL, QString::fromLocal8Bit(filter.c_str()));
        if (0 == qfilemame.length()) {
            // User pressed cancel
            return;
        }
        project_file = qfilemame.toLocal8Bit().data();
    }

    try
    {
        // todo use
        std::ofstream ofs(project_file.c_str());
        boost::archive::xml_oarchive xml(ofs);
        xml << boost::serialization::make_nvp("Sonicawe", this);
    }
    catch (const std::exception& x)
    {
        QMessageBox::warning( 0,
                     QString("Can't save file"),
                     QString::fromLocal8Bit(x.what()) );
    }
}


pProject Project::
        openProject(std::string project_file)
{
    // todo use
    std::ifstream ifs(project_file.c_str());
    boost::archive::xml_iarchive xml(ifs);

    Project* new_project;
    xml >> boost::serialization::make_nvp("SonicaweProject", new_project);

    pProject project( new_project );

    return project;
}


pProject Project::
        openAudio(std::string audio_file)
{
    Signal::pOperation s( new Adapters::Audiofile( audio_file.c_str() ) );
    return pProject( new Project( s ));
}

} // namespace Sawe
