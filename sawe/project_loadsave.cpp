#include "project.h"

// Serializable Sonic AWE classes 
#include "adapters/audiofile.h"
#include "adapters/microphonerecorder.h"
#include "tools/support/brushfilter.h"
#include "filters/ellipse.h"
#include "filters/rectangle.h"
#include "signal/operationcache.h"

// Serializable Sonic AWE Tools
#include "tools/commentmodel.h"
#include "tools/tooltipmodel.h"

// GpuMisc
#include <demangle.h>

// Std
#include <fstream>

// Boost
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/algorithm/string.hpp>

// Qt
#include <QtGui/QMessageBox>
#include <QFileInfo>
#include <QDir>

BOOST_CLASS_VERSION(Tools::CommentModel, 1)

using namespace std;

namespace Sawe {

template<class Archive> 
void runSerialization(Archive& ar, Project*& project, QString path)
{
    TaskInfo ti("Running serialization");

    QDir dir = QDir::current();
    QDir::setCurrent( QFileInfo( path ).absolutePath() );

    ar.template register_type<Adapters::Audiofile>();
    ar.template register_type<Adapters::MicrophoneRecorder>();
    ar.template register_type<Tools::Support::MultiplyBrush>();
    ar.template register_type<Filters::Ellipse>();
    ar.template register_type<Filters::Rectangle>();
    ar.template register_type<Tools::CommentModel>();
    ar.template register_type<Tools::TooltipModel>();
    ar.template register_type<Tools::ToolFactory>();
    ar.template register_type<Tools::ToolRepo>();
    ar.template register_type<Tools::RenderModel>();
    ar.template register_type<Signal::OperationCacheLayer>();

    ar & boost::serialization::make_nvp("Sonic_AWE", project);

    QDir::setCurrent( dir.absolutePath() );
}


bool Project::
        save()
{
    if (project_file_name.empty()) {
        return saveAs();
    }

    QByteArray mainwindowState = _mainWindow->saveState();
    QByteArray mainwindowState2 = _mainWindow->saveState();
	BOOST_ASSERT( mainwindowState.size() == mainwindowState2.size());
	BOOST_ASSERT( 0 == memcmp(mainwindowState.data(), mainwindowState2.data(), mainwindowState2.size()));
    _mainWindow->restoreState( mainwindowState );

    try
    {
		TaskTimer tt("Saving project to '%s'", project_file_name.c_str());
        std::ofstream ofs(project_file_name.c_str());
        boost::archive::xml_oarchive xml(ofs);
		Project* p = this;
		runSerialization(xml, p, project_file_name.c_str());
    }
    catch (const std::exception& x)
    {
        QString msg = "Error: " + QString::fromStdString(vartype(x)) +
                      "\nDetails: " + QString::fromLocal8Bit(x.what());
        QMessageBox::warning( 0, "Can't save file", msg );
        TaskInfo("======================\nCan't save file\n%s\n======================", msg.toStdString().c_str());
    }

    return true;
}


pProject Project::
        openProject(std::string project_file)
{
    std::ifstream ifs(project_file.c_str());
    string xmltest;
	xmltest.resize(5);

    ifs.read( &xmltest[0], 5 );
    if( !boost::iequals( xmltest, "<?xml") )
        throw std::invalid_argument("Project file '" + project_file + "' is not an xml file");

    for (int i=xmltest.size()-1; i>=0; i--)
        ifs.putback( xmltest[i] );

    boost::archive::xml_iarchive xml(ifs);

    Project* new_project = 0;
	runSerialization(xml, new_project, project_file.c_str());

    new_project->project_file_name = project_file;

    pProject project( new_project );

    return project;
}

} // namespace Sawe
