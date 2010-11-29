#include "project.h"

// Serializable Sonic AWE classes 
#include "adapters/audiofile.h"
#include "tools/support/brushfilter.h"
#include "filters/ellipse.h"

// Serializable Sonic AWE Tools
#include "tools/commentmodel.h"

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

using namespace std;

namespace Sawe {

template<class Archive> 
void runSerialization(Archive& ar, Project*& project)
{
	ar.template register_type<Adapters::Audiofile>();
	ar.template register_type<Tools::Support::MultiplyBrush>();
	ar.template register_type<Filters::Ellipse>();
        ar.template register_type<Tools::CommentModel>();

	ar & boost::serialization::make_nvp("Sonic_AWE", project);
}


void Project::
        save()
{
    if (project_file_name.empty()) {
        saveAs();
        return;
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
		runSerialization(xml, p);
    }
    catch (const std::exception& x)
    {
        QMessageBox::warning( 0,
                     QString("Can't save file"),
					 "Error: " + QString::fromStdString(vartype(x)) + 
					 "\nDetails: " + QString::fromLocal8Bit(x.what()) );
    }
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
	runSerialization(xml, new_project);

    new_project->project_file_name = project_file;

    pProject project( new_project );

    return project;
}

} // namespace Sawe
