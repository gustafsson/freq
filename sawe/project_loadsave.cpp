// settings
#define SAWE_USE_XML
#ifdef _DEBUG
#define SAWE_USE_XML
#endif

// class header
#include "project.h"

// Serializable Sonic AWE classes 
#include "adapters/audiofile.h"
#include "adapters/matlaboperation.h"
#include "adapters/microphonerecorder.h"
#include "tools/support/brushfilter.h"
#include "filters/ellipse.h"
#include "filters/rectangle.h"
#include "signal/operationcache.h"
#include "signal/chain.h"
#include "signal/target.h"
#include "signal/operation-basic.h"

// Serializable Sonic AWE Tools
#include "tools/commentmodel.h"
#include "tools/tooltipmodel.h"
#include "tools/selections/support/splinefilter.h"
#include "tools/support/operation-composite.h"

// GpuMisc
#include <demangle.h>

// Std
#include <fstream>

// Boost
#ifdef SAWE_USE_XML
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#else
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#endif
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
    TaskInfo ti("Running %s from '%s'", typename Archive::is_loading()?"deserialization":"serialization", path.toLocal8Bit().data());

    QDir dir = QDir::current();
    QDir::setCurrent( QFileInfo( path ).absolutePath() );

    TaskInfo("Current path is '%s'", QDir::currentPath().toLocal8Bit().data());

    ar.template register_type<Adapters::Audiofile>();
    ar.template register_type<Adapters::MicrophoneRecorder>();
    ar.template register_type<Tools::Support::MultiplyBrush>();
    ar.template register_type<Filters::Ellipse>();
    ar.template register_type<Filters::Rectangle>();
    ar.template register_type<Tools::Selections::Support::SplineFilter>();
    ar.template register_type<Tools::Selections::Support::SplineFilter::SplineVertex>();
    ar.template register_type<Tools::CommentModel>();
    ar.template register_type<Tools::TooltipModel>();
    ar.template register_type<Tools::ToolFactory>();
    ar.template register_type<Tools::ToolRepo>();
    ar.template register_type<Tools::RenderModel>();
    ar.template register_type<Signal::OperationCacheLayer>();
    ar.template register_type<Signal::Layers>();
    ar.template register_type<Signal::Chain>();
    ar.template register_type<Signal::ChainHead>();
    ar.template register_type<Adapters::MatlabOperation>();
    ar.template register_type<Project>();
    ar.template register_type<Signal::OperationCachedSub>();
    ar.template register_type<Signal::OperationSuperposition>();
    ar.template register_type<Signal::OperationSetSilent>();
    ar.template register_type<Signal::OperationRemoveSection>();
    ar.template register_type<Tools::Support::OperationSubOperations>();
    ar.template register_type<Tools::Support::OperationOnSelection>();
    ar.template register_type<Tools::Support::OperationCrop>();

    // add new types at the end to preserve backwards compatibility
	
	const unsigned magicConst=74610957;
	unsigned magic = magicConst;
    ar & boost::serialization::make_nvp("Magic",magic);
    if (magic != magicConst)
        throw std::ios_base::failure("Not a Sonic AWE project");
	

    ar & boost::serialization::make_nvp("Sonic_AWE",project);
	

    QDir::setCurrent( dir.absolutePath() );
    TaskInfo("Current path is '%s'", QDir::currentPath().toLocal8Bit().data());
}


#if !defined(TARGET_reader)
bool Project::
        save()
{
    if (project_filename_.empty()) {
        return saveAs();
    }

    QByteArray mainwindowState = _mainWindow->saveState();
    QByteArray mainwindowState2 = _mainWindow->saveState();
	BOOST_ASSERT( mainwindowState.size() == mainwindowState2.size());
	BOOST_ASSERT( 0 == memcmp(mainwindowState.data(), mainwindowState2.data(), mainwindowState2.size()));
    _mainWindow->restoreState( mainwindowState );

    {
        int microphoneCounter = 0;
        foreach(Signal::pChain c, layers.layers())
        {
            Adapters::MicrophoneRecorder* r = dynamic_cast<Adapters::MicrophoneRecorder*>( c->root_source().get() );
            if (r)
                r->setProjectName(project_filename_, ++microphoneCounter);
        }
    }

    try
    {
        TaskTimer tt("Saving project to '%s'", project_filename_.c_str());

#ifdef SAWE_USE_XML
        std::ofstream ofs(project_filename_.c_str(), ios_base::out | ios_base::trunc);
        boost::archive::xml_oarchive xml(ofs);
#else
        std::ofstream ofs(project_filename_.c_str(), ios_base::out | ios_base::trunc | ios_base::binary);
        boost::archive::binary_oarchive xml(ofs);
#endif
        Project* p = this;
        runSerialization(xml, p, project_filename_.c_str());

        p->is_modified_ = false;
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
#endif


pProject Project::
        openProject(std::string project_file)
{
#ifdef SAWE_USE_XML
        std::ifstream ifs(project_file.c_str(), ios_base::in);

        {
            string xmltest;
            xmltest.resize(5);

            ifs.read( &xmltest[0], 5 );
            if( !boost::iequals( xmltest, "<?xml") )
                throw std::invalid_argument("Project file '" + project_file + "' is not an xml file");

            for (int i=xmltest.size()-1; i>=0; i--)
                ifs.putback( xmltest[i] );
        }

        boost::archive::xml_iarchive xml(ifs);
#else
        std::ifstream ifs(project_file.c_str(), ios_base::in | ios_base::binary);
        boost::archive::binary_iarchive xml(ifs);
#endif

    Project* new_project = 0;
	runSerialization(xml, new_project, project_file.c_str());

    new_project->project_filename_ = project_file;
    new_project->updateWindowTitle();
    new_project->is_modified_ = false;

    pProject project( new_project );

    return project;
}

} // namespace Sawe
