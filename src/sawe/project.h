#ifndef SAWEPROJECT_H
#define SAWEPROJECT_H

#include "toolmodel.h"
#include "toolmainloop.h"
#include "tools/toolfactory.h"
#include "tools/commands/commandinvoker.h"
#include "signal/processing/chain.h"

// boost
#include <boost/scoped_ptr.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/vector.hpp> 
#include <boost/serialization/binary_object.hpp> 
#include <boost/serialization/version.hpp>

// Qt
#include <QMainWindow>
#include <QScopedPointer>

namespace Tools {
    namespace Commands {
        class CommandInvoker;
    }
}

namespace Ui {
    class SaweMainWindow;
}

namespace Sawe {

/**
  Saves, restores and creates Sonic AWE project files. '*.sonicawe'

  Shows Qt Message boxes if files can't be found or access is denied and
  returns null.

  SaweMainWindow is created, shown and owned by its Sawe::Project.
  SaweMainWindow owns a centralwidget which owns the current tool which is
  parent to RenderView. RenderView is not however considered to be owned by the
  current tool (even though it is a child in the Qt object tree). Rather
  RenderController controls the lifetime of RenderView and ToolSelector is used
  to move RenderView around in the Qt object tree.
*/
class SaweDll Project
{
public:
    Project(std::string project_title);
    ~Project();


    /**
      Appends 's' to head. If there is a current selection this only applies 's' to that selection.
      */
    void appendOperation(Signal::pOperation s);
    void appendOperation(Signal::OperationDesc::Ptr s);


    /**
      Roughly speaking 'layers' and 'head' can be taken as model, 'tools' as
      controller and 'mainWindow' as view.
      */
    Tools::ToolRepo& toolRepo();
    Tools::ToolFactory& tools();


    /**
      It is an error to call tools() or toolRepo() during initialization when
      areToolsInitialized() returns false.
      */
    bool areToolsInitialized();

    /**

      */
    //void userinput_update( bool request_high_fps = true );
    //void target(Signal::pTarget target, bool request_high_fps = true, Signal::IntervalType center = 0 );

    static void addRecentFile(std::string filename);

    /**
      Opens a Sonic AWE project or imports an audio file. If
      'project_file_or_audio_file' is empty, a Qt Open File dialog will be
      opened.

      Returns null if one of the following is true:
       - the user presses cancel in the open file dialog
       - the file can't be opened
     */
    static boost::shared_ptr<Project> open(std::string project_file_or_audio_file = "");


    /**
      Creates a new recording on QSettings().value("inputdevice").
     */
    static boost::shared_ptr<Project> createRecording();


    /**
      Returns true if the project has been saved since it was opened.
      */
    bool isModified();


    /**
      Sets the modified flag to true. Temporary, will be removed by list of
      actions instead. To support undo history.
      */
    void setModified( bool is_modified=true );


    Tools::Commands::CommandInvoker* commandInvoker();

#if !defined(TARGET_reader)
    /**
      If 'project_filename_' is empty, calls saveAs.

      @returns true if the project was saved.
     */
    bool save();


    /**
      Opens a Qt Save File dialog and renames 'project_filename_'.

      @returns true if the project was saved.
     */
    bool saveAs();
    bool saveAs(std::string newprojectfilename);
#endif

    /**
      Obtain the main window for this project.
      */
    Ui::SaweMainWindow* mainWindow();
    QWidget* mainWindowWidget();


    /**
      Project file name.
      */
    std::string project_title();


    /**
      Project file name.
      */
    std::string project_filename();


    /**
      The default settings are stored when the project is created and can
      be restored layer through this method.
      */
    void resetLayout();
    void resetView();


    /**
      Returns true if 'this' was deserialized from a project file.
      */
    bool isSaweProject();


    /**
     * @brief processing_chain return the signal processing chain used to add
     * targets and operations.
     * @return
     */
    Signal::Processing::Chain::Ptr processing_chain() { return processing_chain_; }
    Signal::Processing::TargetMarker::Ptr default_target();
    Signal::OperationDesc::Extent extent();
    float length();

private:
    Project(); // used by deserialization
    void createMainWindow();
    void updateWindowTitle();

    QByteArray getGuiState() const;
    void setGuiState(QByteArray guiState);
    QByteArray defaultGuiState;

    bool is_modified_, is_sawe_project_;
    boost::scoped_ptr<Tools::Commands::CommandInvoker> command_invoker_;

    Signal::Processing::Chain::Ptr processing_chain_;
    std::string project_filename_, project_title_;

    boost::scoped_ptr<Tools::ToolRepo> _tools;
    // MainWindow owns all other widgets together with the ToolRepo
    QPointer<QMainWindow> _mainWindow;

    static boost::shared_ptr<Project> openProject(std::string project_file);
#if !defined(TARGET_reader)
    static boost::shared_ptr<Project> openAudio(std::string audio_file);
#endif
#if !defined(TARGET_reader) && !defined(TARGET_hast)
    static boost::shared_ptr<Project> openCsvTimeseries(std::string audio_file);
#endif

    friend class boost::serialization::access;
    template<class Archive> void save(Archive& ar, const unsigned int /*version*/) const {
        TaskInfo ti("Project::serialize");

        //Use Signal::Processing namespace
        QByteArray guiState = getGuiState();
        save_bytearray( ar, guiState );

        Tools::ToolFactory& tool_repo = *dynamic_cast<Tools::ToolFactory*>(_tools.get());
        ar & BOOST_SERIALIZATION_NVP(tool_repo);
    }
    template<class Archive> void load(Archive& ar, const unsigned int version) {
        TaskInfo ti("Project::deserialize");

        //Use Signal::Processing namespace
		createMainWindow();

        QByteArray guiState;
        load_bytearray( ar, guiState );
        if (0 < version)
            setGuiState( guiState );
        else
            _mainWindow->restoreState( guiState, version);

        // createMainWindow has already created all tools
        // this deserialization sets their settings
        Tools::ToolFactory& tool_repo = *dynamic_cast<Tools::ToolFactory*>(_tools.get());
        ar & BOOST_SERIALIZATION_NVP(tool_repo);
    }

    template<class Archive> static void save_bytearray(Archive& ar, QByteArray& c)
    {
        int DataSize = c.size();
        ar & BOOST_SERIALIZATION_NVP(DataSize);

        boost::serialization::binary_object Data( c.data(), DataSize );
        ar & BOOST_SERIALIZATION_NVP(Data);
    }
    template<class Archive> static void load_bytearray(Archive& ar, QByteArray& c)
    {
        int DataSize = 0;
        ar & BOOST_SERIALIZATION_NVP(DataSize);
        c.resize(DataSize);

        boost::serialization::binary_object Data( c.data(), DataSize );
        ar & BOOST_SERIALIZATION_NVP(Data);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
typedef boost::shared_ptr<Project> pProject;

} // namespace Sawe

BOOST_CLASS_VERSION(Sawe::Project, 1)

#endif // SAWEPROJECT_H
