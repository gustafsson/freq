#ifndef SAWEPROJECT_H
#define SAWEPROJECT_H

#include "signal/worker.h"

#include <boost/shared_ptr.hpp>
#include <QGLWidget>
#include <QMainWindow>
#include <QScopedPointer>

#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/shared_ptr.hpp>

namespace Sawe {
    class Project;
}
#include "tools/toolfactory.h"
class MainWindow;

namespace Sawe {

/**
  Saves, restores and creates Sonic AWE project files. '*.sonicawe'

  Shows Qt Message boxes if files can't be found or access is denied and returns null.

  MainWindow and displaywidget are created and shown by their Sawe::Project.
*/
class Project
{
public:
    /**
      A project currently is entirely defined by its head source.
      */
    Project(Signal::pOperation head_source);
    ~Project();

    /**
      All sources can be reached from one head Source: worker->source().

      For a source to be writable, there must exists a template specialization to Hdf5Output::add.

      For a source to be readable, there must exists a template specialization to Hdf5Input::read_exact.

      Selections are saved by saving the list of filters i the first CwtFilter.
      */
    Signal::pWorker worker;

    Signal::pOperation head_source() { return worker->source(); }
    void head_source(Signal::pOperation s) { worker->source(s); }


    /**
        'head_source' can be taken as model, 'tools' as controller and
        '_mainWindow' as view.
      */
    Tools::ToolFactory tools;

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
      Creates a new recording on 'record_device'.
     */
    static boost::shared_ptr<Project> createRecording(int record_device=-1);


    /**
      If 'project_file' is empty, a Qt Save File dialog will be opened.
     */
    void save(std::string project_file="");

    QMainWindow* mainWindow();

private:
    Project() : tools(this) {} // used by deserialization
    void createMainWindow();

    QScopedPointer<QMainWindow> _mainWindow; // MainWindow owns all other widgets

    static boost::shared_ptr<Project> openProject(std::string project_file);
    static boost::shared_ptr<Project> openAudio(std::string audio_file);


    friend class boost::serialization::access;
    template<class archive> void save(archive& ar, const unsigned int /*version*/) const {
        Signal::pOperation head_source = worker->source();

        using boost::serialization::make_nvp;
        ar & make_nvp("Headsource", head_source);
    }
    template<class archive> void load(archive& ar, const unsigned int /*version*/) {
        Signal::pOperation head_source;

        using boost::serialization::make_nvp;
        ar & make_nvp("Headsource", head_source);

        worker->source(head_source);
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()
};
typedef boost::shared_ptr<Project> pProject;

// BOOST_CLASS_VERSION(Project, 1) TODO use

} // namespace Sawe

#endif // SAWEPROJECT_H
