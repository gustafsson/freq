#ifndef SAWEPROJECT_H
#define SAWEPROJECT_H

#include "tfr-chunksink.h"
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include "mainwindow.h"
#include "displaywidget.h"

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
    Project(Signal::pSource head_source);

    /**
      All sources can be reached from one head Source.

      For a source to be savable, there must exists a template specialization to Hdf5Output::add.

      For a source to be writable, there must exists a template specialization to Hdf5Input::read_exact.

      Selections are saved by saving the list of filters i the first FilterOperation.
      */
    Signal::pSource head_source;

    /**
      If the file can't be found it is regared as empty.

      If 'project_file_or_audio_file' is empty, a Qt Open File dialog will be opened.

      If the user presses cancel in the open file dialog null is returned.
      If the file can't be opened, null is returned.
     */
    static boost::shared_ptr<Project> open(std::string project_file_or_audio_file = "");

    /**
      If 'project_file_or_audio_file' is empty, a Qt Open File dialog will be opened.
     */
    static boost::shared_ptr<Project> createRecording(int record_device=-1);

    /**
      If 'project_file' is empty, a Qt Save File dialog will be opened.
     */
    void save(std::string project_file="");

    boost::shared_ptr<MainWindow> mainWindow();
    boost::shared_ptr<DisplayWidget> displayWidget();

private:
    boost::shared_ptr<MainWindow> _mainWindow;
    boost::shared_ptr<DisplayWidget> _displayWidget;

    void createMainWindow();

    static boost::shared_ptr<Project> openProject(std::string project_file);
    static boost::shared_ptr<Project> openAudio(std::string audio_file);
};
typedef boost::shared_ptr<Project> pProject;

} // namespace Sawe

#endif // SAWEPROJECT_H
