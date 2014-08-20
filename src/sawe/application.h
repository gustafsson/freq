#ifndef SAWEAPPLICATION_H
#define SAWEAPPLICATION_H

// Sawe namespace
#include "project.h"

// Qt
#include <QtWidgets> // QApplication

// std
#include <set>

class QGLWidget;
class RedirectStdout;

namespace Sawe {

class SaweDll Application: public QApplication
{
    Q_OBJECT

public:
    Application( int& argc, char **argv, bool prevent_log_system_and_execute_args = false);
    ~Application();

    static void         logSystemInfo(int& argc, char **argv);
    static QString      log_directory();
    static QGLWidget*   shared_glwidget();
    static void         display_fatal_exception();
    static void         display_fatal_exception(const std::exception& );
    static Application* global_ptr();

    static void         check_license();

    virtual bool notify(QObject * receiver, QEvent * e);

    void				openadd_project( pProject p );
    int					default_record_device;
    bool                has_other_projects_than( Project* p4 );
    std::set<boost::weak_ptr<Sawe::Project>> projects();

    void execute_command_line_options();

    void clearCaches();

    boost::shared_ptr<RedirectStdout> rs;

signals:
    void clearCachesSignal();
    void licenseChecked();

public slots:
    pProject slotNew_recording( );
    pProject slotOpen_file( std::string project_file_or_audio_file="" );
    void slotClosed_window( QWidget* );

private:
    void apply_command_line_options( pProject p );
    static void show_fatal_exception( const std::string& str );

    QPointer<QGLWidget> shared_glwidget_;
    static std::string _fatal_error;
    std::set<pProject> _projects;
};

} // namespace Sawe

#endif // SAWEAPPLICATION_H
