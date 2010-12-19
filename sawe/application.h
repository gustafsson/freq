#ifndef SAWEAPPLICATION_H
#define SAWEAPPLICATION_H

#include <QtGui/QApplication>
#include <QGLWidget>
#include "sawe/project.h"
#include <set>

namespace Sawe {

class Application: public QApplication
{
    Q_OBJECT

public:
    Application( int& argc, char **argv);
    ~Application();

    static QGLWidget*   shared_glwidget();
    static std::string  version_string() { return global_ptr()->_version_string; }
    static void         display_fatal_exception();
    static void         display_fatal_exception(const std::exception& );
    static Application* global_ptr();

    virtual bool notify(QObject * receiver, QEvent * e);

    void				openadd_project( pProject p );
    int					default_record_device;

public slots:
    pProject slotNew_recording( int record_device = -1 );
    pProject slotOpen_file( std::string project_file_or_audio_file="" );
    void slotClosed_window( QWidget* );

private:
    QGLWidget* shared_glwidget_;
    static Application* _app;
    static std::string _fatal_error;
    std::string _version_string;
    std::set<pProject> _projects;
};

} // namespace Sawe

#endif // SAWEAPPLICATION_H
