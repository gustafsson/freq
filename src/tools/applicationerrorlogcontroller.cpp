#include "applicationerrorlogcontroller.h"

#include "support/sendfeedback.h"
#include "sendfeedbackdialog.h"

#include "TaskTimer.h"
#include "demangle.h"
#include "exceptionassert.h"
#include "expectexception.h"

#include <QMutexLocker>
#include <QStyle>
#include <QApplication>
#include <QSettings>
#include <QTimer>
#include <QCommonStyle>
#include <QMainWindow>
#include <QAction>

namespace Tools {

QMutex mutex;
QString has_unreported_error_key("has unreported error");

ApplicationErrorLogController::
        ApplicationErrorLogController()
    :
      send_feedback_(new Support::SendFeedback(this)),
      open_feedback_dialog_(new OpenFeedbackDialog)
{    
    qRegisterMetaType<boost::exception_ptr>("boost::exception_ptr");

    moveToThread (&thread_);
    thread_.start ();

    connect(send_feedback_, SIGNAL(finished(QNetworkReply*)), SLOT(finishedSending(QNetworkReply*)), Qt::QueuedConnection);
    connect(this, SIGNAL(got_exception(boost::exception_ptr)), SLOT(log(boost::exception_ptr)));
}


ApplicationErrorLogController* ApplicationErrorLogController::
        instance()
{
    QMutexLocker l(&mutex);

    static ApplicationErrorLogController* p = 0;
    if (!p)
        p = new ApplicationErrorLogController();

    return p;
}


void ApplicationErrorLogController::
        registerException(boost::exception_ptr e)
{
    if (!e)
        return;

    try {
        rethrow_exception(e);
    } catch (const std::exception& x) {
        TaskInfo(boost::format("ApplicationErrorLogController::registerException: %s") % vartype(x));
    }

    // Will be executed in instance()->thread_
    emit instance()->got_exception (e);

    QSettings().setValue (has_unreported_error_key, true);
    // disable has_unreported_error_key
    QSettings().remove (has_unreported_error_key);

    foreach(QPointer<QToolBar> a, instance()->toolbars_) {
        if (a) {
            a->setVisible (true);
        }
    }
}


void ApplicationErrorLogController::
        registerMainWindow(QMainWindow* mainwindow)
{
    QIcon icon = QCommonStyle().standardIcon(QStyle::SP_MessageBoxWarning);
    QString name = QApplication::instance ()->applicationName ();
    QToolBar* bar = new QToolBar(mainwindow);
    bar->setObjectName ("ApplicationErrorLogControllerBar");
    mainwindow->addToolBar(Qt::TopToolBarArea, bar);
    bar->addAction(
                icon,
                "An error has been reported by " + name + ". Click to file a bug report",
                instance()->open_feedback_dialog_, SLOT(open()));

    bool visible = QSettings().value (has_unreported_error_key).toBool ();
    bar->setVisible (visible);

    instance()->toolbars_.push_back (bar);
}


void ApplicationErrorLogController::
        log(boost::exception_ptr e)
{
    if (!e)
        return;

    try
      {
        rethrow_exception(e);
      }
    catch ( const boost::exception& x)
      {
        QMutexLocker l(&mutex);

        TaskTimer ti(boost::format("Caught exception '%s'")
                 % vartype(x));

        // This might be slow. For instance; 'to_string(Backtrace::info)' takes 1 second to execute.
        std::string str = boost::diagnostic_information(x);

        std::cout.flush ();
        std::cerr.flush ();
        std::cerr << std::endl << std::endl
             << "======================" << std::endl
             << str << std::endl
             << "======================" << std::endl << std::endl;
        std::cerr.flush ();

        char const* condition = 0;
        if( char const * const * mi = boost::get_error_info<ExceptionAssert::ExceptionAssert_condition>(x) )
            condition = *mi;

        std::string message;
        if( std::string const * mi = boost::get_error_info<ExceptionAssert::ExceptionAssert_message>(x) )
            message = *mi;

        TaskTimer ti2("Sending feedback");

        // Place message before details
        QString msg;
        if (condition)
          {
            msg += condition;
            msg += "\n";
          }

        if (!message.empty ())
            msg += QString::fromStdString (message + "\n\n");
        msg += QString::fromStdString (str);

        QString omittedMessage = send_feedback_->sendLogFiles ("errorlog", msg, "");
        if (!omittedMessage.isEmpty ())
            TaskInfo(boost::format("omittedMessage = %s") % omittedMessage.toStdString ());
      }
}


void ApplicationErrorLogController::
        finishedSending(QNetworkReply* reply)
{
    TaskInfo ti("SendFeedback reply");

    QString s = reply->readAll();

    TaskInfo(boost::format("%s")
             % s.replace("\\r\\n","\n").replace("\r","").toStdString());

    if (QNetworkReply::NoError != reply->error())
    {
        TaskInfo(boost::format("SendFeedback error=%s (code %d)")
                 % (QNetworkReply::NoError == reply->error()?"no error":reply->errorString().toStdString())
                 % (int)reply->error());
    }
}


OpenFeedbackDialog::
        OpenFeedbackDialog()
    :
      send_feedback_dialog_(new SendFeedbackDialog(0))
{
}


void OpenFeedbackDialog::
        open()
{
    ApplicationErrorLogController* c = ApplicationErrorLogController::instance ();

    foreach(QPointer<QToolBar> a, c->toolbars_)
    {
        if (a)
            a->setVisible (false);
    }

    QTimer::singleShot (0, send_feedback_dialog_, SLOT(open()));

    QSettings().remove (has_unreported_error_key);
}


} // namespace Tools



namespace Tools {

void ApplicationErrorLogController::
        test()
{
    // It should collect information about crashes to send anonymous feedback.
    if (false) {
        int argc = 0;
        char* argv = 0;
        QApplication a(argc,&argv);

        try {
            EXCEPTION_ASSERT(false);
        } catch (...) {
            ApplicationErrorLogController::registerException (boost::current_exception());
        }
    }
}

} // namespace Tools
