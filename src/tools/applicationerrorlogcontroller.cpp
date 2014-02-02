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
QString has_unreported_error_key("has_unreported_error");
QString had_previous_crash_key("had_previous_crash_key");

ApplicationErrorLogController::
        ApplicationErrorLogController()
    :
      send_feedback_(new Support::SendFeedback(this))
{    
    qRegisterMetaType<boost::exception_ptr>("boost::exception_ptr");

    moveToThread (&thread_);
    thread_.start ();


    connect (send_feedback_, SIGNAL(finished(QNetworkReply*)), SLOT(finishedSending(QNetworkReply*)), Qt::QueuedConnection);
    connect (this, SIGNAL(got_exception(boost::exception_ptr)), SLOT(log(boost::exception_ptr)), Qt::QueuedConnection);
    connect (QApplication::instance (), SIGNAL(aboutToQuit()), this, SLOT(finishedOk()), Qt::BlockingQueuedConnection);

    bool had_previous_crash = QSettings().value (had_previous_crash_key, false).toBool ();

    try
      {
        EXCEPTION_ASSERT(!had_previous_crash);
      }
    catch ( const boost::exception& x)
      {
        emit got_exception (boost::current_exception());
      }

    QSettings().setValue (has_unreported_error_key, had_previous_crash);
    QSettings().setValue (had_previous_crash_key, true); // Clear on clean exit
}


void ApplicationErrorLogController::
        finishedOk()
{
    QSettings().remove (had_previous_crash_key);
    QSettings().remove (has_unreported_error_key);
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
    emit instance()->showToolbar (true);
    QSettings().setValue (has_unreported_error_key, true);
}


void ApplicationErrorLogController::
        registerMainWindow(QMainWindow* mainwindow)
{
    QIcon icon = QCommonStyle().standardIcon(QStyle::SP_MessageBoxWarning);
    QString name = QApplication::instance ()->applicationName ();
    QToolBar* bar = new QToolBar(mainwindow);
    OpenFeedbackDialog* open_feedback_dialog = new OpenFeedbackDialog(mainwindow, bar);
    bar->setObjectName ("ApplicationErrorLogControllerBar");
    mainwindow->addToolBar(Qt::TopToolBarArea, bar);
    bar->addAction(
                icon,
                "An error has been reported by " + name + ". Click to file a bug report",
                open_feedback_dialog, SLOT(open()));

    bool visible = QSettings().value (has_unreported_error_key, false).toBool ();

    //Call "bar->setVisible (visible)" after finishing loading the mainwindow.
    QMetaObject::invokeMethod (bar, "setVisible", Qt::QueuedConnection, Q_ARG(bool, visible));

    // Hide and show the toolbar in all dialogs from the correct thread
    // regardless of where the showToolbar signal was emitted.
    connect (open_feedback_dialog, SIGNAL(dialogOpened()), instance(), SIGNAL(showToolbar()));
    connect (instance(), SIGNAL(showToolbar(bool)), open_feedback_dialog, SLOT(showToolbar(bool)));
}


void ApplicationErrorLogController::
        log(boost::exception_ptr e)
{
    if (!e)
        return;

    emit showToolbar (true);

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
        OpenFeedbackDialog(QWidget *parent, QToolBar* bar)
    :
      QObject(parent),
      bar_(bar),
      send_feedback_dialog_(new SendFeedbackDialog(0))
{
    send_feedback_dialog_->setParent (parent);
    send_feedback_dialog_->hide ();
}


void OpenFeedbackDialog::
        showToolbar (bool v)
{
    if (bar_)
        bar_->setVisible(v);
}


void OpenFeedbackDialog::
        open()
{
    emit dialogOpened();

    QMetaObject::invokeMethod (send_feedback_dialog_, "open");

    QSettings().remove (has_unreported_error_key);
}


} // namespace Tools



namespace Tools {

void ApplicationErrorLogController::
        test()
{
    // It should collect information about crashes to send anonymous feedback.
    if (false) {
        std::string name = "ApplicationErrorLogController";
        int argc = 1;
        char * argv = &name[0];
        QApplication a(argc,&argv);

        try {
            EXCEPTION_ASSERT(false);
        } catch (...) {
            ApplicationErrorLogController::registerException (boost::current_exception());
        }
    }
}

} // namespace Tools
