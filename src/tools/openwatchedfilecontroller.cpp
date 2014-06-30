#include "openwatchedfilecontroller.h"
#include "signal/operationwrapper.h"

#include <QFileSystemWatcher>
#include <QTimer>

namespace Tools {

class OpenfileWatcher: public FileChangedBase
{
public:
    OpenfileWatcher(QPointer<OpenfileController> openfilecontroller, QString path);

    void setWrapper(shared_state<Signal::OperationDesc>::weak_ptr wrapper);
    void setWrappedOperationDesc(Signal::OperationDesc::ptr);
    Signal::OperationDesc::ptr getWrappedOperationDesc();

private:
    void fileChanged (const QString & path) override;
    void delayedFileChanged () override;

    shared_state<Signal::OperationDesc>::weak_ptr wrapper_;
    Signal::OperationDesc::ptr last_loaded_operation_;
    QPointer<OpenfileController> openfilecontroller_;
    QFileSystemWatcher watcher_;
    shared_state<Signal::OperationDesc>::weak_ptr operation_;
    QString path_;
    QTimer timer_;
};


class OpenfileWatcherOperationDesc: public Signal::OperationDescWrapper {
public:
    QSharedPointer<OpenfileWatcher> openfile_watcher;
};


OpenfileWatcher::
        OpenfileWatcher(QPointer<OpenfileController> openfilecontroller, QString path)
    :
      openfilecontroller_(openfilecontroller),
      path_(path)
{
    watcher_.addPath (path);
    timer_.setSingleShot(true);

    connect(&watcher_, SIGNAL(fileChanged(QString)), SLOT(fileChanged(QString)));
    connect(&timer_, SIGNAL(timeout()), SLOT(delayedFileChanged()));

    delayedFileChanged ();
}


void OpenfileWatcher::
        setWrapper(shared_state<Signal::OperationDesc>::weak_ptr wrapper)
{
    wrapper_ = wrapper;

    setWrappedOperationDesc (last_loaded_operation_);
}


void OpenfileWatcher::
        setWrappedOperationDesc(Signal::OperationDesc::ptr o)
{
    if (Signal::OperationDesc::ptr wrapper = wrapper_.lock ())
    {
        auto w = wrapper.write ();
        Signal::OperationDescWrapper* odw = dynamic_cast<Signal::OperationDescWrapper*>(w.get ());
        if (odw) {
            odw->setWrappedOperationDesc (o);
            w.unlock ();

            Signal::Processing::IInvalidator::ptr i = wrapper.raw ()->getInvalidator ();
            if (i)
                i->deprecateCache (Signal::Interval::Interval_ALL);
        }
    }
}


Signal::OperationDesc::ptr OpenfileWatcher::
        getWrappedOperationDesc()
{
    return last_loaded_operation_;
}


void OpenfileWatcher::
        fileChanged ( const QString & path)
{
//    TaskInfo ti(boost::format("File changed: %s") % path.toStdString ());

    timer_.start(250);
}


void OpenfileWatcher::
        delayedFileChanged ()
{
    TaskInfo ti(boost::format("Delayed file changed: %s") % path_.toStdString ());

    Signal::OperationDesc::ptr newop = openfilecontroller_->reopen(path_, last_loaded_operation_);
    if (!newop) {
        TaskInfo("Could not open file, ignoring reload");
        return;
    }

    last_loaded_operation_ = newop;

    setWrappedOperationDesc (last_loaded_operation_);
}


OpenWatchedFileController::
        OpenWatchedFileController(QPointer<OpenfileController> openfilecontroller)
    :
    QObject(&*openfilecontroller),
    openfilecontroller_(openfilecontroller)
{
}


Signal::OperationDesc::ptr OpenWatchedFileController::
        openWatched(QString url)
{
    QSharedPointer<OpenfileWatcher> w {new OpenfileWatcher(openfilecontroller_, url)};

    // The file must exist to begin with
    if (w->getWrappedOperationDesc ())
    {
        OpenfileWatcherOperationDesc* d;
        Signal::OperationDesc::ptr o {d = new OpenfileWatcherOperationDesc};

        d->openfile_watcher = w; // ownership
        w->setWrapper (o);       // to lock while calling setWrappedOperationDesc

        return o;
    }
    // Failed to open url
    return Signal::OperationDesc::ptr();
}

} // namespace Tools

#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include <QApplication>
#include <QThread>

namespace Tools {

class DummyFileWatchedOperationDesc : public Signal::OperationDesc {
public:
    DummyFileWatchedOperationDesc(QString which):which(which) {}
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* ) const { return I; }
    virtual Signal::Interval affectedInterval( const Signal::Interval& I ) const { return I; }
    virtual OperationDesc::ptr copy() const { return OperationDesc::ptr(); }
    virtual Signal::Operation::ptr createOperation(Signal::ComputingEngine*) const { return Signal::Operation::ptr(); }
    virtual QString toString() const { return which; }

private:
    QString which;
};

class DummyFileWatchedOpener : public OpenfileController::OpenfileInterface {
public:
    Patterns patterns() { return Patterns(); }

    Signal::OperationDesc::ptr reopen(QString url, Signal::OperationDesc::ptr) {
        QFile file(url);
        if (!file.exists ())
            return Signal::OperationDesc::ptr();

        file.open (QIODevice::ReadOnly);
        QString str(file.readAll ());
        return Signal::OperationDesc::ptr(new DummyFileWatchedOperationDesc(str));
    }
};

void OpenWatchedFileController::
        test()
{
    // It should reopen a file when it is modified.
    {
        int argc=0;
        QApplication application(argc,0);

        QDir tmplocation = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
        QString filename = tmplocation.filePath("dummywav.wav");

        QFile file(filename);
        file.open (QIODevice::WriteOnly);
        file.write ("foobar");
        file.close ();

        EXCEPTION_ASSERT(file.exists ());

        QPointer<OpenfileController> ofc(new OpenfileController);
        QPointer<OpenfileController::OpenfileInterface> ofi(new DummyFileWatchedOpener);
        ofc->registerOpener (ofi);

        OpenWatchedFileController owfc(ofc);
        Signal::OperationDesc::ptr od = owfc.openWatched (filename);

        EXCEPTION_ASSERT(od);

        application.processEvents ();
        EXCEPTION_ASSERT_EQUALS(od.read ()->toString().toStdString(), "foobar");

        QThread::msleep(1);
        file.open (QIODevice::WriteOnly);
        file.write ("baz");
        file.close ();

        EXCEPTION_ASSERT_EQUALS(od.read ()->toString().toStdString(), "foobar");
        application.processEvents ();
        QThread::msleep(300);
        application.processEvents ();
        EXCEPTION_ASSERT_EQUALS(od.read ()->toString().toStdString(), "baz");

        file.remove ();
    }
}

} // namespace Tools
