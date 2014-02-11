#include "openwatchedfilecontroller.h"
#include "signal/operationwrapper.h"

#include <QFileSystemWatcher>

namespace Tools {


class OpenfileWatcher: public FileChangedBase, public Signal::OperationDescWrapper
{
public:
    OpenfileWatcher(QPointer<OpenfileController> openfilecontroller, QString path);

private:
    void fileChanged (const QString & path);

    QPointer<OpenfileController> openfilecontroller_;
    QFileSystemWatcher watcher_;
};

OpenfileWatcher::
        OpenfileWatcher(QPointer<OpenfileController> openfilecontroller, QString path)
    :
      OperationDescWrapper(Signal::OperationDesc::Ptr()),
      openfilecontroller_(openfilecontroller)
{
    watcher_.addPath (path);
    connect(&watcher_, SIGNAL(fileChanged(QString)), SLOT(fileChanged(QString)));

    fileChanged (path);
}


void OpenfileWatcher::
        fileChanged ( const QString & path)
{
    TaskInfo ti(boost::format("File changed: %s") % path.toStdString ());

    Signal::OperationDesc::Ptr file = openfilecontroller_->open(path);

    setWrappedOperationDesc (file);
}


OpenWatchedFileController::
        OpenWatchedFileController(QPointer<OpenfileController> openfilecontroller)
    :
    QObject(&*openfilecontroller),
    openfilecontroller_(openfilecontroller)
{
}


Signal::OperationDesc::Ptr OpenWatchedFileController::
        openWatched(QString url)
{
    OpenfileWatcher* w;
    Signal::OperationDesc::Ptr o(w = new OpenfileWatcher(openfilecontroller_, url));

    // The file must exist to begin with
    if (w->getWrappedOperationDesc ())
        return o;

    // Failed to open url
    return Signal::OperationDesc::Ptr();
}

} // namespace Tools

#include <QFile>
#include <QDir>
#include <QStandardPaths>
#include <QApplication>

namespace Tools {

class DummyFileWatchedOperationDesc : public Signal::OperationDesc {
public:
    DummyFileWatchedOperationDesc(QString which):which(which) {}
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* ) const { return I; }
    virtual Signal::Interval affectedInterval( const Signal::Interval& I ) const { return I; }
    virtual OperationDesc::Ptr copy() const { return OperationDesc::Ptr(); }
    virtual Signal::Operation::Ptr createOperation(Signal::ComputingEngine*) const { return Signal::Operation::Ptr(); }
    virtual QString toString() const { return which; }

private:
    QString which;
};

class DummyFileWatchedOpener : public OpenfileController::OpenfileInterface {
public:
    Patterns patterns() { return Patterns(); }

    Signal::OperationDesc::Ptr open(QString url) {
        QFile file(url);
        if (!file.exists ())
            return Signal::OperationDesc::Ptr();

        file.open (QIODevice::ReadOnly);
        QString str(file.readAll ());
        return Signal::OperationDesc::Ptr(new DummyFileWatchedOperationDesc(str));
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
        Signal::OperationDesc::Ptr od = owfc.openWatched (filename);

        EXCEPTION_ASSERT(od);

        application.processEvents ();
        EXCEPTION_ASSERT_EQUALS(read1(od)->toString().toStdString(), "foobar");

        file.open (QIODevice::WriteOnly);
        file.write ("baz");
        file.close ();

        EXCEPTION_ASSERT_EQUALS(read1(od)->toString().toStdString(), "foobar");
        application.processEvents ();
        EXCEPTION_ASSERT_EQUALS(read1(od)->toString().toStdString(), "baz");

        file.remove ();
    }
}

} // namespace Tools
