#include "checkupdates.h"
#include "sawe/reader.h"
#include "sawe/application.h"
#include "support/buildhttppost.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include <QSettings>
#include <QTimer>
#include <QMessageBox>
#include <QDesktopServices>
#include <QNetworkAccessManager>
#include <QNetworkReply>

#define STRINGIFY(x) #x
#define TOSTR(x) STRINGIFY(x)

#ifdef SONICAWE_UNAME
#define UNAME TOSTR(SONICAWE_UNAME)
#else
#define UNAME "Linux #78-Ubuntu SMP Tue Oct 11 16:11:24 UTC 2011 x86_64"
#endif

namespace Tools {

CheckUpdates::
        CheckUpdates(Ui::SaweMainWindow *parent) :
    QObject(parent),
    manualUpdate(false),
    targetUrl("http://feedback.sonicawe.com/checkforupdates.php"),
    checkUpdatesTag("CheckForUpdates")
{
    ::Ui::MainWindow* ui = parent->getItems();
    connect(ui->actionFind_updates, SIGNAL(triggered()), SLOT(checkForUpdates()));


    if ("not"==Sawe::Reader::reader_text().substr(0,3))
        return;

    QTimer::singleShot(500, this, SLOT(autoCheckForUpdates()));
}


CheckUpdates::
        ~CheckUpdates()
{

}


void CheckUpdates::
        checkForUpdates()
{
    QSettings().remove(checkUpdatesTag);
    manualUpdate = true;
    autoCheckForUpdates();
}


void CheckUpdates::
        autoCheckForUpdates()
{
    QSettings settings;
    if (!settings.contains(checkUpdatesTag))
    {
        int q = QMessageBox::question(dynamic_cast<QWidget*>(parent()),
                              "Check for updates",
                              "Do you want Sonic AWE to check for available updates automatically?",
                              QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        settings.setValue(checkUpdatesTag, q == QMessageBox::Yes);
    }

    if (!manualUpdate && !settings.value(checkUpdatesTag).toBool())
        return;


    Support::BuildHttpPost postdata;

    postdata.addKeyValue( "kind", manualUpdate?"manual":"auto" );
    postdata.addKeyValue( "uname", UNAME );
    postdata.addKeyValue( "name", Sawe::Reader::name.c_str() );
    postdata.addKeyValue( "value", QSettings().value("value").toString() );
    postdata.addKeyValue( "version", Sawe::Application::version_string().c_str() );
    postdata.addKeyValue( "title", Sawe::Application::title_string().c_str() );

    manager.reset( new QNetworkAccessManager(this) );
    connect(manager.data(), SIGNAL(finished(QNetworkReply*)),
            this, SLOT(replyFinished(QNetworkReply*)));
    postdata.send( manager.data(), targetUrl );
}


void CheckUpdates::
        replyFinished(QNetworkReply* reply)
{
    QString s = reply->readAll();
    TaskInfo("CheckUpdates reply error=%s\n%s",
             QNetworkReply::NoError == reply->error()?"no error":reply->errorString().toStdString().c_str(),
             s.replace("\\r\\n","\n").replace("\r","").toStdString().c_str());

    if (QNetworkReply::NoError != reply->error())
    {
        QMessageBox::warning(dynamic_cast<QWidget*>(parent()), "Could not check for updates", reply->errorString() + "\n" + s);
        QSettings().remove(checkUpdatesTag);
    }
    else if (s.contains("sorry", Qt::CaseInsensitive) ||
             s.contains("error", Qt::CaseInsensitive) ||
             s.contains("fatal", Qt::CaseInsensitive) ||
             s.contains("fail", Qt::CaseInsensitive) ||
             s.contains("html", Qt::CaseInsensitive))
    {
        QMessageBox::warning(dynamic_cast<QWidget*>(parent()), "Could not check for updates", s);
        QSettings().remove(checkUpdatesTag);
    }
    else
    {
        if (0 == s.compare("ok", Qt::CaseInsensitive))
        {
            if (manualUpdate)
            {
                QMessageBox::information(dynamic_cast<QWidget*>(parent()), "Sonic AWE updates", "Sonic AWE is up to date");
                manualUpdate = false;
            }
        }
        else
        {
            int i = s.indexOf("\nurl=",0, Qt::CaseInsensitive);
            QString message, url;
            if (i == -1)
            {
                message = s;
                url = "http://muchdifferent.com/?page=signals-download"
                      "&licence=" + QSettings().value("value").toString() +
                      "#download";
            }
            else
            {
                message = s.mid(0, i);
                url = s.mid(i+5);
            }

            int q = QMessageBox::question(dynamic_cast<QWidget*>(parent()), "Sonic AWE updates", message, QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
            if (q == QMessageBox::Yes)
                QDesktopServices::openUrl(url);
        }
    }
}

} // namespace Tools
