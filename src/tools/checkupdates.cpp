#include "checkupdates.h"
#include "sawe/reader.h"
#include "sawe/application.h"
#include "sawe/configuration.h"
#include "support/buildhttppost.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "computationkernel.h"
#include "dropnotifyform.h"

#include <QSettings>
#include <QTimer>
#include <QMessageBox>
#include <QDesktopServices>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QHostInfo>

namespace Tools {

CheckUpdates::
        CheckUpdates(::Ui::SaweMainWindow *parent) :
    QObject(parent),
    manualUpdate(false),
    targetUrl("http://feedback.sonicawe.com/checkforupdates.php"),
    checkUpdatesTag("CheckForUpdates"),
    pending_request(0)
{
    ::Ui::MainWindow* ui = parent->getItems();
    connect(ui->actionFind_updates, SIGNAL(triggered()), SLOT(checkForUpdates()));

    // Only check for updates automatically once per process
    static bool hasChecked = false;
    if (hasChecked)
        return;
    hasChecked = true;

    if ("not"==Sawe::Reader::reader_text().substr(0,3))
    {
        // wait for reader to finish
        connect( Sawe::Application::global_ptr(), SIGNAL(licenseChecked()), SLOT(autoCheckForUpdatesSoon()) );
    }
    else
    {
        autoCheckForUpdatesSoon();
    }
}


CheckUpdates::
        ~CheckUpdates()
{
    if (pending_request)
        pending_request->abort ();
}


void CheckUpdates::
        checkForUpdates()
{
    QSettings().remove(checkUpdatesTag);
    manualUpdate = true;
    autoCheckForUpdates();
}


void CheckUpdates::
        autoCheckForUpdatesSoon()
{
    QTimer::singleShot(500, this, SLOT(autoCheckForUpdates()));
}


void CheckUpdates::
        autoCheckForUpdates()
{
    if ("not"!=Sawe::Reader::reader_text().substr(0,3))
        disconnect(this, SLOT(autoCheckForUpdatesSoon()));

    QSettings settings;
    if (!settings.contains(checkUpdatesTag) && !manualUpdate)
        settings.setValue(checkUpdatesTag, true);

    if (!settings.contains(checkUpdatesTag))
    {
        // treat this as a manual update and notify user if Sonic AWE is
        // up-to-date the first time the program is started. And to notify
        // MuchDifferent about the installation.
        //manualUpdate = true;

        int q = QMessageBox::question(dynamic_cast<QWidget*>(parent()),
                              "Check for updates",
                              "Do you want Sonic AWE to check for available updates automatically?",
                              QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
        settings.setValue(checkUpdatesTag, q == QMessageBox::Yes);
    }

    bool checkAuto = settings.value(checkUpdatesTag).toBool();
    if (!manualUpdate && !checkAuto)
        return;

    TaskInfo ti("Checking for updates %s", manualUpdate?"manually":"automatically");
    settings.setValue(checkUpdatesTag, false); // disable if this causes the application to crash

    Support::BuildHttpPost postdata;

    postdata.addKeyValue( "kind", manualUpdate?checkAuto?"manual-auto":"manual":"auto" );
    postdata.addKeyValue( "uname", Sawe::Configuration::operatingSystemName().c_str() );
    postdata.addKeyValue( "device", Sawe::Configuration::computationDeviceName().c_str() );
    postdata.addKeyValue( "revision", Sawe::Configuration::revision().c_str() );
    postdata.addKeyValue( "name", Sawe::Reader::name.c_str() );
    postdata.addKeyValue( "text", Sawe::Reader::reader_text().c_str() );
    postdata.addKeyValue( "hostname", QHostInfo::localHostName() );
    postdata.addKeyValue( "domainname", QHostInfo::localDomainName() );
    postdata.addKeyValue( "value", QSettings().value("value").toString() );
    postdata.addKeyValue( "version", Sawe::Configuration::version_string().c_str() );
    postdata.addKeyValue( "title", Sawe::Configuration::title_string().c_str() );

    manager.reset( new QNetworkAccessManager(this) );
    connect(manager.data(), SIGNAL(finished(QNetworkReply*)),
            this, SLOT(replyFinished(QNetworkReply*)));
    pending_request = postdata.send( manager.data(), targetUrl );

    settings.setValue(checkUpdatesTag, checkAuto); // restore value before attempting update
}


void CheckUpdates::
        replyFinished(QNetworkReply* reply)
{
    pending_request = 0;
    QString s = reply->readAll();
    TaskInfo ti("CheckUpdates reply");
    TaskInfo("%s", s.replace("\\r\\n","\n").replace("\r","").toStdString().c_str());
    if (QNetworkReply::NoError != reply->error())
    {
        TaskInfo("CheckUpdates error=%s (code %d)",
                 QNetworkReply::NoError == reply->error()?"no error":reply->errorString().toStdString().c_str(),
                 (int)reply->error());
    }

    if ("not"==Sawe::Reader::reader_text().substr(0,3))
        return;
    else
        disconnect(this, SLOT(autoCheckForUpdatesSoon()));

    if (QNetworkReply::NoError != reply->error())
    {
        if (QNetworkReply::HostNotFoundError != reply->error())
        {
            if (manualUpdate)
            {
                QMessageBox::warning(dynamic_cast<QWidget*>(parent()), "Could not check for updates", reply->errorString() + "\n" + s);
                //QSettings().remove(checkUpdatesTag);
            }
        }
    }
    else if (s.contains("sorry", Qt::CaseInsensitive) ||
             s.contains("error", Qt::CaseInsensitive) ||
             s.contains("fatal", Qt::CaseInsensitive) ||
             s.contains("fail", Qt::CaseInsensitive) ||
             s.contains("html", Qt::CaseInsensitive))
    {
        if (manualUpdate)
        {
            QMessageBox::warning(dynamic_cast<QWidget*>(parent()), "Could not check for updates", s);
            //QSettings().remove(checkUpdatesTag);
        }
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


            typedef ::Ui::SaweMainWindow SaweMainWindowType;
            SaweMainWindowType* mainwindow = dynamic_cast<SaweMainWindowType*>(parent());

            // DropNotifyForm has Qt::WA_DeleteOnClose
            new DropNotifyForm(
                    mainwindow->centralWidget(),
                    mainwindow->getProject()->tools().render_view(),
                    message,
                    url,
                    "Download");

//            int q = QMessageBox::question(dynamic_cast<QWidget*>(parent()), "Sonic AWE updates", message, QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
//            if (q == QMessageBox::Yes)
//                QDesktopServices::openUrl(url);
        }
    }
}

} // namespace Tools
