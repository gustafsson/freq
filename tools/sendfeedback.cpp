#include "sendfeedback.h"
#include "ui_sendfeedback.h"

#include "sawe/application.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "sawe/reader.h"
#include <TaskTimer.h>

#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QDir>
#include <QSettings>
#include <QNetworkReply>
#include <QMessageBox>
#include <QDesktopServices>
#include <QFileDialog>

namespace Tools {

SendFeedback::
        SendFeedback(::Ui::SaweMainWindow *mainwindow) :
    QDialog(mainwindow),
    ui(new Ui::SendFeedback),
    targetUrl("http://feedback.sonicawe.com/sendfeedback.php")
{
    ui->setupUi(this);

    ::Ui::MainWindow* mui = mainwindow->getItems();
    connect(mui->actionReport_a_bug, SIGNAL(triggered()), SLOT(open()));

    ui->lineEditEmail->setText( Sawe::Reader::name.c_str() );

    connect(ui->pushButtonBrowse, SIGNAL(clicked()), SLOT(browse()));
}


SendFeedback::
        ~SendFeedback()
{
    delete ui;
}


void SendFeedback::
        browse()
{
    QString filename = QFileDialog::getOpenFileName(0, "Find attachment");
    ui->lineEditAttachFile->setText(filename);
}


void SendFeedback::
        accept()
{
    this->setEnabled( false );

    sendLogFiles(
            ui->lineEditEmail->text(),
            ui->textEditMessage->toPlainText(),
            ui->lineEditAttachFile->text() );
}


QString bound = "---------------------------7d935033608e2";

void addFormItem(QByteArray& feedbackdata, QString key, QByteArray value, bool isfile )
{
    QString crlf;
    crlf = 0x0d;
    crlf += 0x0a;

    feedbackdata += "--" + bound + crlf;
    feedbackdata += "Content-Disposition: form-data; name=\"" + QUrl::toPercentEncoding(key) + "\";";
    if (isfile)
    {
        feedbackdata += "filename=\"" + QUrl::toPercentEncoding(key) + "\";";
    }
    feedbackdata += "size=" + QString("%1").arg(value.size()) + "";

    feedbackdata += crlf + "Content-Type: application/octet" + crlf + crlf;
    feedbackdata += value;
    feedbackdata += crlf;
}


void SendFeedback::
        sendLogFiles(QString email, QString message, QString extraFile)
{
    manager.reset( new QNetworkAccessManager(this) );
    connect(manager.data(), SIGNAL(finished(QNetworkReply*)),
            this, SLOT(replyFinished(QNetworkReply*)));

    // thanks abeinoe; http://www.qtcentre.org/threads/18452-How-to-upload-file-to-HTTP-server-(POST-Method)
    QByteArray feedbackdata;

    QString crlf;
    crlf = 0x0d;
    crlf += 0x0a;

    addFormItem(feedbackdata, "email", email.toUtf8(), false);
    addFormItem(feedbackdata, "message", message.toUtf8(), false);
    addFormItem(feedbackdata, "value", QSettings().value("value").toString().toUtf8(), false);

    QString logdir = Sawe::Application::log_directory();
    unsigned count = 0;
    QFileInfoList filesToSend = QDir(logdir).entryInfoList();
    if (QFile::exists(extraFile))
        filesToSend.append(extraFile);

    foreach(QFileInfo f, filesToSend)
    {
        if (!f.isFile())
            continue;

        count++;
        QString p = f.absoluteFilePath();
        QFile mfile(p);
        mfile.open(QIODevice::ReadOnly);
        QByteArray file = mfile.readAll();
        mfile.close();

        addFormItem(feedbackdata, f.fileName(), file, true);
    }
    feedbackdata += "--" + bound + "--" + crlf;


    unsigned N = feedbackdata.size();
    TaskInfo("SendFeedback sends %s in %u files",
             DataStorageVoid::getMemorySizeText(N).c_str(),
             count);


    QNetworkRequest feedbackrequest(targetUrl);
    feedbackrequest.setHeader(QNetworkRequest::ContentTypeHeader,"multipart/form-data; boundary=" + bound);


    manager->post(feedbackrequest, feedbackdata);
}


void SendFeedback::
        replyFinished(QNetworkReply* reply)
{
    QString s = reply->readAll();
    TaskInfo("SendFeedback error=%s\n%s",
             QNetworkReply::NoError == reply->error()?"no error":reply->errorString().toStdString().c_str(),
             s.replace("\\r\\n","\n").replace("\r","").toStdString().c_str());

    if (QNetworkReply::NoError != reply->error())
        QMessageBox::warning(0, "Could not send feedback", reply->errorString() + "\n" + s);
    else if (s.contains("sorry", Qt::CaseInsensitive) || s.contains("error", Qt::CaseInsensitive) || s.contains("fatal", Qt::CaseInsensitive) || s.contains("fail", Qt::CaseInsensitive) || !s.contains("sendfeedback finished", Qt::CaseInsensitive))
        QMessageBox::warning(0, "Could not send feedback", s);
    else
        QDialog::accept();

    setEnabled( true );
}


} // namespace Tools
