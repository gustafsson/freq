#include "sendfeedbackdialog.h"
#include "ui_sendfeedback.h"

#include "support/sendfeedback.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"
#include "sawe/reader.h"
#include "tasktimer.h"

#include <QMessageBox>
#include <QFileDialog>

namespace Tools {

SendFeedbackDialog::
        SendFeedbackDialog(::Ui::SaweMainWindow *mainwindow) :
    QDialog(mainwindow),
    sendfeedback(new Support::SendFeedback(this)),
    ui(new Ui::SendFeedback)
{
    ui->setupUi(this);

    if (mainwindow) {
        ::Ui::MainWindow* mui = mainwindow->getItems();
        connect(mui->actionReport_a_bug, SIGNAL(triggered()), SLOT(open()));
    }

    ui->lineEditEmail->setText( Sawe::Reader::name.c_str() );

    connect(ui->pushButtonBrowse, SIGNAL(clicked()), SLOT(browse()));

    connect(sendfeedback, SIGNAL(finished(QNetworkReply*)),
            this, SLOT(replyFinished(QNetworkReply*)));
}


SendFeedbackDialog::
        ~SendFeedbackDialog()
{
    delete ui;
}


void SendFeedbackDialog::
        browse()
{
    QString filename = QFileDialog::getOpenFileName(0, "Find attachment");
    ui->lineEditAttachFile->setText(filename);
}


void SendFeedbackDialog::
        accept()
{
    this->setEnabled( false );

    QString omittedMessage = sendfeedback->sendLogFiles(
            ui->lineEditEmail->text(),
            ui->textEditMessage->toPlainText(),
            ui->lineEditAttachFile->text() );

    if (omittedMessage.isEmpty()) {
        QMessageBox::information(
                    dynamic_cast<QWidget*>(parent()),
                    "Some files were to large",
                    omittedMessage);
    }
}


void SendFeedbackDialog::
        replyFinished(QNetworkReply* reply)
{
    QString s = reply->readAll();
    TaskInfo ti("SendFeedback reply");
    TaskInfo("%s", s.replace("\\r\\n","\n").replace("\r","").toStdString().c_str());
    if (QNetworkReply::NoError != reply->error())
    {
        TaskInfo("SendFeedback error=%s (code %d)",
                 QNetworkReply::NoError == reply->error()?"no error":reply->errorString().toStdString().c_str(),
                 (int)reply->error());
    }

    if (QNetworkReply::NoError != reply->error())
        QMessageBox::warning(dynamic_cast<QWidget*>(parent()), "Could not send feedback", reply->errorString() + "\n" + s);
    else if (s.contains("sorry", Qt::CaseInsensitive) ||
             s.contains("error", Qt::CaseInsensitive) ||
             s.contains("fatal", Qt::CaseInsensitive) ||
             s.contains("fail", Qt::CaseInsensitive) ||
             s.contains("html", Qt::CaseInsensitive) ||
             !s.contains("sendfeedback finished", Qt::CaseInsensitive))
    {
        QMessageBox::warning(dynamic_cast<QWidget*>(parent()), "Could not send feedback", s);
    }
    else
    {
        QDialog::accept();
        QMessageBox::information(dynamic_cast<QWidget*>(parent()), "Feedback", "Your input has been sent. Thank you!");
        ui->textEditMessage->setText ("");
        ui->lineEditAttachFile->setText ("");
    }

    setEnabled( true );
}


} // namespace Tools
