#include "splashscreen.h"
#include "ui_splashscreen.h"

#include <math.h>

#include <QTimer>

namespace Tools {

SplashScreen
        ::SplashScreen(QWidget *parent)
            :
    QDialog(parent),
    ui(new Ui::SplashScreen)
{
    ui->setupUi(this);

    setWindowFlags(Qt::FramelessWindowHint);

    ui->progressBar->setValue(0);

    tickLoader();

    show();
}


SplashScreen::
        ~SplashScreen()
{
    delete ui;
}


void SplashScreen::
        tickLoader()
{
    if (ui->progressBar->value() >= ui->progressBar->maximum())
    {
        close();
        return;
    }

    ui->progressBar->setValue( ui->progressBar->value() + 1 );

    float v = (ui->progressBar->value() - ui->progressBar->minimum())
            / (float)(ui->progressBar->maximum()  - ui->progressBar->minimum());

    if (.6 < v)
    {
        ui->labelText->setText("Establishing power through brick building");
    }

    this->setWindowOpacity(0.9 + 0.1*fabs(fmodf(v*10, 2)-1));

    if (1 > v)
        QTimer::singleShot(10, this, SLOT(tickLoader()));
    else
    {
        ui->labelText->setText("World domination complete");
        QTimer::singleShot(1000, this, SLOT(tickLoader()));
    }
}

} // namespace Tools
