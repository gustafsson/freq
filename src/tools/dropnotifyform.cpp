#include "dropnotifyform.h"
#include "ui_dropnotifyform.h"


#include "renderview.h"


#include <TAni.h>
#include <boost/assert.hpp>

#include <QDesktopServices>
#include <QVBoxLayout>
#include <QSettings>

namespace Tools {

TAni<float> formHeight(1);


DropNotifyForm::
        DropNotifyForm(QWidget *parent, RenderView* render_view, QString text, QString url, QString buttontext)
            :
    QWidget(parent),
    ui(new Ui::DropNotifyForm),
    render_view(render_view),
    //dt(0.01), // enable animated dropdown
    dt(1.0),    // disable animated dropdown
    url("http://muchdifferent.com/?page=signals-cuda")
{
    ui->setupUi(this);

#ifdef USE_CUDA
    ui->labelInfoText->setText("Cool, CUDA works!");
#elif defined(USE_OPENCL)
    QSettings settings;
    QString botherAboutOpenCLtag = "botheraboutopencl";
    if (settings.contains(botherAboutOpenCLtag))
    {
        close();
        return;
    }
    settings.setValue(botherAboutOpenCLtag, false);
    ui->labelInfoText->setText("Cool, OpenCL works! But Sonic AWE is faster with CUDA");
#else
    // TODO figure out if the current computer has a CUDA capable GPU or not
    // TODO change to "Sonic AWE is faster with CUDA (or OpenCL)" when OpenCL is faster than the CPU version.
    ui->labelInfoText->setText("Sonic AWE is faster with CUDA");
#endif

    if (!text.isEmpty())
    {
        ui->labelInfoText->setText(text);
        ui->labelInfoText->setToolTip(text);
        this->url = url;
    }
    if (!buttontext.isEmpty())
        ui->pushButtonReadMore->setText( buttontext );

    connect(ui->pushButtonClose, SIGNAL(clicked()), SLOT(close()));
    connect(ui->pushButtonReadMore, SIGNAL(clicked()), SLOT(readMore()));

    this->setBackgroundRole( QPalette::Mid );
    this->setAutoFillBackground( true );
    this->setAttribute( Qt::WA_DeleteOnClose );

    parentLayout = dynamic_cast<QVBoxLayout*>(parent->layout());
    EXCEPTION_ASSERT( parentLayout );
    parentLayout->insertWidget( 0, this );
    spacing = parentLayout->spacing();
    parentLayout->setSpacing( 0 );

    formHeight = height() + spacing;
    setMaximumHeight( 0 );

    if (dt<1)
    {
        connect(&animate, SIGNAL(timeout()), SLOT(ani()));
        animate.start( 2000 );
    }
    else
    {
        ani();
    }
}


DropNotifyForm::
        ~DropNotifyForm()
{
    delete ui;
}


void DropNotifyForm::
        readMore()
{
    QDesktopServices::openUrl(url);
    close();
}


void DropNotifyForm::
        ani()
{
    formHeight.TimeStep( 2*dt );

    int h = formHeight;
    if (h<spacing)
        parentLayout->setSpacing( h );
    else
        this->setMaximumHeight( h-spacing );

    if (formHeight != &formHeight)
    {
        animate.start( std::max(10, (int)(dt*1000)) );
        render_view->userinput_update();
    }
}


} // namespace Tools
