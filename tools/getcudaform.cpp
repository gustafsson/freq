#include "getcudaform.h"
#include "ui_getcudaform.h"


#include "renderview.h"


#include <TAni.h>
#include <boost/assert.hpp>

#include <QDesktopServices>
#include <QVBoxLayout>
#include <QSettings>

namespace Tools {

TAni<float> formHeight(1);

QUrl GetCudaForm::url("http://muchdifferent.com/?page=signals-cuda");


GetCudaForm::
        GetCudaForm(QWidget *parent, RenderView* render_view)
            :
    QWidget(parent),
    ui(new Ui::GetCudaForm),
    render_view(render_view),
    dt(0.01)
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

    connect(ui->pushButtonClose, SIGNAL(clicked()), SLOT(close()));
    connect(ui->pushButtonReadMore, SIGNAL(clicked()), SLOT(readMore()));

    this->setBackgroundRole( QPalette::Mid );
    this->setAutoFillBackground( true );

    parentLayout = dynamic_cast<QVBoxLayout*>(parent->layout());
    BOOST_ASSERT( parentLayout );
    parentLayout->insertWidget( 0, this );
    spacing = parentLayout->spacing();
    parentLayout->setSpacing( 0 );

    formHeight = height() + spacing;
    setMaximumHeight( 0 );

    connect(&animate, SIGNAL(timeout()), SLOT(ani()));
    animate.start( 2000 );
}


GetCudaForm::
        ~GetCudaForm()
{
    delete ui;
}


void GetCudaForm::
        readMore()
{
    QDesktopServices::openUrl(url);
    close();
}


void GetCudaForm::
        ani()
{
    int h = formHeight;
    if (h<spacing)
        parentLayout->setSpacing( h );
    else
        this->setMaximumHeight( h-spacing );

    formHeight.TimeStep( 2*dt );

    if (formHeight != &formHeight)
    {
        animate.start( std::max(10, (int)(dt*1000)) );
        render_view->userinput_update();
    }
}



} // namespace Tools
