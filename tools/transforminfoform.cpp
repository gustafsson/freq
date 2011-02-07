#include "transforminfoform.h"
#include "ui_transforminfoform.h"
#include "rendercontroller.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "heightmap/collection.h"
#include "heightmap/blockfilter.h"
#include "heightmap/renderer.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"

namespace Tools
{

TransformInfoForm::TransformInfoForm(Sawe::Project* project, RenderController* rendercontroller) :
    ui(new Ui::TransformInfoForm),
    project(project),
    rendercontroller(rendercontroller)
{
    ui->setupUi(this);

    Ui::SaweMainWindow* MainWindow = project->mainWindow();
    dock = new QDockWidget(MainWindow);
    dock->setObjectName(QString::fromUtf8("dockWidgetTransformInfoForm"));
    dock->setMinimumSize(QSize(42, 79));
    dock->setMaximumSize(QSize(524287, 524287));
    dock->setContextMenuPolicy(Qt::NoContextMenu);
    //dock->setFeatures(QDockWidget::DockWidgetFeatureMask);
    dock->setFeatures(QDockWidget::AllDockWidgetFeatures);
    dock->setEnabled(true);
    dock->setAutoFillBackground(true);
    dock->setWidget(this);
    dock->setWindowTitle("Transform info");
    dock->show();

    MainWindow->addDockWidget(Qt::RightDockWidgetArea, dock);

    connect(MainWindow->getItems()->actionTransform_info, SIGNAL(toggled(bool)), dock, SLOT(setVisible(bool)));
    connect(dock, SIGNAL(visibilityChanged(bool)), MainWindow->getItems()->actionTransform_info, SLOT(setChecked(bool)));

    connect(rendercontroller, SIGNAL(transformChanged()), SLOT(transformChanged()));

    transformChanged();
}



TransformInfoForm::~TransformInfoForm()
{
    delete ui;
}


void TransformInfoForm::
        transformChanged()
{
    ui->tableWidget->clear();
    ui->tableWidget->setRowCount(0);
    ui->tableWidget->setColumnCount(2);
    QStringList header;
    header.push_back("Name");
    header.push_back("Value");
    ui->tableWidget->setHorizontalHeaderLabels( header );
    ui->tableWidget->verticalHeader()->hide();

    QTableWidgetItem*prototype = new QTableWidgetItem;
    prototype->setFlags( prototype->flags() & ~Qt::ItemIsEditable);
    ui->tableWidget->setItemPrototype( prototype );

    if (rendercontroller->model()->collections.empty())
        return;

    Signal::pOperation s = rendercontroller->model()->collections[0]->postsink();

    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(s.get());
    Tfr::Filter* f = dynamic_cast<Tfr::Filter*>( ps->sinks()[0]->source().get() );
    Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(f->transform().get());
    Tfr::Stft* stft = dynamic_cast<Tfr::Stft*>(f->transform().get());

    float fs = project->head_source()->sample_rate();

    if (cwt)
    {
        addRow("Type", "Morlet wavelet");
        addRow("T/F resolution", QString("%1").arg(cwt->tf_resolution()));
        addRow("Time support", QString("%1").arg(cwt->wavelet_time_support_samples( fs )/fs));
        addRow("Scales", QString("%1").arg(cwt->nScales(fs)));
        addRow("Scales per octave", QString("%1").arg(cwt->scales_per_octave()));
        addRow("Sigma", QString("%1").arg(cwt->sigma()));
        addRow("Bins", QString("%1").arg(cwt->find_bin( cwt->scales_per_octave())));
        addRow("Max hz", QString("%1").arg(cwt->get_max_hz(fs)));
        addRow("Min hz", QString("%1").arg(cwt->get_min_hz(fs)));
        addRow("Amplification factor", QString("%1").arg(rendercontroller->model()->renderer->y_scale));
    }
    else if (stft)
    {
        addRow("Type", "Short time fourier");
        addRow("Window type", "Regular");
        addRow("Window size", QString("%1").arg(stft->chunk_size()));
        addRow("Overlap", "0");
        addRow("Amplification factor", QString("%1").arg(rendercontroller->model()->renderer->y_scale));
    }
    else
    {
        addRow("Type", "Unknown");
        addRow("Error", "Doesn't recognize transform");
    }
}


void TransformInfoForm::
        addRow(QString name, QString value)
{
    class QTableReadOnlyText: public QTableWidgetItem
    {
    public:
        QTableReadOnlyText(QString text): QTableWidgetItem(text)
        {
            setFlags( flags() & ~Qt::ItemIsEditable);
        }

    };

    ui->tableWidget->insertRow( ui->tableWidget->rowCount());
    ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 0, new QTableReadOnlyText (name));
    ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, 1, new QTableReadOnlyText (value));
}

} // namespace Tools
