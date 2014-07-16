#include "harmonicsinfoform.h"
#include "ui_harmonicsinfoform.h"

#include "tooltipcontroller.h"
#include "tooltipview.h"
#include "renderview.h"

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include <QDockWidget>

namespace Tools {

HarmonicsInfoForm::HarmonicsInfoForm(
        Sawe::Project*project,
        TooltipController* harmonicscontroller,
        RenderView* render_view)
            :
    ui(new Ui::HarmonicsInfoForm),
    project(project),
    harmonicscontroller(harmonicscontroller),
    render_view(render_view),
    rebuilding(0)
{
    ui->setupUi(this);

    Ui::SaweMainWindow* MainWindow = project->mainWindow();
    dock = new QDockWidget(MainWindow);
    dock->setObjectName(QString::fromUtf8("dockWidgetHarmonicsInfoForm"));
    dock->setMinimumSize(QSize(42, 79));
    dock->setMaximumSize(QSize(524287, 524287));
    dock->resize(123, 150);
    dock->setContextMenuPolicy(Qt::NoContextMenu);
    dock->setFeatures(QDockWidget::AllDockWidgetFeatures);
    dock->setAllowedAreas(Qt::AllDockWidgetAreas);
    dock->setEnabled(true);
    dock->setWidget(this);
    dock->setWindowTitle("Harmonics info");

    deleteRow = new QAction( this );
    deleteRow->setText("Delete row");
    ui->tableWidget->addAction(deleteRow);
    ui->tableWidget->setContextMenuPolicy( Qt::ActionsContextMenu );
    connect(deleteRow, SIGNAL(triggered()), SLOT(deleteCurrentRow()));

    actionHarmonics_info = new QAction( this );
    actionHarmonics_info->setObjectName("actionHarmonics_info");
    actionHarmonics_info->setText("Harmonics info");
    actionHarmonics_info->setCheckable( true );
    actionHarmonics_info->setChecked( true );

    MainWindow->getItems()->menu_Windows->addAction( actionHarmonics_info );

    connect(actionHarmonics_info, SIGNAL(toggled(bool)), dock, SLOT(setVisible(bool)));
    connect(actionHarmonics_info, SIGNAL(triggered()), dock, SLOT(raise()));
    connect(dock, SIGNAL(visibilityChanged(bool)), SLOT(checkVisibility(bool)));

    connect(harmonicscontroller, SIGNAL(tooltipChanged()), SLOT(harmonicsChanged()));
    connect(ui->tableWidget, SIGNAL(currentCellChanged(int,int,int,int)), SLOT(currentCellChanged()));

    harmonicsChanged();

    dock->setVisible( false );
    actionHarmonics_info->setChecked( false );
}


HarmonicsInfoForm::~HarmonicsInfoForm()
{
    delete ui;
}


void HarmonicsInfoForm::
        checkVisibility(bool visible)
{
    Ui::SaweMainWindow* MainWindow = project->mainWindow();
    visible |= !MainWindow->tabifiedDockWidgets( dock ).empty();
    visible |= dock->isVisibleTo( dock->parentWidget() );
    actionHarmonics_info->setChecked(visible);
}


class CurrentViewUserData: public QObjectUserData
{
public:
    CurrentViewUserData( QPointer<TooltipView> view ):view_(view) {}

    TooltipView* view() { return view_; }
private:
    QPointer<TooltipView> view_;
};

void HarmonicsInfoForm::
        harmonicsChanged()
{
    rebuilding = 2;

    QStringList header;
    header.push_back("Time");
    header.push_back("Frequency");
    header.push_back("Value here");
    header.push_back("Mode");
    header.push_back("Fundamental frequency");
    header.push_back("Harmonic number");
    header.push_back("Tone name");
    header.push_back("Compliance");

    ui->tableWidget->clear();
    ui->tableWidget->setRowCount(0);
    ui->tableWidget->setSelectionBehavior(QAbstractItemView::SelectRows);
    ui->tableWidget->setColumnCount(header.size());
    ui->tableWidget->setHorizontalHeaderLabels( header );
    ui->tableWidget->verticalHeader()->hide();

    QTableWidgetItem*prototype = new QTableWidgetItem;
    prototype->setFlags( prototype->flags() & ~Qt::ItemIsEditable);
    ui->tableWidget->setItemPrototype( prototype );

    //Tfr::FreqAxis fa = project->tools().render_model.display_scale();
    foreach( const QPointer<TooltipView>& view, harmonicscontroller->views() )
    {
        if (!view)
            continue;

        TooltipModel *model = view->model();

        unsigned row = ui->tableWidget->rowCount();
        ui->tableWidget->insertRow( row );
        setCellInLastRow(0, QString("%1").arg(model->pos_time));
        setCellInLastRow(1, QString("%1").arg(model->pos_hz));
        setCellInLastRow(2, QString("%1").arg(model->max_so_far));
        setCellInLastRow(3, QString("%1").arg(model->automarkingStr().c_str()));
        setCellInLastRow(4, QString("%1").arg(model->pos_hz/model->markers));
        setCellInLastRow(5, QString("%1").arg(model->markers));
        setCellInLastRow(6, QString("%1").arg(model->toneName().c_str()));
        setCellInLastRow(7, QString("%1").arg(model->compliance));

        ui->tableWidget->setUserData(row, new CurrentViewUserData(view));
    }

    static bool once_per_process = true;
    if (once_per_process)
    {
        dock->setVisible( true );
        dock->raise();

        Ui::SaweMainWindow* MainWindow = project->mainWindow();
        if (Qt::NoDockWidgetArea == MainWindow->dockWidgetArea(dock))
        {
            MainWindow->addDockWidget(Qt::BottomDockWidgetArea, dock);
        }

        once_per_process = false;
    }

    rebuilding = 1;

    // Select current row
    unsigned row = 0;
    foreach( const QPointer<TooltipView>& view, harmonicscontroller->views() )
    {
        if (!view)
            continue;

        if (view.data() == harmonicscontroller->current_view())
            ui->tableWidget->selectRow(row);

        ++row;
    }
    rebuilding = 0;
}


void HarmonicsInfoForm::
        currentCellChanged()
{
    if (2==rebuilding)
        return;

    int currentRow = ui->tableWidget->currentRow();
    if (currentRow < 0)
    {
        harmonicscontroller->setCurrentView( 0 );
        return;
    }

    QObjectUserData* userData = ui->tableWidget->userData(currentRow);
    CurrentViewUserData* cvud = dynamic_cast<CurrentViewUserData*>(userData);
    if (cvud)
    {
        harmonicscontroller->setCurrentView( cvud->view() );
        if (0==rebuilding)
        {
            render_view->model->setPosition( cvud->view()->model()->pos() );
            render_view->redraw ();
        }
    }
}


void HarmonicsInfoForm::
        deleteCurrentRow()
{
    int currentRow = ui->tableWidget->currentRow();
    if (currentRow < 0)
        return;

    QObjectUserData* userData = ui->tableWidget->userData(currentRow);
    CurrentViewUserData* cvud = dynamic_cast<CurrentViewUserData*>(userData);
    if (cvud)
        delete cvud->view()->model()->comment;
}


void HarmonicsInfoForm::
        setCellInLastRow(int column, QString text)
{
    class QTableReadOnlyText: public QTableWidgetItem
    {
    public:
        QTableReadOnlyText(QString text): QTableWidgetItem(text)
        {
            setFlags( flags() & ~Qt::ItemIsEditable);
        }
    };

    ui->tableWidget->setItem(ui->tableWidget->rowCount()-1, column, new QTableReadOnlyText (text));
}


} // namespace Tools
