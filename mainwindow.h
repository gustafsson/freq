#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include <QtGui/QListWidgetItem>
#include "displaywidget.h"
#include "tfr-filter.h"
#include <QTreeWidgetItem>
#include <QComboBox>
#include <QAction>

#ifdef Q_WS_MAC
void qt_mac_set_menubar_icons(bool enable);
#endif


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(const char* title, QWidget *parent = 0);
    ~MainWindow();
    
    void connectLayerWindow(DisplayWidget *d);
    void setTimelineWidget( QWidget* );

protected:
//    virtual void keyPressEvent( QKeyEvent *e );
//    virtual void keyReleaseEvent ( QKeyEvent * e );
    virtual void closeEvent(QCloseEvent *);

public slots:
    void updateOperationsTree( Signal::pSource s);
    void updateLayerList( Tfr::pFilter f );
    void slotDbclkFilterItem(QListWidgetItem*);
    void slotNewSelection(QListWidgetItem*);
    void slotDeleteSelection(void);
    void slotCheckWindowStates(bool);

signals:
    void sendCurrentSelection(int, bool);
    void sendRemoveItem(int);

private:
    class Ui_MainWindow *ui;
    
    void connectActionToWindow(QObject *a, QObject *b);
};


class QComboBoxAction: public QComboBox {
    Q_OBJECT
public:
    QComboBoxAction() {
        connect( this, SIGNAL(currentIndexChanged(int)), this, SLOT(invokeAction(int)));
    }

    void addActionItem( QAction* a ) {
        addItem( a->icon(), "" /*a->text()*/, qVariantFromValue( (QObject*)a ) );
    }

private slots:
    virtual void invokeAction( int i ) {
        QObject* p = itemData( i ).value<QObject*>();
        dynamic_cast<QAction*>(p)->trigger();
    }
};

#endif // MAINWINDOW_H
