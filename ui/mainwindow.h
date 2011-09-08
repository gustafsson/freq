#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtGui/QMainWindow>
#include <QtGui/QListWidgetItem>
#include "sawe/project.h"
#include <QTreeWidgetItem>
#include <QComboBox>
#include <QAction>
#include <vector>
#include <QToolButton>
#include <QMessageBox>
#include <QPointer>

#include "comboboxaction.h"

#ifdef Q_WS_MAC
void qt_mac_set_menubar_icons(bool enable);
#endif

// TODO remove
namespace Tools
{
    class RenderController;
    class SelectionView;
}


namespace Ui {

class MainWindow;

class SaweMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    SaweMainWindow(const char* title, Sawe::Project* project, QWidget *parent = 0);
    virtual ~SaweMainWindow();
    
    MainWindow* getItems() { return ui; }

    void disableFullscreen();

signals:
    void onMainWindowCloseEvent( QWidget* closed );

protected:
    virtual void closeEvent(QCloseEvent *);

    struct ActionWindowPair
    {
        QWidget *w; QAction *a;
        ActionWindowPair(QWidget *iw, QAction *ia){w = iw; a = ia;}
    };
    std::vector<ActionWindowPair> controlledWindows;
    

//public slots:
    //void updateOperationsTree( Signal::pOperation s);
    //void updateLayerList( Signal::pOperation s );
    //void slotDbclkFilterItem(QListWidgetItem*);
//    void slotNewSelection(QListWidgetItem*);
//    void slotDeleteSelection(void);
//    void slotCheckWindowStates(bool);
//    void slotCheckActionStates(bool);

//signals:
//    void sendCurrentSelection(int, bool);
//    void sendRemoveItem(int);
private slots:

    void openRecentFile();
#if !defined(TARGET_reader)
    bool askSaveChanges();
    void saveProject();
    void saveProjectAs();
#endif
    void toggleFullscreen( bool );
    void toggleFullscreenNoMenus( bool fullscreen );
    void restoreLayout();
    void reenterProductKey();

    void checkVisibilityToolProperties(bool visible);
private:
    Sawe::Project* project;
    MainWindow *ui;
    ComboBoxAction fullscreen_combo;
    QAction* escape_action;
    QWidget* fullscreen_widget;

    void add_widgets();
};


} // namespace Ui

#endif // MAINWINDOW_H
