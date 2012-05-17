#ifndef DROPNOTIFYFORM_H
#define DROPNOTIFYFORM_H

#include <QWidget>
#include <QUrl>
#include <QTimer>

class QVBoxLayout;

namespace Tools {

namespace Ui {
    class DropNotifyForm;
}

class RenderView;

class DropNotifyForm : public QWidget
{
    Q_OBJECT

public:
    explicit DropNotifyForm(QWidget *parent, RenderView* render_view, QString text="", QString url="", QString buttontext="");
    ~DropNotifyForm();

public slots:
    void readMore();
    void ani();

private:
    Ui::DropNotifyForm *ui;
    RenderView* render_view;
    QTimer animate;
    QVBoxLayout* parentLayout;
    int spacing;

    float dt;

    QUrl url;
};


} // namespace Tools
#endif // DROPNOTIFYFORM_H
