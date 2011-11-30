#ifndef GETCUDAFORM_H
#define GETCUDAFORM_H

#include <QWidget>
#include <QUrl>
#include <QTimer>

class QVBoxLayout;

namespace Tools {

namespace Ui {
    class GetCudaForm;
}

class RenderView;

class GetCudaForm : public QWidget
{
    Q_OBJECT

public:
    explicit GetCudaForm(QWidget *parent, RenderView* render_view);
    ~GetCudaForm();

public slots:
    void readMore();
    void ani();

private:
    Ui::GetCudaForm *ui;
    RenderView* render_view;
    QTimer animate;
    QVBoxLayout* parentLayout;
    int spacing;

    float dt;

    static QUrl url;
};


} // namespace Tools
#endif // GETCUDAFORM_H
