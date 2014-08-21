#ifndef COMMENTVIEW_H
#define COMMENTVIEW_H

#include "commentmodel.h"
#include "sawe/project.h"

#include <QWidget>
#include <QGraphicsScene>

namespace Ui { class CommentView; }

namespace Tools {

class RenderView;

class CommentView : public QWidget
{
    Q_OBJECT

public:
    CommentView(ToolModelP modelp, QGraphicsScene* graphicsscene, RenderView* render_view, Sawe::Project* project, QWidget *parent=0);
    ~CommentView();

    std::string html();
    void setHtml(std::string);

    RenderView* view;
    QGraphicsProxyWidget* getProxy();
    CommentModel* model();
    ToolModelP modelp;

    virtual void closeEvent(QCloseEvent *);
    virtual void wheelEvent(QWheelEvent *);
    virtual void resizeEvent(QResizeEvent *);
    virtual void paintEvent(QPaintEvent *);
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseDoubleClickEvent ( QMouseEvent * event );
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual QSize sizeHint() const;

    void setEditFocus(bool focus);
    bool isThumbnail();
    void resize(int w, int h);
    void resize(QSize);

signals:
    void thumbnailChanged( bool );
    void gotFocus();

public slots:
    void updatePosition();
    void updateText();
    void recreatePolygon();
    void thumbnail(bool);

private:
    void validateSize();

    ::Ui::CommentView *ui;

    QGraphicsProxyWidget* proxy;
    QWidget* containerWidget;
    Sawe::Project* project;

    QPoint ref_point;
    QPolygonF poly;
    bool keep_pos;
    bool z_hidden;
    QPoint dragPosition;
    QPoint resizePosition;
    double lastz;
    QRegion maskedRegion;
    bool validateSizeLater;

    bool testFocus();
};

} // namespace Tools

#endif // COMMENTVIEW_H
