#ifndef SELECTION_H
#define SELECTION_H

#include <QQuickItem>

#include "squircle.h"
#include "selectionrenderer.h"

class Selection : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(Squircle* renderOnHeightmap READ renderOnHeightmap WRITE setRenderOnHeightmap NOTIFY renderOnHeightmapChanged)
    Q_PROPERTY(Squircle* filteredHeightmap READ filteredHeightmap WRITE setFilteredHeightmap NOTIFY filteredHeightmapChanged)
    Q_PROPERTY(double t1 READ t1 WRITE setT1 NOTIFY selectionChanged)
    Q_PROPERTY(double t2 READ t2 WRITE setT2 NOTIFY selectionChanged)
    Q_PROPERTY(double f1 READ f1 WRITE setF1 NOTIFY selectionChanged)
    Q_PROPERTY(double f2 READ f1 WRITE setF2 NOTIFY selectionChanged)
    Q_PROPERTY(bool valid READ valid NOTIFY validChanged)
public:
    explicit Selection(QQuickItem *parent = 0);

    Squircle* renderOnHeightmap() const { return render_on_heightmap_; }
    void setRenderOnHeightmap(Squircle*s);

    Squircle* filteredHeightmap() const { return filter_heightmap_; }
    void setFilteredHeightmap(Squircle*s);

    double t1() const {return t1_;}
    double t2() const {return t2_;}
    double f1() const {return f1_;}
    double f2() const {return f2_;}
    void setT1(double v);
    void setT2(double v);
    void setF1(double v);
    void setF2(double v);
    bool valid() const;

signals:
    void filteredHeightmapChanged();
    void renderOnHeightmapChanged();
    void selectionChanged();
    void validChanged();

public slots:
    void discardSelection();

private slots:
    void onRendererChanged(SquircleRenderer* renderer);
    void onSelectionChanged();

private:
    Squircle* filter_heightmap_ = 0;
    Squircle* render_on_heightmap_ = 0;
    QPointer<SelectionRenderer> selection_renderer_;

    double t1_=0, t2_=0;
    float f1_=0, f2_=0;
    bool valid_=false;

    Signal::OperationDesc::ptr::weak_ptr selection_;
};

#endif // SELECTION_H
