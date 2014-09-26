#ifndef ZOOMCAMERACOMMAND_H
#define ZOOMCAMERACOMMAND_H

#include "navigationcommand.h"

namespace Tools {
    class RenderModel;
namespace Commands {

class ZoomCameraCommand : public Tools::Commands::NavigationCommand
{
public:
    ZoomCameraCommand(Tools::RenderModel* model, float dt, float ds, float dz);

    virtual std::string toString();
    virtual void executeFirst();

private:
    float dt, ds, dz;

    enum ZoomMode {
        Zoom,
        ScaleX,
        ScaleZ
    };

    bool zoom(Tools::Support::RenderCamera&, float delta, ZoomMode mode);
    static void doZoom(float delta, float* scale, float* min_scale, float* max_scale, double& p2);
};

} // namespace Commands
} // namespace Tools

#endif // ZOOMCAMERACOMMAND_H
