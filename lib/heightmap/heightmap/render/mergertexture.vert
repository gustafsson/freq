attribute vec4 qt_Vertex;
attribute vec2 qt_MultiTexCoord0;
uniform mat4 qt_ModelViewMatrix;
uniform mat4 qt_ProjectionMatrix;
varying vec2 qt_TexCoord0;

void main(void)
{
    gl_Position = qt_ProjectionMatrix * qt_ModelViewMatrix * qt_Vertex;
    qt_TexCoord0 = qt_MultiTexCoord0;
}
