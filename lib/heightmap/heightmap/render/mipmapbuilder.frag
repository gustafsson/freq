uniform mediump sampler2D qt_Texture0;

varying mediump vec2 qt_TexCoord0;
varying mediump vec2 qt_TexCoord1;
varying mediump vec2 qt_TexCoord2;
varying mediump vec2 qt_TexCoord3;

void main(void)
{
    mediump vec4 v = vec4(
                    texture2D(qt_Texture0, qt_TexCoord0, -1.0).x,
                    texture2D(qt_Texture0, qt_TexCoord1, -1.0).x,
                    texture2D(qt_Texture0, qt_TexCoord2, -1.0).x,
                    texture2D(qt_Texture0, qt_TexCoord3, -1.0).x);

    mediump float r;
#if defined(MipmapOperator_ArithmeticMean)
    r = (v.x+v.y+v.z+v.w)*0.25;
#elif defined(MipmapOperator_GeometricMean)
    r = sqrt(sqrt(v.x*v.y*v.z*v.w));
    //r = pow(v.x*v.y*v.z*v.w,0.25);
#elif defined(MipmapOperator_HarmonicMean)
    r = 4./(1./v.x + 1./v.y + 1./v.z + 1./v.w);
#elif defined(MipmapOperator_QuadraticMean)
    r = length(v)*0.5;
    //r = sqrt((v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w)*0.25);
#elif defined(MipmapOperator_CubicMean)
    r = pow((v.x*v.x*v.x+v.y*v.y*v.y+v.z*v.z*v.z+v.w*v.w*v.w)*0.25, 1.0/3.0);
#elif defined(MipmapOperator_Max)
    r = max(max(v.x,v.y),max(v.z,v.w));
#elif defined(MipmapOperator_Min)
    r = min(min(v.x,v.y),min(v.z,v.w));
#elif defined(MipmapOperator_OTA)
    // set the smallest in v.x and the largest value in v.w (i.e almost sort v.xyzw ascending)
    if (v.x>v.y) v.xy = v.yx;
    if (v.z>v.w) v.zw = v.wz;
    if (v.y>v.z) v.yz = v.zy;
    if (v.x>v.y) v.xy = v.yx;
    if (v.z>v.w) v.zw = v.wz;
    // v.y and v.z might not be sorted, but that doesn't matter
    // discard min and max, take mean of the middle
    r = (v.y+v.z)/2.0;
#endif
    gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}
