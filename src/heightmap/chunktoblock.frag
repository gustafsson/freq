// GLSL fragment shader

uniform sampler2D mytex;
uniform float normalization;
uniform int amplitude_axis;

void main()
{
    float v = sqrt(texture2D(mytex, gl_TexCoord[0].st).r);
    v *= normalization;

    if (0==amplitude_axis)
        v *= 25.0;
    if (1==amplitude_axis) {
        v = 0.019 * log2(v) + 0.3333;
        v = v < 0.0 ? 0.0 : v;
    }

    gl_FragColor = vec4(v, 0, 0, 1);
}
