// GLSL fragment shader

uniform sampler2D mytex;
uniform float normalization;

void main()
{
    vec2 t = texture2D(mytex, gl_TexCoord[0].st).rg;
    gl_FragColor = vec4(length(t)*normalization, 0, 0, 1);
}
