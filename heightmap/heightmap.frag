// GLSL fragment shader
//varying vec3 eyeSpacePos;
//varying vec3 worldSpaceNormal;
//varying vec3 eyeSpaceNormal;
varying float vertex_height;
varying float shadow;


uniform sampler2D tex;
uniform sampler2D tex_color;
uniform int colorMode;
uniform int heightLines;
uniform float yScale;
uniform vec4 fixedColor;

vec4 getWavelengthColor( float wavelengthScalar ) {
    return texture2D(tex_color, vec2(wavelengthScalar,0));
}

float getHeightLineColor(float height)
{
   float value = height - floor(height);
   return value > 0.1 ? 1.0 : 0.75;
}

void main()
{
    float v = texture2D(tex, gl_TexCoord[0].xy).x;

    v *= yScale;

    vec4 curveColor;

    float f = abs(v);

    if (colorMode == 0) // rainbow
    {
        curveColor = getWavelengthColor( f );
        f = 1.0 - (1.0-f)*(1.0-f)*(1.0-f);
    }
    else if (colorMode == 2) // colorscale
    {
        curveColor = fixedColor;
        if (v<0.0) {curveColor = 1.0-curveColor;}
    }
    else // grayscale
    {
        curveColor = vec4(1.0);
    }

    //float fresnel   = pow(1.0 - facing, 5.0); // Fresnel approximation
    curveColor = curveColor*shadow; // + vec4(fresnel);
    curveColor = mix(vec4(1.0), curveColor, f);

    if (0!=heightLines)
    {
        if (vertex_height != 0.0)
            v = vertex_height;
        float heightLine1 = getHeightLineColor( abs(v) * 20.0);
        float heightLine2 = getHeightLineColor( abs(v) * 5.0);
        curveColor = heightLine1 *heightLine2*heightLine2* curveColor;
    }

    curveColor.w = 1.0; //-saturate(fresnel);
    gl_FragColor = curveColor;
}
