// GLSL fragment shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
varying float vertex_height;
/*
uniform vec4 deepColor;    // = vec4(0.0, 0.0, 0.1, 1.0);
uniform vec4 shallowColor; // = vec4(0.1, 0.4, 0.3, 1.0);
uniform vec4 skyColor;     // = vec4(0.5, 0.5, 0.5, 1.0);
uniform vec3 lightDir;     // = vec3(0.0, 1.0, 0.0);
*/
uniform sampler2D tex;
uniform sampler2D tex_color;
uniform int colorMode;
uniform int heightLines;
uniform float yScale;
uniform vec4 fixedColor;

vec4 getWavelengthColor( float wavelengthScalar ) {
    return texture2D(tex_color, wavelengthScalar);
}

/* Tremendously slow... way faster to interpolate from a small texture 'tex_color' instead */
vec4 getWavelengthColorCompute( float wavelengthScalar ) {
    vec4 spectrum[7];
        /* white background */
    spectrum[0] = vec4( 1, 0, 0, 0 ),
    spectrum[1] = vec4( 0, 0, 1, 0 ),
    spectrum[2] = vec4( 0, 1, 1, 0 ),
    spectrum[3] = vec4( 0, 1, 0, 0 ),
    spectrum[4] = vec4( 1, 1, 0, 0 ),
    spectrum[5] = vec4( 1, 0, 1, 0 ),
    spectrum[6] = vec4( 1, 0, 0, 0 );
        /* black background
        { 0, 0, 0 },
        { 1, 0, 1 },
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 0, 1, 0 },
        { 1, 1, 0 },
        { 1, 0, 0 }}; */

    int count = 6;//sizeof(spectrum)/sizeof(spectrum[0])-1;
    float f = float(count)*wavelengthScalar;
    int i1 = int(floor(max(0.0, min(f-1.0, float(count)))));
    int i2 = int(floor(min(f, float(count))));
    int i3 = int(floor(min(f+1.0, float(count))));
    int i4 = int(floor(min(f+2.0, float(count))));
    float t = (f-float(i2))*0.5;
    float s = 0.5 + t;

    vec4 rgb = mix(spectrum[i1], spectrum[i3], s) + mix(spectrum[i2], spectrum[i4], t);
    return rgb*0.5;
}

float getHeightLineColor(float height)
{
   float value = height - floor(height);
   value = 1.0 - value * value * value * value + 0.1;
   
   //float value2 = height*10.0 - floor(height*10.0);
   //value2 = 1.0 - value2 * value2 * value2 * value2 + 0.1;
   
   //value2 = clamp(value2 + max(0.0 , eyeSpacePos.z - 1.0)/3.0, 0.0, 1.0);
   //value = value * (0.5 + value2 * 0.5);
   return clamp(sqrt(value), 0.0, 1.0);
}

void main()
{
//    vec3 eyeVector              = normalize(eyeSpacePos);
//    vec3 eyeSpaceNormalVector   = normalize(eyeSpaceNormal);
//    vec3 worldSpaceNormalVector = normalize(worldSpaceNormal);
    vec3 eyeVector              = eyeSpacePos;
    vec3 eyeSpaceNormalVector   = eyeSpaceNormal;
    vec3 worldSpaceNormalVector = worldSpaceNormal;

    float facing    = max(0.0, dot(eyeSpaceNormalVector, -eyeVector));
    float fresnel   = pow(1.0 - facing, 5.0); // Fresnel approximation
    float diffuse   = max(0.0, worldSpaceNormalVector.y); // max(0.0, dot(worldSpaceNormalVector, lightDir));
    float v = texture2D(tex, gl_TexCoord[0].xy).x;

    v *= yScale;

//    float v=texPos.y;
    //float v = intensity;

//    vec4 waterColor = mix(shallowColor, deepColor, facing);

//    gl_FragColor = gl_Color;
//    gl_FragColor = vec4(fresnel);
//    gl_FragColor = vec4(diffuse);
//    gl_FragColor = waterColor;
//    gl_FragColor = waterColor*diffuse;
//    gl_FragColor = waterColor*diffuse + skyColor*fresnel;
//    gl_FragColor = vec4(pow(1.0-v,5.0));
//    gl_FragColor = setWavelengthColor( v );
    float f = 1.0-pow(1.0-clamp(v, 0.0, 1.0),5.0);
    vec4 curveColor;
    float x;

    switch (colorMode) {
        case 0: curveColor = getWavelengthColor( f );
                x = 1.0-(1.0-f)*(1.0-f)*(1.0-f);
                curveColor = curveColor*((diffuse+facing+2.0)*.25); // + vec4(fresnel);
                curveColor = mix(vec4( 1,1,1,0), min(vec4(0.7),curveColor), x);
        break;
        case 1: f = 1.0-f;
                curveColor = vec4( f, f, f, 0 );
                x = 1.0-f;
        break;
        case 2: curveColor = vec4( 1.0 - fixedColor.x*f, 1.0 - fixedColor.y*f, 1.0 - fixedColor.z*f, 0 );
                f = 1.0-f;
                x = 1.0-f;
        break;
    }

    if (0!=heightLines)
    {
        float heightLine = getHeightLineColor( log(vertex_height+ 0.000001) );
        curveColor = vec4(heightLine, heightLine, heightLine, 1) * curveColor;
    }

    curveColor.w = 1.0; //-saturate(fresnel);
    gl_FragColor = curveColor;
}
