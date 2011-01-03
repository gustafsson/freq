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
    return texture2D(tex_color, vec2(wavelengthScalar,0));
}

float getHeightLineColor(float height)
{
   float value = height - floor(height);
   return value > 0.1 ? 1.0 : 0.75;
//   value = 1.0 - value * value * value * value + 0.1;
   
   //float value2 = height*10.0 - floor(height*10.0);
   //value2 = 1.0 - value2 * value2 * value2 * value2 + 0.1;
   
   //value2 = clamp(value2 + max(0.0 , eyeSpacePos.z - 1.0)/3.0, 0.0, 1.0);
   //value = value * (0.5 + value2 * 0.5);

//   return clamp(sqrt(value), 0.0, 1.0);
//   return value;
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

//    vec4 waterColor = mix(shallowColor, deepColor, facing);

    vec4 curveColor;

    float f = abs(v);

   switch (colorMode) {
        case 0: curveColor = getWavelengthColor( f );
                f = 1.0 - (1.0-f)*(1.0-f)*(1.0-f);
        break;
        case 1: curveColor = vec4(0.0);
        break;
        case 2: curveColor = fixedColor;
                if (v<0.0) {curveColor = 1.0-curveColor;}
        break;
    }

    float shadow = min(0.7, ((diffuse+facing+2.0)*.25)); // + vec4(fresnel);
    curveColor = curveColor*shadow;
    curveColor = mix(vec4(1.0), curveColor, f);

    if (0!=heightLines)
    {
        float heightLine1 = getHeightLineColor( abs(vertex_height) * 20.0);
        float heightLine2 = getHeightLineColor( abs(vertex_height) * 5.0);
        curveColor = heightLine1 *heightLine2*heightLine2* curveColor;
    }

    curveColor.w = 1.0; //-saturate(fresnel);
    gl_FragColor = curveColor;
}
