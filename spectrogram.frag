// GLSL fragment shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
uniform vec4 deepColor;    // = vec4(0.0, 0.0, 0.1, 1.0);
uniform vec4 shallowColor; // = vec4(0.1, 0.4, 0.3, 1.0);
uniform vec4 skyColor;     // = vec4(0.5, 0.5, 0.5, 1.0);
uniform vec3 lightDir;     // = vec3(0.0, 1.0, 0.0);
varying float intensity;
int clamp(int val, int max) {
    if (val<0) return 0;
    if (val>max) return max;
    return val;
}

vec4 setWavelengthColor( float wavelengthScalar ) {
    vec4 spectrum[] = {
        /* white background */
        vec4( 1, 1, 1, 0 ),
        vec4( 0, 0, 1, 0 ),
        vec4( 0, 1, 1, 0 ),
        vec4( 0, 1, 0, 0 ),
        vec4( 1, 1, 0, 0 ),
        vec4( 1, 0, 1, 0 ),
        vec4( 1, 0, 0, 0 )};
        /* black background
        { 0, 0, 0 },
        { 1, 0, 1 },
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 0, 1, 0 },
        { 1, 1, 0 },
        { 1, 0, 0 }}; */

    int count = 7;//sizeof(spectrum)/sizeof(spectrum[0]);
    float f = count*wavelengthScalar;
    int i = clamp(f, count-1);
    int j = clamp(f+1, count-1);
    float t = f-i;

    vec4 rgb = spectrum[i]*(1-t) + spectrum[j]*t;
    return rgb;
}

void main()
{
    vec3 eyeVector              = normalize(eyeSpacePos);
    vec3 eyeSpaceNormalVector   = normalize(eyeSpaceNormal);
    vec3 worldSpaceNormalVector = normalize(worldSpaceNormal);

    float facing    = max(0.0, dot(eyeSpaceNormalVector, -eyeVector));
    float fresnel   = pow(1.0 - facing, 5.0); // Fresnel approximation
    float diffuse   = max(0.0, dot(worldSpaceNormalVector, lightDir));
    
    vec4 waterColor = mix(shallowColor, deepColor, facing);
    
//    gl_FragColor = gl_Color;
//    gl_FragColor = vec4(fresnel);
//    gl_FragColor = vec4(diffuse);
//    gl_FragColor = waterColor;
//    gl_FragColor = waterColor*diffuse;
//    gl_FragColor = waterColor*diffuse + skyColor*fresnel;
    gl_FragColor = setWavelengthColor( intensity );
//    gl_FragColor = pow(1-intensity,5);
}
