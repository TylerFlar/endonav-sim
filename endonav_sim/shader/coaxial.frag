#version 330

// Coaxial endoscope lighting, derived from:
//   - EndoPBR (arXiv 2502.20669, 2025): Disney/Cook-Torrance microfacet
//     BRDF + spotlight L = L0 cos^n(theta) / d^q.
//   - NVIDIA GPU Gems Ch 16: wrap-around diffuse + warm SSS shadow tint.
//   - Photo-Realistic Tissue Reflectance Modelling (Dey et al. MICCAI '05):
//     thin specular mucus over diffuse-with-SSS dermis.
//
// Coaxial simplification: the camera IS the light source, so L == V == H.
// This collapses GGX so N.H = N.L = N.V everywhere — only NdotL needed.

const float PI = 3.14159265358979;

uniform vec3  cameraPos;       // light position (== camera, coaxial)
uniform vec3  cameraForward;   // unit vector, +z of camera in world
uniform float lightScale;      // overall radiometric gain
uniform float specStrength;    // 0..1 specular weight
uniform float specPower;       // controls roughness (higher = sharper)
uniform float spotExp;         // EndoPBR spotlight cosine exponent (n)

in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_color;

out vec4 fragColor;

void main() {
    // ---- Geometry -------------------------------------------------------
    vec3  N    = normalize(v_normal);
    vec3  Lvec = cameraPos - v_world_pos;
    float r    = length(Lvec);
    vec3  L    = Lvec / max(r, 1e-4);
    // L == V == H in the coaxial case.
    float NdotL_raw = dot(N, L);
    float NdotL = max(NdotL_raw, 0.0);
    float NdotH = NdotL;
    float NdotV = NdotL;

    // ---- EndoPBR spotlight: cos^n(theta) / d^q --------------------------
    // theta = angle between camera forward and the light->surface direction.
    // cameraForward dot (-L) since L points from surface to camera.
    float spot_cos = clamp(dot(cameraForward, -L), 0.0, 1.0);
    float spotlight = pow(spot_cos, spotExp);
    float light_falloff = spotlight / max(r * r, 1e-4);     // q = 2

    // ---- Diffuse: wrap-around for cheap SSS softening -------------------
    float wrap = 0.45;
    float wrapped = max((NdotL_raw + wrap) / (1.0 + wrap), 0.0);

    // SSS warm tint on the shadow side (Henrik Wann Jensen's skin trick).
    vec3  sss_tint = vec3(1.0, 0.55, 0.55);
    float shade    = 1.0 - smoothstep(0.0, 0.6, NdotL_raw);
    vec3  diffuse_color = mix(v_color, v_color * sss_tint, shade * 0.7);
    vec3  diffuse = diffuse_color * wrapped * light_falloff;

    // ---- Specular: Cook-Torrance/GGX, simplified for L==V ---------------
    // Map our existing specPower knob to a Disney roughness in [0.05, 0.95].
    float roughness = clamp(1.0 / (1.0 + specPower * 0.3), 0.05, 0.95);
    float a  = roughness * roughness;
    float a2 = a * a;
    // D (GGX normal distribution)
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * denom * denom + 1e-6);
    // F (Schlick), F0 ~ 0.04 dielectric. Use NdotL fallback since VdotH==1.
    float F0 = 0.04;
    float F  = F0 + (1.0 - F0) * pow(1.0 - NdotL, 5.0);
    // G (Smith / Schlick-GGX), simplified for NdotL == NdotV.
    float k_  = (roughness + 1.0);
    k_ = (k_ * k_) * 0.125;
    float Gv = NdotV / (NdotV * (1.0 - k_) + k_);
    float G  = Gv * Gv;
    float spec_num = D * F * G;
    float spec_den = 4.0 * NdotL * NdotV + 1e-4;
    vec3  specular = vec3(spec_num / spec_den) * specStrength * light_falloff;

    // ---- Combine and tonemap -------------------------------------------
    vec3 color = (diffuse + specular) * lightScale;

    // ACES filmic-fit tonemap (Krzysztof Narkowicz). Better highlight
    // shoulder than Reinhard while keeping mids close, so the brightness
    // falloff validator stays approximately the same shape.
    const float ta = 2.51, tb = 0.03, tc = 2.43, td = 0.59, te = 0.14;
    color = clamp((color * (ta * color + tb)) / (color * (tc * color + td) + te), 0.0, 1.0);

    fragColor = vec4(color, 1.0);
}
