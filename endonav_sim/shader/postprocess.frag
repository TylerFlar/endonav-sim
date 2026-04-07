#version 330

// Resolves the supersampled scene texture into the output framebuffer with:
//   - letterbox bars (black) outside the active image region
//   - box-filter supersample resolve (AA)
//   - mild radial chromatic aberration (R outward, B inward)
//   - per-pixel sensor noise (uncorrelated grain)
// All effects are tuned to qualitatively match the real phantom camera
// (G:/01_Active/Code/arclab/_datasets/artificial_kidney_path_data).

uniform sampler2D scene_tex;
uniform int   supersample;
uniform vec2  texel;            // 1 / scene_tex_size  (active region size, ss)
uniform vec2  frame_size;       // output frame px (e.g. 1024x768)
uniform vec2  active_size;      // active image region px (e.g. 870x760)
uniform vec2  bar_offset;       // (left, top) bar offsets in px
uniform float chroma_strength;  // ~0.004
uniform float noise_strength;   // ~0.025
uniform float frame_id;         // increments per render -> animated noise

in vec2 v_uv;
out vec4 fragColor;

// Cheap hash → uniform random in [0,1).
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 box_sample(vec2 uv) {
    vec3 acc = vec3(0.0);
    int N = supersample;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            vec2 offset = (vec2(float(i), float(j)) + 0.5 - float(N) * 0.5) * texel;
            acc += texture(scene_tex, uv + offset).rgb;
        }
    }
    return acc / float(N * N);
}

void main() {
    // Output pixel position in frame coordinates.
    vec2 frag_px = v_uv * frame_size;
    vec2 active_px = frag_px - bar_offset;

    if (active_px.x < 0.0 || active_px.y < 0.0 ||
        active_px.x >= active_size.x || active_px.y >= active_size.y) {
        // Letterbox bar.
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 uv = active_px / active_size;             // [0,1] within active rect

    // Chromatic aberration: radial RGB shift around the active-region center.
    vec2 c = uv - 0.5;
    vec2 uv_r = 0.5 + c * (1.0 + chroma_strength);
    vec2 uv_b = 0.5 + c * (1.0 - chroma_strength);
    vec3 col;
    col.r = box_sample(uv_r).r;
    col.g = box_sample(uv).g;
    col.b = box_sample(uv_b).b;

    // Sensor noise — tiny CMOS, security-camera quality. We deliberately
    // use BLOCKY/clumpy noise (low spatial frequency) instead of per-pixel
    // grain to mimic h264 compression artifacts on a low-bitrate stream.
    //
    // Quantize the pixel coordinate so neighbours within a block share the
    // same hash sample. Multiple block sizes are summed for a natural look:
    //   - 4x4 luma blocks (the dominant compression-style grain)
    //   - 2x2 chroma blocks
    //   - 8x8 occasional macroblock judder
    vec2 px = frag_px;
    vec2 b_lum = floor(px / 4.0);
    vec2 b_chr = floor(px / 2.0);
    vec2 b_mac = floor(px / 8.0);

    float n_lum = hash12(b_lum + vec2(frame_id * 13.137, frame_id * 7.713)) - 0.5;
    float n_r = hash12(b_chr + vec2(frame_id * 5.231,  91.7)) - 0.5;
    float n_g = hash12(b_chr + vec2(frame_id * 8.911,  17.3)) - 0.5;
    float n_b = hash12(b_chr + vec2(frame_id * 11.717, 53.9)) - 0.5;
    float n_mac = hash12(b_mac + vec2(frame_id * 3.71, 200.7)) - 0.5;

    // Brightness-dependent gain: darker pixels get full noise, near-white
    // pixels get half (cheap CMOS still has noise even when bright but the
    // perceived effect is much milder than on dark areas).
    float luma_pre = dot(col, vec3(0.299, 0.587, 0.114));
    float bright_atten = mix(1.0, 0.45, smoothstep(0.4, 0.9, luma_pre));

    col += vec3(n_lum) * noise_strength * bright_atten;
    col += vec3(n_r, n_g, n_b) * (noise_strength * 0.45 * bright_atten);

    // Macroblock judder — darker = more visible, scales with sqrt(1-luma).
    col += vec3(n_mac) * noise_strength * 1.0 * sqrt(max(1.0 - luma_pre, 0.0));

    fragColor = vec4(col, 1.0);
}
