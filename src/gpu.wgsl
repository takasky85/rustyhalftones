// Halftone preview shader. A per-pixel port of PlateEval::is_on / tone_preview in
// engine/render.rs: rotate the pixel into the screen lattice, sample the ink map at the
// cell centre, adjust the tone, and threshold against the spot rank.

const TAU: f32 = 6.2831853;
const BINS: f32 = 1024.0;
const MAX_LAYERS: u32 = 8u;

struct Layer {
    ink: vec4<f32>,                    // rgb, unused
    p0: vec4<f32>,                     // cos, sin, pitch mm, cell px (ink-map pixels per screen cell)
    p1: vec4<f32>,                     // dot gain, min dot, max dot, dropout
    p2: vec4<f32>,                     // invert, shape id, elongation, spot min
    p3: vec4<f32>,                     // spot inv, plate index, patch count, unused
    patch_rect: array<vec4<f32>, 12>,  // x, y, w, h in page mm
    patch_tone: array<vec4<f32>, 3>,   // 12 nominal tones
};

struct Frame {
    hdr0: vec4<f32>,                   // page origin x, y (framebuffer px), px per mm, layer count
    image: vec4<f32>,                  // image rect x, y, w, h (page mm)
    hdr1: vec4<f32>,                   // page w, page h, dots (1) or tone (0), antialias
    hdr2: vec4<f32>,                   // max lod, ink map base width, target is sRGB, unused
    layers: array<Layer, 8>,
};

struct Spots {
    cum: array<vec4<f32>, 3080>,       // 8 plates x 1028 floats, then the 64 x 64 blue-noise mask
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> spots: Spots;
@group(0) @binding(2) var ink_tex: texture_2d_array<f32>;
@group(0) @binding(3) var ink_samp: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var xy = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    return vec4<f32>(xy[vi], 0.0, 1.0);
}

fn cum_at(i: u32) -> f32 {
    return spots.cum[i / 4u][i % 4u];
}

fn mask_at(ix: f32, iy: f32) -> f32 {
    let mx = u32(ix - 64.0 * floor(ix / 64.0));
    let my = u32(iy - 64.0 * floor(iy / 64.0));
    return cum_at(8224u + my * 64u + mx);
}

fn spot_raw(shape: i32, fx: f32, fy: f32, elong: f32) -> f32 {
    var v: f32 = 0.0;
    switch shape {
        case 0: {
            v = fx * fx + fy * fy;
        }
        case 1: {
            let k = max(elong, 1.0);
            let a = fx * k;
            let b = fy / k;
            v = a * a + b * b;
        }
        case 2: {
            v = -(cos(TAU * fx) + cos(TAU * fy));
        }
        case 3: {
            v = abs(fx) + abs(fy);
        }
        case 4: {
            v = max(abs(fx), abs(fy));
        }
        case 5: {
            let a = fx * fx;
            let b = fy * fy;
            v = a * a + b * b;
        }
        default: {
            v = abs(fy);
        }
    }
    return v;
}

fn spot_rank(li: u32, fx: f32, fy: f32) -> f32 {
    let p2 = frame.layers[li].p2;
    let p3 = frame.layers[li].p3;
    let v = spot_raw(i32(p2.y), fx, fy, p2.z);
    let x = clamp((v - p2.w) * p3.x * BINS, 0.0, BINS - 0.001);
    let b = u32(x);
    let f = x - f32(b);
    let base = u32(p3.y) * 1028u + b;
    let c0 = cum_at(base);
    let c1 = cum_at(base + 1u);
    return c0 + f * (c1 - c0);
}

fn dot_gain_comp(t: f32, g0: f32) -> f32 {
    let g = clamp(g0, -0.2, 0.25);
    if abs(g) < 0.00001 {
        return t;
    }
    let a = 1.0 + 4.0 * g;
    let disc = max(a * a - 16.0 * g * t, 0.0);
    return clamp((a - sqrt(disc)) / (8.0 * g), 0.0, 1.0);
}

fn adjust_tone(li: u32, raw: f32) -> f32 {
    let p1 = frame.layers[li].p1;
    var t = clamp(raw, 0.0, 1.0);
    if frame.layers[li].p2.y > 7.5 {
        return t;
    }
    if frame.layers[li].p2.x > 0.5 {
        t = 1.0 - t;
    }
    t = dot_gain_comp(t, p1.x);
    if t <= max(p1.w, 0.0001) {
        return 0.0;
    }
    return min(max(t, p1.y), p1.z);
}

fn ink_at(li: u32, u: f32, v: f32, cell_px: f32) -> f32 {
    let lf = clamp(log2(max(cell_px, 1.0)) - 1.0, 0.0, frame.hdr2.x);
    let plate = i32(frame.layers[li].p3.y);
    return textureSampleLevel(ink_tex, ink_samp, vec2<f32>(u, v), plate, lf).r;
}

fn region_at(li: u32, x: f32, y: f32) -> i32 {
    let im = frame.image;
    if x >= im.x && x < im.x + im.z && y >= im.y && y < im.y + im.w {
        return 0;
    }
    let n = i32(frame.layers[li].p3.z);
    for (var i: i32 = 0; i < n; i = i + 1) {
        let r = frame.layers[li].patch_rect[i];
        if x >= r.x && x < r.x + r.z && y >= r.y && y < r.y + r.w {
            return 1 + i;
        }
    }
    return -1;
}

fn patch_tone(li: u32, region: i32) -> f32 {
    let i = u32(region - 1);
    return frame.layers[li].patch_tone[i / 4u][i % 4u];
}

fn tone_at(li: u32, region: i32, cx: f32, cy: f32) -> f32 {
    if region == 0 {
        let im = frame.image;
        let u = clamp((cx - im.x) / im.z, 0.0, 1.0);
        let v = clamp((cy - im.y) / im.w, 0.0, 1.0);
        return adjust_tone(li, ink_at(li, u, v, frame.layers[li].p0.w));
    }
    return patch_tone(li, region);
}

// 1.0 where the page point (x, y) mm is covered by ink in a hard-threshold rendering.
fn is_on(li: u32, x: f32, y: f32) -> f32 {
    let region = region_at(li, x, y);
    if region < 0 {
        return 0.0;
    }
    if region == 0 && frame.layers[li].p2.y > 7.5 {
        let im = frame.image;
        let dims = vec2<f32>(textureDimensions(ink_tex).xy);
        let uv = clamp(vec2<f32>((x - im.x) / im.z, (y - im.y) / im.w), vec2<f32>(0.0), vec2<f32>(0.99999));
        let px = vec2<i32>(floor(uv * dims));
        return select(0.0, 1.0, textureLoad(ink_tex, px, i32(frame.layers[li].p3.y), 0).r > 0.5);
    }
    let p0 = frame.layers[li].p0;
    let c = p0.x;
    let s = p0.y;
    let pitch = p0.z;
    let u = x * c - y * s;
    let v = x * s + y * c;
    let su = u / pitch;
    let sv = v / pitch;
    let iu = floor(su);
    let iv = floor(sv);
    let cu = (iu + 0.5) * pitch;
    let cv = (iv + 0.5) * pitch;
    let cx = cu * c + cv * s;
    let cy = -cu * s + cv * c;
    let t = tone_at(li, region, cx, cy);
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 0.9999 {
        return 1.0;
    }
    if frame.layers[li].p2.y > 6.5 {
        // FM screen, and the patches of a dithered plate
        if mask_at(iu, iv) <= t {
            return 1.0;
        }
        return 0.0;
    }
    let fx = su - iu - 0.5;
    let fy = sv - iv - 0.5;
    if spot_rank(li, fx, fy) <= t {
        return 1.0;
    }
    return 0.0;
}

// Continuous tone stand-in for the dots when zoomed out too far to resolve them.
fn tone_preview(li: u32, x: f32, y: f32, footprint: f32) -> f32 {
    let region = region_at(li, x, y);
    if region < 0 {
        return 0.0;
    }
    if region == 0 {
        let im = frame.image;
        let u = clamp((x - im.x) / im.z, 0.0, 1.0);
        let v = clamp((y - im.y) / im.w, 0.0, 1.0);
        let fp = max(footprint, frame.layers[li].p0.z) * (frame.hdr2.y / im.z);
        return adjust_tone(li, ink_at(li, u, v, fp));
    }
    return patch_tone(li, region);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px_mm = 1.0 / frame.hdr0.z;
    let x = (pos.x - frame.hdr0.x) * px_mm;
    let y = (pos.y - frame.hdr0.y) * px_mm;
    var col = vec3<f32>(1.0);
    let n = min(u32(frame.hdr0.w + 0.5), MAX_LAYERS);
    for (var li: u32 = 0u; li < n; li = li + 1u) {
        var cov: f32 = 0.0;
        if frame.hdr1.z < 0.5 {
            cov = tone_preview(li, x, y, px_mm);
        } else if frame.hdr1.w > 0.5 {
            let d = px_mm * 0.25;
            cov = 0.25 * (is_on(li, x - d, y - d) + is_on(li, x + d, y - d) + is_on(li, x - d, y + d) + is_on(li, x + d, y + d));
        } else {
            cov = is_on(li, x, y);
        }
        if cov > 0.0 {
            let ink = frame.layers[li].ink.rgb;
            col = col * (vec3<f32>(1.0) - cov * (vec3<f32>(1.0) - ink));
        }
    }
    if frame.hdr2.z > 0.5 {
        col = srgb_to_linear(col);
    }
    return vec4<f32>(col, 1.0);
}
