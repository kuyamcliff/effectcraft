//! Stylize effects, batch 5: VHS, CRT and ASCII Art; and Light Leak (Generate).
//!
//! EffectCraft's own effects: retro looks motion designers usually build from stacks of other
//! effects or buy as plug-ins, and which After Effects doesn't ship. The parameters are ours. The
//! maths is plain image processing: a luma / chroma (YIQ) split with the chroma smeared along
//! the scan line, per-line time-base jitter, a barrel-distorted screen with a phosphor mask,
//! character glyphs chosen by cell brightness, and soft Gaussian light blobs screened over the
//! layer. Every random choice hashes the pixel, the frame and the effect's seed, so a frame
//! always renders the same.

use effectcraft_keyframe::Value;
use effectcraft_project::ParamUi;
use effectcraft_raster::{Image, Px};
use rayon::prelude::*;

use crate::util::{hash1, layer_rect, lerp4, premul, smoothstep, unpremul};
use crate::{Buf, EffectCtx, EffectSpec, col, num, p, popup, slider};

fn spec(id: &'static str, name: &'static str, category: &'static str, params: Vec<crate::ParamSpec>, render: crate::RenderFn) -> EffectSpec {
    EffectSpec { id, name, category, params, render, gpu: false, float: true }
}

fn pct(default: f64) -> (Value, ParamUi) {
    (num(default), slider(0.0, 100.0, 0.0, 100.0, 1))
}

/// A parameter in 0–1 from a percentage; non-finite values count as 0.
fn unit(ctx: &EffectCtx, id: &str) -> f32 {
    let v = ctx.params.f(id) / 100.0;
    if v.is_finite() { v.clamp(0.0, 1.0) as f32 } else { 0.0 }
}

/// A finite parameter within `lo..=hi` (`lo` when not finite).
fn within(ctx: &EffectCtx, id: &str, lo: f64, hi: f64) -> f64 {
    let v = ctx.params.f(id);
    if v.is_finite() { v.clamp(lo, hi) } else { lo }
}

/// Blend With Original: `orig` over the effect result `fx` by `k` (0 = effect only).
fn blend_with(orig: Px, fx: Px, k: f32) -> Px {
    if k <= 0.0 { fx } else { lerp4(fx, orig, k.min(1.0)) }
}

/// Smooth value noise along one axis: hashes at whole `u` blended with a smoothstep, in 0–1.
fn smooth1(u: f64, a: u32, seed: u32) -> f32 {
    let i = u.floor();
    let f = (u - i) as f32;
    let i = i.rem_euclid(1_000_000.0) as u32;
    let t = f * f * (3.0 - 2.0 * f);
    let (h0, h1) = (hash1(i, a, seed), hash1(i.wrapping_add(1), a, seed));
    h0 + (h1 - h0) * t
}

/// The frame number of layer time `t` at 30 fps (what jitter and noise change with), finite.
fn frame_of(t: f64) -> u32 {
    if t.is_finite() { (t * 30.0).floor().rem_euclid(1_000_000.0) as u32 } else { 0 }
}

fn rgb_to_yiq(c: [f32; 3]) -> [f32; 3] {
    [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2], 0.596 * c[0] - 0.274 * c[1] - 0.322 * c[2], 0.211 * c[0] - 0.523 * c[1] + 0.312 * c[2]]
}

fn yiq_to_rgb(c: [f32; 3]) -> [f32; 3] {
    [c[0] + 0.956 * c[1] + 0.621 * c[2], c[0] - 0.272 * c[1] - 0.647 * c[2], c[0] - 1.106 * c[1] + 1.703 * c[2]]
}

// ------------------------------------------------------------------ VHS

/// The most VHS smears the chroma (px at 100 % layer scale).
const MAX_BLEED: f64 = 64.0;

/// VHS's settings for one frame, in buffer pixels.
struct Vhs {
    scale: f64,
    bleed: f64,
    taps: usize,
    jitter: f64,
    tracking: f32,
    noise: f32,
    scan: f32,
    sat: f32,
    blend: f32,
    frame: u32,
    seed: u32,
    /// The tracking band: centre row and half height.
    band: (f64, f64),
    /// The buffer's width (px), for dropout streak lengths.
    width: f64,
    /// Luma softness: half width of the horizontal low-pass (px).
    soft: f64,
    /// Head switching: the layer's bottom row and the height of the torn strip above it (px).
    head: (f64, f64),
}

impl Vhs {
    fn new(ctx: &EffectCtx, b: &Buf) -> Vhs {
        let scale = if b.scale.is_finite() && b.scale > 0.0 { b.scale } else { 1.0 };
        let bleed = within(ctx, "chromaBleed", 0.0, MAX_BLEED) * scale;
        let tracking = unit(ctx, "tracking");
        let speed = within(ctx, "trackingSpeed", -100.0, 100.0);
        let (_, y0, _, lh) = layer_rect(ctx, b);
        let lh = if lh.is_finite() && lh > 1.0 { lh } else { b.img.height.max(1) as f64 };
        // The band rolls down the layer (wrapping) at `speed` tenths of its height a second.
        let roll = if ctx.time.is_finite() { (ctx.time * speed * 0.1).rem_euclid(1.0) } else { 0.0 };
        Vhs {
            scale,
            bleed,
            taps: (bleed.ceil() as usize).clamp(1, 24),
            jitter: within(ctx, "jitter", 0.0, 100.0) * scale,
            tracking,
            noise: unit(ctx, "noise"),
            scan: unit(ctx, "scanlines"),
            sat: (within(ctx, "saturation", 0.0, 400.0) / 100.0) as f32,
            blend: unit(ctx, "blend"),
            frame: frame_of(ctx.time),
            seed: ctx.seed,
            band: (y0 + roll * lh * 1.2 - lh * 0.1, lh * 0.06 * (0.3 + tracking as f64)),
            soft: within(ctx, "softness", 0.0, 32.0) * scale,
            width: b.img.width.max(1) as f64,
            head: (y0 + lh, lh * 0.045 * unit(ctx, "headSwitching") as f64),
        }
    }

    /// Row `y`'s scan line, sideways slip (px) and whether it is torn (tracking band or head
    /// switching).
    fn row(&self, y: usize) -> (u32, f64, bool) {
        let yf = y as f64 + 0.5;
        let line = (yf / self.scale).floor() as i64 as u32;
        // Time-base error: a smooth wobble down the picture that changes every frame, plus a
        // little per-line noise (independent lines alone saw edges into combs).
        let wobble = smooth1(line as f64 / 14.0, self.frame, self.seed ^ 0x3c) as f64 - 0.5;
        let fine = hash1(line, self.frame, self.seed) as f64 - 0.5;
        let mut shift = (wobble * 0.9 + fine * 0.15) * self.jitter;
        let d = ((yf - self.band.0) / self.band.1.max(1e-6)).abs();
        let mut torn = self.tracking > 0.0 && d < 1.0;
        if torn {
            shift += (hash1(line, self.frame, self.seed ^ 0x9e37) as f64 - 0.3) * 60.0 * self.scale * self.tracking as f64 * (1.0 - d);
        }
        // Head switching: the last lines of the field bend right, more towards the bottom.
        let up = self.head.0 - yf;
        if self.head.1 > 0.0 && (0.0..self.head.1).contains(&up) {
            let k = 1.0 - up / self.head.1;
            shift += k * k * 24.0 * self.scale * (0.7 + 0.6 * hash1(line, self.frame, self.seed ^ 0x77) as f64);
            torn |= k > 0.6;
        }
        (line, shift, torn)
    }

    /// Brightness low-passed along the line (a tent over `soft` px each side).
    fn luma(&self, src: &Image, x: f64, y: f64) -> (f32, f32) {
        if self.soft < 0.25 {
            let (c, a) = unpremul(src.sample_bilinear_clamped(x, y));
            return (rgb_to_yiq(c)[0], a);
        }
        let (mut l, mut a, mut wsum) = (0.0f32, 0.0f32, 0.0f32);
        for t in -2i32..=2 {
            let w = 3.0 - t.abs() as f32;
            let p = src.sample_bilinear_clamped(x + self.soft * t as f64 / 2.0, y);
            let (c, pa) = unpremul(p);
            l += rgb_to_yiq(c)[0] * w;
            a += pa * w;
            wsum += w;
        }
        (l / wsum, a / wsum)
    }

    /// The chroma (I, Q): averaged over the `bleed` pixels left of `x`, on this line and the
    /// one above (a VCR's chroma delay line), so colour smears right of edges and softens
    /// vertically.
    fn chroma(&self, src: &Image, x: f64, y: f64) -> (f32, f32) {
        let (mut i, mut q) = (0.0f32, 0.0f32);
        for row in [y, y - self.scale] {
            for t in 0..self.taps {
                let at = x - self.bleed * (t as f64 + 0.5) / self.taps as f64;
                let iq = rgb_to_yiq(unpremul(src.sample_bilinear_clamped(at, row)).0);
                i += iq[1];
                q += iq[2];
            }
        }
        let n = 2.0 * self.taps as f32;
        (i / n * self.sat, q / n * self.sat)
    }

    /// A dropout on `line` at `x`: a short bright streak where the tape lost its signal, on a
    /// `chance` of the lines (more in a torn band). Brightness to add.
    fn dropout(&self, x: usize, line: u32, chance: f32) -> f32 {
        if chance <= 0.0 || hash1(line, self.frame, self.seed ^ 0xd0) >= chance {
            return 0.0;
        }
        let start = hash1(line, self.frame, self.seed ^ 0xd1) as f64 * 1.2 - 0.1;
        let len = (0.02 + 0.1 * hash1(line, self.frame, self.seed ^ 0xd2) as f64) * self.width;
        let u = (x as f64 + 0.5 - start * self.width) / len.max(1.0);
        if (0.0..1.0).contains(&u) { (1.0 - u as f32) * 0.9 } else { 0.0 }
    }

    fn pixel(&self, src: &Image, x: usize, y: usize, (line, shift, torn): (u32, f64, bool)) -> Px {
        let (xf, yf) = (x as f64 + 0.5 - shift, y as f64 + 0.5);
        let (c, _) = unpremul(src.sample_bilinear_clamped(xf, yf));
        let (mut luma, a) = self.luma(src, xf, yf);
        if a <= 0.0 {
            return [0.0; 4];
        }
        let (i, q) = self.chroma(src, xf, yf);
        if self.noise > 0.0 {
            // Tape noise is stretched along the line.
            let n = smooth1((x as f64 + 0.5) / (3.0 * self.scale), line.wrapping_mul(7919) ^ self.frame, self.seed) - 0.5;
            luma += n * self.noise * 0.6;
        }
        luma += self.dropout(x, line, if torn { 0.12 + 0.3 * self.tracking } else { 0.004 * self.noise });
        let k = if (line / 2).is_multiple_of(2) { 1.0 } else { 1.0 - 0.6 * self.scan };
        let rgb = yiq_to_rgb([luma * k, i * k, q * k]);
        // Never brighter than the source's brightest channel or white (HDR highlights keep theirs).
        let top = c.iter().fold(1.0f32, |m, v| m.max(*v));
        let rgb = rgb.map(|v| v.clamp(0.0, top));
        blend_with(src.get(x as i64, y as i64), premul(rgb, a), self.blend)
    }
}

/// VHS: worn videotape. Each scan line slips sideways a little (time-base jitter) and a band of
/// torn lines rolls down the picture (Tracking Error); the colour is carried at a far lower
/// resolution than the brightness, so it smears to the right of edges (Chroma Bleed); tape noise
/// speckles the brightness and alternate lines darken (Scanlines).
fn vhs(ctx: &EffectCtx, mut b: Buf) -> Buf {
    let v = Vhs::new(ctx, &b);
    let src = b.img.clone();
    b.img.rows_mut().for_each(|(y, row)| {
        let r = v.row(y);
        for (x, o) in row.iter_mut().enumerate() {
            *o = v.pixel(&src, x, y, r);
        }
    });
    b
}

// ------------------------------------------------------------------ CRT

/// How far inside a rounded rectangle (−1…1 both ways, corner radius `r` of those units) the
/// point (`u`, `v`) is, as coverage 0–1 over about a pixel (`half` px per unit).
fn rounded_inside(u: f64, v: f64, r: f64, half: f64) -> f32 {
    // Signed distance to the rounded box (negative inside).
    let (qx, qy) = (u.abs() - (1.0 - r), v.abs() - (1.0 - r));
    let d = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0) - r;
    (-d * half + 0.5).clamp(0.0, 1.0) as f32
}

/// CRT's settings, in buffer pixels.
struct Crt {
    scale: f64,
    curve: f64,
    scan: f32,
    period: f64,
    mask: u32,
    mask_k: f32,
    vignette: f32,
    boost: f32,
    blend: f32,
    /// Layer top, centre and half size.
    top: f64,
    centre: (f64, f64),
    half: (f64, f64),
    triad: usize,
}

impl Crt {
    fn new(ctx: &EffectCtx, b: &Buf) -> Crt {
        let scale = if b.scale.is_finite() && b.scale > 0.0 { b.scale } else { 1.0 };
        let (x0, y0, lw, lh) = layer_rect(ctx, b);
        let (lw, lh) = (lw.max(1.0), lh.max(1.0));
        Crt {
            scale,
            curve: unit(ctx, "curvature") as f64 * 0.35,
            scan: unit(ctx, "scanlineStrength"),
            period: within(ctx, "scanlineSize", 1.0, 1000.0) * scale,
            mask: ctx.params.e("maskType").min(2),
            mask_k: unit(ctx, "maskStrength"),
            vignette: unit(ctx, "vignette"),
            boost: 1.0 + (within(ctx, "brightness", 0.0, 400.0) / 100.0) as f32,
            blend: unit(ctx, "blend"),
            top: y0,
            centre: (x0 + lw * 0.5, y0 + lh * 0.5),
            half: (lw * 0.5, lh * 0.5),
            triad: scale.round().max(1.0) as usize,
        }
    }

    /// The phosphor mask's gain per channel, scaled so the mask passes as much light on average
    /// as no mask (a real tube is driven harder to make up for it) at buffer pixel (`x`, `y`).
    fn mask_at(&self, x: usize, y: f64) -> [f32; 3] {
        let mut m = [1.0f32; 3];
        if self.mask >= 2 || self.mask_k <= 0.0 {
            return m;
        }
        let dim = 1.0 - self.mask_k * 0.7;
        let gain = 3.0 / (1.0 + 2.0 * dim);
        // Shadow Mask: every other row of triads shifts by one stripe.
        let row_shift = usize::from(self.mask == 1 && ((y / (3.0 * self.scale)).floor() as i64).rem_euclid(2) == 1);
        let lit = (x / self.triad + row_shift) % 3;
        for (k, mk) in m.iter_mut().enumerate() {
            *mk = if k == lit { gain } else { dim * gain };
        }
        m
    }

    fn pixel(&self, src: &Image, x: usize, y: usize) -> Px {
        let (xf, yf) = (x as f64 + 0.5, y as f64 + 0.5);
        // Screen position in −1…1 across the layer, bulged outwards.
        let (u, v) = ((xf - self.centre.0) / self.half.0, (yf - self.centre.1) / self.half.1);
        let (u, v) = (u * (1.0 + self.curve * v * v), v * (1.0 + self.curve * u * u));
        if !(u.abs() <= 1.0 && v.abs() <= 1.0) {
            return [0.0; 4];
        }
        let sy = self.centre.1 + v * self.half.1;
        let (c, a) = unpremul(src.sample_bilinear_clamped(self.centre.0 + u * self.half.0, sy));
        if a <= 0.0 {
            return [0.0; 4];
        }
        // Scan lines: a dark gap between lines, `period` pixels apart in the picture. A bright
        // line's beam is wider, so it fills more of the gap (bloom).
        let lum = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).clamp(0.0, 1.0);
        let phase = (sy - self.top) / self.period;
        let gap = 0.5 + 0.5 * (phase * std::f64::consts::TAU).cos() as f32;
        let line = (1.0 - self.scan * gap * (1.0 - 0.6 * lum)) / (1.0 - 0.3 * self.scan);
        let r2 = (u * u + v * v) as f32;
        let k = line * (1.0 - self.vignette * (r2 * 0.5).powf(1.5).min(1.0)) * self.boost;
        let m = self.mask_at(x, yf);
        let rgb = [c[0] * m[0] * k, c[1] * m[1] * k, c[2] * m[2] * k];
        // The tube's rounded corners, anti-aliased over about a pixel.
        let edge = rounded_inside(u, v, 0.08, self.half.0.min(self.half.1));
        blend_with(src.get(x as i64, y as i64), premul(rgb, a * edge), self.blend)
    }
}

/// CRT: the layer on a tube television. The glass bulges (Curvature; the corners curl away to
/// transparency), the picture is drawn in horizontal scan lines, each pixel lights red, green and
/// blue phosphor stripes (Aperture Grille) or dots (Shadow Mask), the edges fall off
/// (Vignette) and the tube glows a little brighter to make up for its dark gaps.
fn crt(ctx: &EffectCtx, mut b: Buf) -> Buf {
    let c = Crt::new(ctx, &b);
    let src = b.img.clone();
    b.img.rows_mut().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = c.pixel(&src, x, y);
        }
    });
    b
}

// ------------------------------------------------------------------ ASCII Art

/// 5 × 7 glyphs (one row per byte, bit 4 the left column), from empty to full.
const RAMP: [[u8; 7]; 10] = [
    [0, 0, 0, 0, 0, 0, 0],                                           // space
    [0, 0, 0, 0, 0, 0, 0b00100],                                     // .
    [0, 0, 0b00100, 0, 0, 0b00100, 0],                               // :
    [0, 0, 0, 0b01110, 0, 0, 0],                                     // -
    [0, 0, 0b11111, 0, 0b11111, 0, 0],                               // =
    [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],             // +
    [0, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0],             // *
    [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0],       // #
    [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011], // %
    [0b01110, 0b10001, 0b10111, 0b10101, 0b10111, 0b10000, 0b01111], // @
];

/// 0 and 1.
const BINARY: [[u8; 7]; 2] = [[0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110], [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110]];

/// Whether the glyph for brightness `v` (0–1) in character set `set` covers glyph-grid point
/// (`gx` of 5, `gy` of 7).
fn glyph_on(set: u32, v: f32, gx: usize, gy: usize) -> bool {
    let v = v.clamp(0.0, 1.0);
    let bit = |g: &[u8; 7]| g.get(gy).is_some_and(|r| (r >> (4 - gx.min(4))) & 1 == 1);
    match set {
        // Blocks: an ordered-dither fill (empty, ░, ▒, ▓, █).
        1 => {
            const B: [[f32; 2]; 2] = [[0.125, 0.625], [0.875, 0.375]];
            let level = (v * 4.0).round() / 4.0;
            level > B[gy % 2][gx % 2]
        }
        2 => bit(&BINARY[usize::from(v >= 0.5)]),
        _ => bit(&RAMP[((v * (RAMP.len() - 1) as f32).round() as usize).min(RAMP.len() - 1)]),
    }
}

/// The average (premultiplied) colour of the `cw` × `ch` cell at (`x0`, `y0`).
fn cell_average(src: &Image, x0: usize, y0: usize, (cw, ch): (usize, usize)) -> Px {
    let (w, h) = (src.width as usize, src.height as usize);
    let mut sum = [0.0f32; 4];
    let mut n = 0.0f32;
    for y in y0..(y0 + ch).min(h) {
        for x in x0..(x0 + cw).min(w) {
            let p = src.get(x as i64, y as i64);
            for k in 0..4 {
                sum[k] += p[k];
            }
            n += 1.0;
        }
    }
    sum.map(|v| v / n.max(1.0))
}

/// ASCII Art: the layer redrawn in characters. Each cell's brightness picks a character from a
/// ramp (Standard), a block shade (Blocks) or 0 / 1 (Binary); the characters take the cell's
/// colour (Original Colors) or the Text Color, over the Background Color.
fn ascii(ctx: &EffectCtx, mut b: Buf) -> Buf {
    let s = if b.scale.is_finite() && b.scale > 0.0 { b.scale } else { 1.0 };
    // Character cells are 3:4 (Cell Size is their width), like a terminal's.
    let size = within(ctx, "cellSize", 2.0, 512.0) * s;
    let (cw, ch) = (size.round().max(1.0) as usize, (size * 4.0 / 3.0).round().max(1.0) as usize);
    let set = ctx.params.e("charset").min(2);
    let mono = ctx.params.e("colorMode") == 1;
    // Between the characters: the Background Color, or the layer's cells darkened.
    let dim_layer = ctx.params.e("backgroundMode") == 1;
    let invert = ctx.params.b("invert");
    let text = ctx.params.color("textColor");
    let bg = ctx.params.color("backgroundColor");
    let finite = |c: [f32; 4]| c.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 });
    let (text, bg) = (finite(text), finite(bg));
    let blend = unit(ctx, "blend");
    let (w, h) = (b.img.width as usize, b.img.height as usize);
    let (gw, gh) = (w.div_ceil(cw), h.div_ceil(ch));
    let src = &b.img;
    let cells: Vec<Px> = (0..gw * gh).into_par_iter().map(|i| cell_average(src, (i % gw) * cw, (i / gw) * ch, (cw, ch))).collect();
    let orig = b.img.clone();
    b.img.rows_mut().for_each(|(y, row)| {
        // Each 5 × 7 glyph sits in a 6 × 8 grid: a column and a row of spacing keep neighbouring
        // characters apart.
        let (cy, gy) = (y / ch, ((y % ch) * 8) / ch);
        for (x, o) in row.iter_mut().enumerate() {
            let a = orig.get(x as i64, y as i64)[3];
            if a <= 0.0 {
                continue;
            }
            let (cx, gx) = (x / cw, ((x % cw) * 6) / cw);
            let c = cells.get(cy * gw + cx).copied().unwrap_or_default();
            let (rgb, ca) = unpremul(c);
            let mut lum = if ca > 0.0 { 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] } else { 0.0 };
            if invert {
                lum = 1.0 - lum;
            }
            // The characters' density shows the brightness, so they carry the cell's hue at
            // full value (its own colour would count the brightness twice).
            let peak = rgb.iter().fold(0.0f32, |m, v| m.max(*v)).max(0.2);
            let ink = if mono { [text[0], text[1], text[2]] } else { rgb.map(|v| (v / peak).clamp(0.0, 1.0)) };
            let back = if dim_layer { rgb.map(|v| v.clamp(0.0, 1.0) * 0.22) } else { [bg[0], bg[1], bg[2]] };
            let out = if gx < 5 && gy < 7 && glyph_on(set, lum, gx, gy) { ink } else { back };
            *o = blend_with(*o, premul(out, a), blend);
        }
    });
    b
}

// ------------------------------------------------------------------ Light Leak

/// The light of `blobs` (centre x, y, radius, weight) at buffer point (`x`, `y`), and where its
/// colour lies between Color A (0) and Color B (1).
fn leak_at(blobs: &[(f64, f64, f64, f32)], x: f64, y: f64) -> (f32, f32) {
    let (mut v, mut t) = (0.0f32, 0.0f32);
    for (k, (bx, by, r, w)) in blobs.iter().enumerate() {
        let d2 = ((x - bx).powi(2) + (y - by).powi(2)) / (r * r);
        let g = (-d2 * 2.0).exp() as f32 * w;
        v += g;
        t += g * k as f32 / 2.0;
    }
    (v, if v > 0.0 { (t / v).clamp(0.0, 1.0) } else { 0.0 })
}

/// Light Leak: soft, warm light washing over the footage, as when light reaches film through a
/// gap in the camera. Three blobs drift around Position as Evolution turns; their colour runs
/// from Color A to Color B and they are screened (or added) over the layer, never outside it.
fn light_leak(ctx: &EffectCtx, mut b: Buf) -> Buf {
    let intensity = (within(ctx, "intensity", 0.0, 400.0) / 100.0) as f32;
    if intensity <= 0.0 {
        return b;
    }
    let (x0, y0, lw, lh) = layer_rect(ctx, &b);
    let diag = (lw * lw + lh * lh).sqrt().max(1.0);
    let size = within(ctx, "size", 1.0, 1000.0) / 100.0 * diag * 0.5;
    let pos = ctx.params.v2("position");
    let pos = if pos.iter().all(|v| v.is_finite()) { pos } else { [0.0, 0.0] };
    let (px, py) = b.to_px(pos);
    let (px, py) = if px.is_finite() && py.is_finite() { (px, py) } else { (x0, y0) };
    let evo = ctx.params.f("evolution").to_radians();
    let evo = if evo.is_finite() { evo.rem_euclid(std::f64::consts::TAU * 1e6) } else { 0.0 };
    let seed = within(ctx, "seed", 0.0, 1e6) as u32;
    let ca = ctx.params.color("colorA");
    let cb = ctx.params.color("colorB");
    let finite = |c: [f32; 4]| c.map(|v| if v.is_finite() { v.clamp(0.0, 4.0) } else { 0.0 });
    let (ca, cb) = (finite(ca), finite(cb));
    let add = ctx.params.e("mode") == 1;
    // Three blobs: offset, radius and drift phase from the seed.
    let blobs: Vec<(f64, f64, f64, f32)> = (0..3u32)
        .map(|k| {
            let h = |salt: u32| hash1(k, seed, salt) as f64;
            let ang = h(1) * std::f64::consts::TAU + evo * (0.6 + h(2));
            let dist = size * (0.2 + 0.6 * h(3));
            let r = size * (0.5 + 0.7 * h(4));
            (px + ang.cos() * dist, py + ang.sin() * dist * 0.6, r.max(1e-3), (0.5 + 0.5 * h(5)) as f32)
        })
        .collect();
    b.img.rows_mut().for_each(|(y, row)| {
        let yf = y as f64 + 0.5;
        for (x, o) in row.iter_mut().enumerate() {
            let (c, a) = unpremul(*o);
            if a <= 0.0 {
                continue;
            }
            let (v, t) = leak_at(&blobs, x as f64 + 0.5, yf);
            if v <= 1e-6 {
                continue;
            }
            let k = (v * intensity).min(4.0);
            // The core burns towards white, as overexposed film does.
            let hot = smoothstep(0.8, 1.6, k) * 0.85;
            let light: [f32; 3] = std::array::from_fn(|i| (ca[i] + (cb[i] - ca[i]) * t + (1.0 - ca[i] - (cb[i] - ca[i]) * t) * hot) * k);
            let out: [f32; 3] =
                std::array::from_fn(|i| if add { c[i] + light[i] } else { 1.0 - (1.0 - c[i].clamp(0.0, 1.0)) * (1.0 - light[i].clamp(0.0, 1.0)) });
            *o = premul(out, a);
        }
    });
    b
}

pub fn specs() -> Vec<EffectSpec> {
    let (blend0, pct_ui) = pct(0.0);
    vec![
        spec(
            "ec.stylize.vhs",
            "VHS",
            "Stylize",
            vec![
                p("chromaBleed", "Chroma Bleed", num(8.0), slider(0.0, MAX_BLEED, 0.0, 32.0, 1)),
                p("softness", "Softness", num(1.5), slider(0.0, 32.0, 0.0, 8.0, 1)),
                p("jitter", "Line Jitter", num(4.0), slider(0.0, 100.0, 0.0, 20.0, 1)),
                p("tracking", "Tracking Error", num(35.0), pct_ui.clone()),
                p("trackingSpeed", "Tracking Speed", num(1.5), slider(-100.0, 100.0, -10.0, 10.0, 2)),
                p("noise", "Tape Noise", num(15.0), pct_ui.clone()),
                p("headSwitching", "Head Switching", num(60.0), pct_ui.clone()),
                p("scanlines", "Scanlines", num(8.0), pct_ui.clone()),
                p("saturation", "Saturation", num(85.0), slider(0.0, 400.0, 0.0, 200.0, 0)),
                p("blend", "Blend With Original", blend0.clone(), pct_ui.clone()),
            ],
            vhs,
        ),
        spec(
            "ec.stylize.crt",
            "CRT",
            "Stylize",
            vec![
                p("curvature", "Curvature", num(40.0), pct_ui.clone()),
                p("scanlineStrength", "Scanline Strength", num(50.0), pct_ui.clone()),
                p("scanlineSize", "Scanline Size", num(3.0), slider(1.0, 1000.0, 1.0, 16.0, 1)),
                p("maskType", "Phosphor Mask", Value::Enum(0), popup(&["Aperture Grille", "Shadow Mask", "None"])),
                p("maskStrength", "Mask Strength", num(40.0), pct_ui.clone()),
                p("vignette", "Vignette", num(40.0), pct_ui.clone()),
                p("brightness", "Brightness Boost", num(20.0), slider(0.0, 400.0, 0.0, 100.0, 0)),
                p("blend", "Blend With Original", blend0.clone(), pct_ui.clone()),
            ],
            crt,
        ),
        spec(
            "ec.stylize.ascii",
            "ASCII Art",
            "Stylize",
            vec![
                p("cellSize", "Cell Size", num(8.0), slider(2.0, 512.0, 4.0, 64.0, 0)),
                p("charset", "Characters", Value::Enum(0), popup(&["Standard", "Blocks", "Binary"])),
                p("colorMode", "Color", Value::Enum(0), popup(&["Original Colors", "Text Color"])),
                p("textColor", "Text Color", col(0.2, 1.0, 0.3), ParamUi::Color),
                p("backgroundMode", "Background", Value::Enum(1), popup(&["Background Color", "Darkened Layer"])),
                p("backgroundColor", "Background Color", col(0.0, 0.0, 0.0), ParamUi::Color),
                p("invert", "Invert", Value::Bool(false), ParamUi::Checkbox),
                p("blend", "Blend With Original", blend0, pct_ui),
            ],
            ascii,
        ),
        spec(
            "ec.generate.lightleak",
            "Light Leak",
            "Generate",
            vec![
                p("position", "Position", Value::Vec2([-0.05, 0.35]), ParamUi::Point),
                p("size", "Size", num(80.0), slider(1.0, 1000.0, 10.0, 200.0, 0)),
                p("evolution", "Evolution", num(0.0), ParamUi::Angle),
                p("colorA", "Color A", col(1.0, 0.62, 0.22), ParamUi::Color),
                p("colorB", "Color B", col(1.0, 0.3, 0.12), ParamUi::Color),
                p("intensity", "Intensity", num(70.0), slider(0.0, 400.0, 0.0, 200.0, 0)),
                p("mode", "Blending Mode", Value::Enum(0), popup(&["Screen", "Add"])),
                p("seed", "Random Seed", num(0.0), slider(0.0, 1e6, 0.0, 100.0, 0)),
            ],
            light_leak,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EffectEnv, run_fx};
    use effectcraft_raster::Image;

    /// A horizontal grey ramp with a red square in the middle, fully opaque.
    fn scene(w: u32, h: u32) -> Image {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = x as f32 / (w - 1) as f32;
                let red = (w / 3..2 * w / 3).contains(&x) && (h / 3..2 * h / 3).contains(&y);
                img.set(x, y, if red { [0.9, 0.1, 0.1, 1.0] } else { [v, v, v, 1.0] });
            }
        }
        img
    }

    fn run(id: &str, vals: &[(&str, Value)], img: Image) -> Buf {
        run_fx(id, vals, img, 0.0, EffectEnv::default())
    }

    const IDS: [&str; 4] = ["ec.stylize.vhs", "ec.stylize.crt", "ec.stylize.ascii", "ec.generate.lightleak"];

    #[test]
    fn new_effects_are_deterministic_and_keep_the_layer() {
        for id in IDS {
            let a = run(id, &[], scene(48, 32));
            let b = run(id, &[], scene(48, 32));
            assert_eq!(a.img, b.img, "{id}");
            assert_eq!((a.img.width, a.img.height, a.offset), (48, 32, [0.0, 0.0]), "{id} keeps the layer's size");
            assert_ne!(a.img, scene(48, 32), "{id} changes the image with its defaults");
            // Premultiplied colour never exceeds alpha (Light Leak's Add may go overbright).
            if id != "ec.generate.lightleak" {
                assert!(a.img.data.iter().all(|p| p[..3].iter().all(|c| *c <= p[3] * 1.0001 + 1e-5) || p[3] == 0.0 || id == "ec.stylize.crt"), "{id}");
            }
        }
        // Transparent pixels stay transparent.
        let clear = Image::new(16, 12);
        for id in IDS {
            assert!(run(id, &[], clear.clone()).img.data.iter().all(|p| p[3] == 0.0), "{id}");
        }
    }

    #[test]
    fn vhs_smears_colour_but_not_brightness() {
        // A red square on grey, without jitter, noise, tracking or scanlines: red bleeds to the
        // right of the square's right edge; brightness stays put.
        let quiet =
            [("jitter", num(0.0)), ("tracking", num(0.0)), ("noise", num(0.0)), ("scanlines", num(0.0)), ("saturation", num(100.0)), ("chromaBleed", num(8.0))];
        let mut img = Image::filled(40, 8, [0.5, 0.5, 0.5, 1.0]);
        for x in 10..20 {
            for y in 0..8 {
                img.set(x, y, [0.8, 0.1, 0.1, 1.0]);
            }
        }
        let out = run("ec.stylize.vhs", &quiet, img.clone());
        let right = out.img.get(22, 4);
        let left = out.img.get(8, 4);
        assert!(right[0] > right[1] + 0.05, "red trails to the right: {right:?}");
        assert!((left[0] - left[1]).abs() < 0.02, "nothing leaks to the left: {left:?}");
        let y = |p: Px| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        assert!((y(right) - 0.5).abs() < 0.03, "brightness stays: {}", y(right));
        // Time changes the jitter and noise; the same time renders the same.
        let at = |t: f64| run_fx("ec.stylize.vhs", &[], scene(48, 32), t, EffectEnv::default()).img;
        assert_ne!(at(0.0), at(0.5));
        assert_eq!(at(0.5), at(0.5));
    }

    #[test]
    fn crt_bends_the_corners_away_and_draws_scan_lines() {
        let white = Image::filled(60, 40, [1.0; 4]);
        let out = run("ec.stylize.crt", &[("vignette", num(0.0)), ("maskStrength", num(0.0)), ("brightness", num(0.0))], white.clone());
        assert_eq!(out.img.get(0, 0)[3], 0.0, "the corner curls away");
        assert!(out.img.get(30, 20)[3] > 0.99, "the centre stays");
        // Scan lines: rows alternate brighter and darker with a 3 px period; a bright picture's
        // wider beam fills more of the gap (bloom), so they show less on white than on grey.
        let contrast = |img: &Image| {
            let col: Vec<f32> = (14..20).map(|y| img.get(30, y)[1]).collect();
            let (lo, hi) = col.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
            (hi - lo) / hi
        };
        let quiet = [("vignette", num(0.0)), ("maskStrength", num(0.0)), ("brightness", num(0.0))];
        let grey = run("ec.stylize.crt", &quiet, Image::filled(60, 40, [0.4, 0.4, 0.4, 1.0]));
        assert!(contrast(&grey.img) > 0.2, "{}", contrast(&grey.img));
        assert!(contrast(&out.img) < contrast(&grey.img) && contrast(&out.img) > 0.1, "{}", contrast(&out.img));
        // No curvature keeps the corners.
        let flat = run("ec.stylize.crt", &[("curvature", num(0.0))], white.clone());
        assert!(flat.img.get(0, 0)[3] > 0.0);
        // Aperture grille: neighbouring columns light different channels.
        let grille =
            run("ec.stylize.crt", &[("curvature", num(0.0)), ("scanlineStrength", num(0.0)), ("vignette", num(0.0)), ("maskStrength", num(100.0))], white);
        let (a, b) = (grille.img.get(30, 20), grille.img.get(31, 20));
        assert!(a != b && a.iter().take(3).any(|v| *v > 0.9), "{a:?} {b:?}");
    }

    #[test]
    fn ascii_draws_brighter_cells_with_denser_characters() {
        let ink = |v: f32, set: u32| {
            let out = run(
                "ec.stylize.ascii",
                &[("cellSize", num(10.0)), ("charset", Value::Enum(set)), ("colorMode", Value::Enum(1))],
                Image::filled(40, 40, [v, v, v, 1.0]),
            );
            // Fraction of text-colour pixels.
            out.img.data.iter().filter(|p| p[1] > 0.5).count() as f32 / out.img.data.len() as f32
        };
        for set in [0, 1] {
            let (dark, mid, bright) = (ink(0.0, set), ink(0.5, set), ink(1.0, set));
            assert!(dark < mid && mid < bright, "set {set}: {dark} {mid} {bright}");
        }
        assert_eq!(ink(0.0, 0), 0.0, "black is spaces");
        // Binary: two different glyphs for dark and bright.
        assert!(ink(0.1, 2) != ink(0.9, 2));
        // Original Colors keeps the cell's hue.
        let red = run("ec.stylize.ascii", &[("cellSize", num(10.0))], Image::filled(40, 40, [1.0, 0.0, 0.0, 1.0]));
        assert!(red.img.data.iter().any(|p| p[0] > 0.9 && p[1] < 0.1));
    }

    #[test]
    fn light_leak_brightens_near_its_position_only_inside_the_layer() {
        let grey = Image::filled(80, 60, [0.3, 0.3, 0.3, 1.0]);
        let out = run("ec.generate.lightleak", &[("position", Value::Vec2([10.0, 10.0])), ("size", num(30.0))], grey.clone());
        let near = out.img.get(10, 10);
        let far = out.img.get(79, 59);
        assert!(near[0] > 0.45 && near[0] > near[2], "warm and bright near the leak: {near:?}");
        assert!(far[0] < near[0], "{far:?}");
        assert!(out.img.data.iter().all(|p| (p[3] - 1.0).abs() < 1e-6));
        // Zero intensity changes nothing; Evolution moves the light.
        assert_eq!(run("ec.generate.lightleak", &[("intensity", num(0.0))], grey.clone()).img, grey);
        let evo = |d: f64| run("ec.generate.lightleak", &[("evolution", num(d))], grey.clone()).img;
        assert_ne!(evo(0.0), evo(90.0));
    }

    #[test]
    fn extreme_parameters_are_safe() {
        let nan = f64::NAN;
        let cases: [(&str, &[(&str, Value)]); 8] = [
            ("ec.stylize.vhs", &[("chromaBleed", num(1e9)), ("jitter", num(nan)), ("trackingSpeed", num(1e300)), ("saturation", num(-5.0))]),
            ("ec.stylize.vhs", &[("tracking", num(1e9)), ("noise", num(-1e9)), ("scanlines", num(nan))]),
            ("ec.stylize.crt", &[("curvature", num(1e9)), ("scanlineSize", num(0.0)), ("maskType", Value::Enum(u32::MAX))]),
            ("ec.stylize.crt", &[("scanlineSize", num(nan)), ("vignette", num(nan)), ("brightness", num(1e12))]),
            ("ec.stylize.ascii", &[("cellSize", num(1e9)), ("charset", Value::Enum(77)), ("textColor", Value::Color([nan; 4]))]),
            ("ec.stylize.ascii", &[("cellSize", num(-3.0)), ("colorMode", Value::Enum(1)), ("invert", Value::Bool(true))]),
            ("ec.generate.lightleak", &[("position", Value::Vec2([nan, 1e300])), ("size", num(1e12)), ("evolution", num(1e300))]),
            (
                "ec.generate.lightleak",
                &[("intensity", num(1e9)), ("seed", num(nan)), ("colorA", Value::Color([1e9, -1e9, nan, 1.0])), ("mode", Value::Enum(1))],
            ),
        ];
        for (id, vals) in cases {
            let out = run(id, vals, scene(20, 12));
            assert_eq!((out.img.width, out.img.height), (20, 12), "{id} {vals:?}");
            assert!(out.img.data.iter().all(|p| p.iter().all(|c| c.is_finite()) && (0.0..=1.0 + 1e-4).contains(&p[3])), "{id} {vals:?}");
        }
        // Odd times too.
        for t in [f64::NAN, 1e18, -5.0] {
            let out = run_fx("ec.stylize.vhs", &[], scene(20, 12), t, EffectEnv::default());
            assert!(out.img.data.iter().all(|p| p.iter().all(|c| c.is_finite())));
        }
    }
}
