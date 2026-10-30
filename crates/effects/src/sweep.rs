//! Every effect at its parameter extremes: no panic, no non-finite pixel, no runaway buffer.
//! Slow (every parameter of every effect); run with `cargo test -p effectcraft-effects --release
//! sweep -- --ignored --nocapture`.

use effectcraft_keyframe::Value;
use effectcraft_project::ParamUi;

use crate::{Buf, EffectCtx, EffectEnv, Image, Params, apply, default_value, registry};

/// The values to try for parameter `ui` with default `d`.
fn extremes(ui: &ParamUi, d: &Value) -> Vec<Value> {
    let nums = |extra: &[f64]| {
        let mut v = vec![f64::NAN, f64::INFINITY, -1e9, 1e9, 0.0, -1.0];
        v.extend_from_slice(extra);
        v
    };
    match (ui, d) {
        (ParamUi::Slider { min, max, slider_min, slider_max, .. }, Value::Scalar(_)) => {
            nums(&[*min, *max, *slider_min, *slider_max]).into_iter().map(Value::Scalar).collect()
        }
        (_, Value::Scalar(_)) => nums(&[]).into_iter().map(Value::Scalar).collect(),
        (_, Value::Vec2(_)) => [[f64::NAN, f64::NAN], [1e9, -1e9], [-1e12, 1e12], [0.0, 0.0]].into_iter().map(Value::Vec2).collect(),
        (_, Value::Vec3(_)) => [[f64::NAN; 3], [1e9, -1e9, 1e9], [0.0; 3]].into_iter().map(Value::Vec3).collect(),
        (_, Value::Color(_)) => [[f64::NAN; 4], [1e9, -1e9, 1e9, 1e9], [0.0; 4]].into_iter().map(Value::Color).collect(),
        (ParamUi::Popup { options }, Value::Enum(_)) => {
            let mut v: Vec<Value> = (0..options.len() as u32).map(Value::Enum).collect();
            v.extend([Value::Enum(options.len() as u32), Value::Enum(u32::MAX)]);
            v
        }
        (_, Value::Enum(_)) => vec![Value::Enum(u32::MAX)],
        (_, Value::Bool(_)) => vec![Value::Bool(true), Value::Bool(false)],
        _ => vec![],
    }
}

fn scene(w: u32, h: u32) -> Image {
    let mut img = Image::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            let a = if (x + y) % 7 == 0 { 0.5 } else { 1.0 };
            img.set(x, y, [u * a, v * a, (1.0 - u) * a, a]);
        }
    }
    img
}

/// Problems found running `id` with `param` set to `value` (empty: fine). Each case runs on its
/// own thread so a hang is reported (and left behind) instead of stopping the sweep.
fn check(spec: &'static crate::EffectSpec, param: &str, value: &Value) -> Option<String> {
    let size = [24.0, 16.0];
    let mut params = Params { values: spec.params.iter().map(|p| (p.id.to_string(), default_value(p, size))).collect() };
    params.values.insert(param.to_string(), value.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    let start = std::time::Instant::now();
    std::thread::spawn(move || {
        let ctx = EffectCtx { params: &params, time: 0.5, layer_size: size, seed: 1, adjustment: false, env: EffectEnv::default() };
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| apply(spec, &ctx, Buf { img: scene(24, 16), offset: [0.0, 0.0], scale: 1.0 })));
        let _ = tx.send(out.map_err(|e| e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()));
    });
    let out = match rx.recv_timeout(std::time::Duration::from_secs(20)) {
        Err(_) => return Some("hang: no result after 20 s".into()),
        Ok(Err(msg)) => return Some(format!("panic: {msg}")),
        Ok(Ok(o)) => o,
    };
    let secs = start.elapsed().as_secs_f64();
    let mut problems = vec![];
    if secs > 3.0 {
        problems.push(format!("slow: {secs:.1}s"));
    }
    let px = u64::from(out.img.width) * u64::from(out.img.height);
    if px > 20_000_000 {
        problems.push(format!("buffer {}x{}", out.img.width, out.img.height));
    }
    if out.img.data.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
        problems.push("non-finite pixels".into());
    }
    if !out.offset.iter().all(|o| o.is_finite()) {
        problems.push(format!("offset {:?}", out.offset));
    }
    (!problems.is_empty()).then(|| problems.join(", "))
}

#[test]
#[ignore]
fn sweep_every_effect_at_its_extremes() {
    let only = std::env::var("SWEEP_ONLY").ok();
    let mut found = vec![];
    for spec in registry() {
        if only.as_deref().is_some_and(|o| !spec.id.contains(o)) {
            continue;
        }
        eprintln!("{}", spec.id);
        // A hang leaves its thread running, which would starve the effect's later cases: stop
        // the effect there (run one effect per process with SWEEP_ONLY to sweep the rest).
        'effect: for p in &spec.params {
            for v in extremes(&p.ui, &p.default) {
                if let Some(problem) = check(spec, p.id, &v) {
                    let line = format!("{} {}={:?}: {problem}", spec.id, p.id, v);
                    eprintln!("  !! {line}");
                    found.push(line);
                    if problem.starts_with("hang") {
                        break 'effect;
                    }
                }
            }
        }
    }
    eprintln!("\n{} problems", found.len());
    for f in &found {
        eprintln!("{f}");
    }
}

/// The cases the sweep found that hung, overflowed or turned pixels non-finite at a finite
/// extreme (values an expression can set): each now finishes with finite pixels.
#[test]
fn extreme_values_found_by_the_sweep_stay_safe() {
    let n = |v: f64| Value::Scalar(v);
    let far = Value::Vec2([-1e12, 1e12]);
    let cases: Vec<(&str, &str, Value)> = vec![
        ("ec.blur.cccross", "radiusY", n(1e9)),
        ("ec.blur.reduceflicker", "softness", n(1e9)),
        ("ec.blur.unsharp", "radius", n(1e9)),
        ("ec.generate.vegas", "preBlur", n(1e9)),
        ("ec.blur.cameralens", "blurRadius", n(1e9)),
        ("ec.vr.blur", "blurriness", n(1e9)),
        ("ec.vr.colorgradients", "blend", n(1e9)),
        ("ec.noise.median", "radius", n(1e9)),
        ("ec.noise.fractal", "transform/offset", far.clone()),
        ("ec.noise.turbulent", "transform/offset", far.clone()),
        ("ec.noise.curlnoise", "offset", far.clone()),
        ("ec.distort.turbulentdisplace", "offset", far),
        ("ec.noise.addgrain", "animation/animationSpeed", n(1e9)),
        ("ec.noise.addgrain", "tweaking/softness", n(1e9)),
        ("ec.sim.foam", "producer/producerXSize", n(1e9)),
        ("ec.stylize.ccmrsmoothie", "colorLoop", Value::Enum(u32::MAX)),
        ("ec.color.exposure", "master/exposure", n(1e9)),
        ("ec.color.lumetri", "basicCorrection/tone/exposure", n(1e9)),
        ("ec.sim.ccparticleworld", "particle/maxOpacity", n(1e9)),
        ("ec.sim.ccparticlesystems2", "particle/maxOpacity", n(1e9)),
        ("ec.sim.ccrainfall", "opacity", n(1e9)),
        ("ec.sim.ccsnowfall", "opacity", n(1e9)),
        ("ec.sim.particleplayground", "cannon/cannonColor", Value::Color([1e9, -1e9, 1e9, 1e9])),
    ];
    let mut failed = vec![];
    for (id, param, value) in cases {
        let Some(spec) = registry().iter().find(|s| s.id == id) else {
            failed.push(format!("{id}: not registered"));
            continue;
        };
        // (slowness is the sweep's business; here only hangs, panics and bad pixels count)
        if let Some(problem) = check(spec, param, &value).filter(|p| !p.starts_with("slow")) {
            failed.push(format!("{id} {param}={value:?}: {problem}"));
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

/// The plane filters' huge radii: erosion / dilation allocated `n + 2r` values and the box mean
/// summed `r` values a row before (Minimax, Matte Choker, …); both now cost the row.
#[test]
fn plane_filters_take_any_radius() {
    let src: Vec<f32> = (0..9).map(|i| (i * 7 % 5) as f32).collect();
    let mut dst = vec![0.0; 9];
    for r in [9usize, 1_000_000_000, usize::MAX] {
        crate::util::minmax_row(&src, &mut dst, r, true);
        assert!(dst.iter().all(|v| *v == 4.0), "r {r}: {dst:?}");
        crate::util::minmax_row(&src, &mut dst, r, false);
        assert!(dst.iter().all(|v| *v == 0.0), "r {r}: {dst:?}");
    }
    let p = crate::util::Plane { w: 9, h: 1, data: src.clone() };
    let near = crate::util::box_plane(&p, 1_000_000, 0);
    let huge = crate::util::box_plane(&p, usize::MAX, 0);
    assert!(near.data.iter().chain(&huge.data).all(|v| v.is_finite()));
    // A huge window weighs the repeated edges almost entirely: (first + last) / 2.
    let edges = (src[0] + src[8]) / 2.0;
    assert!(huge.data.iter().all(|v| (v - edges).abs() < 1e-3), "{:?}", huge.data);
    assert!(crate::util::box_radii(1e30, 3).iter().all(|r| *r > 0));
    assert_eq!(crate::util::box_radii(f64::INFINITY, 3), vec![0, 0, 0]);
}
