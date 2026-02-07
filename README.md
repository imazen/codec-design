# Image Codec API Design Guidelines

Best practices for designing image encoder and decoder APIs in Rust, distilled from
production codecs including [zenjpeg](https://github.com/imazen/zenjpeg) and
[zenwebp](https://github.com/imazen/zenwebp).

## Core Principles

1. **Config-first, dimensions-second** — configuration should be reusable across images
2. **Required parameters via constructors, optionals via builders** — no `Default` for configs with required semantics
3. **Three-layer architecture** — Config → Request → Encoder/Decoder
4. **Resource estimation before work** — memory and compute cost prediction
5. **Cooperative cancellation** — long operations should be interruptible
6. **Layered complexity** — simple things simple, complex things possible
7. **One obvious way** — prefer a single correct path over multiple equivalent entry points
8. **Forwards-compatible surface** — implementations will change; minimize API surface so internals can evolve without breaking callers

Principles 7 and 8 work together: every public function is a commitment. Duplicate
entry points that do the same thing (e.g., `encode()` and `encode_image()` and
`encode_pixels()`) create maintenance burden, confuse callers, and constrain future
changes. If two methods differ only in convenience, keep the general one and add a
single shortcut at most. When in doubt, leave it out — it's easy to add API surface
later, impossible to remove it.

**Avoid free functions.** Functionality belongs on types (Config, Request, Decoder),
not as module-level `decode_rgba()` / `decode_rgb()` / `decode_bgra()` functions that
proliferate with every new pixel format. A single method on a type with a layout
parameter scales better than N free functions. Free functions also can't accept
limits, cancellation, or config — callers outgrow them immediately and must learn the
real API anyway.

---

## Three-Layer Architecture

Every codec should follow this layering:

```
Layer 1: EncoderConfig     — HOW to encode. Reusable, Clone, no borrows.
                             Quality, effort, subsampling, progressive, etc.

Layer 2: EncodeRequest<'a> — WHAT to encode and WHERE. Borrows everything.
                             Pixel format, dimensions, metadata, gain maps,
                             auxiliary streams, limits, stop token.
                             This is the contract the encoder optimizes against.

Layer 3: Encoder           — Internal state machine. For streaming only.
                             Created by request.build(). Caller pushes rows/frames.
                             One-shot callers never see this — request.encode() hides it.
```

Same pattern for decode:

```
Layer 1: DecoderConfig     — HOW to decode. Upsampling, color management.
Layer 2: DecodeRequest<'a> — WHAT to decode and desired output format.
                             Knowing output format upfront lets decoder skip conversions.
Layer 3: Decoder           — Internal state machine for streaming output.
```

### Why the intermediate matters

The encoder/decoder needs the FULL picture before starting work:
- **Pixel format known upfront** → pick internal pipeline (YUV vs RGB, u8 vs f32 path)
- **Metadata before pixels** → headers must be written first in streaming
- **Output format before decode** → skip YUV→RGB if caller wants YUV
- **Gain maps / aux streams declared upfront** → allocate appropriate buffers
- **Limits before any work** → fail fast on oversized input

### Intermediate layers can be factored

`EncodeRequest` doesn't have to be monolithic. Codecs can factor it into helper types
for complex input configurations. For example, metadata gets its own struct rather than
crowding the request with individual ICC/EXIF/XMP fields.

---

## Encoder Design

### Config Structure: Separate Lossy and Lossless

For codecs that support both lossy and lossless, use separate config types so
invalid combinations are unrepresentable at compile time:

```rust
/// Lossy-specific parameters. Can't accidentally set lossless options.
#[derive(Clone, Debug)]
pub struct LossyConfig {
    quality: Quality,
    // ... lossy-specific: subsampling, sharp_yuv, SNS, filter, etc.
}

/// Lossless-specific parameters. Can't accidentally set quality.
#[derive(Clone, Debug)]
pub struct LosslessConfig {
    // ... lossless-specific: near_lossless level, exact mode, etc.
}
```

**Why separate types?** `LossyConfig::new(85).near_lossless(60)` shouldn't compile.
Quality is meaningless for lossless. Near-lossless is meaningless for lossy. Shared
parameters like `method`/`effort` (speed/quality tradeoff) appear on both types.

**Both produce the same EncodeRequest, and offer fluent shortcuts:**
```rust
impl LossyConfig {
    // Full control: returns request for metadata/limits/stop/streaming
    pub fn encode_request(&self, w: u32, h: u32, layout: PixelLayout) -> EncodeRequest;

    // Fluent shortcut: config → bytes (request created internally)
    pub fn encode(&self, pixels: &[u8], w: u32, h: u32, layout: PixelLayout) -> Result<Vec<u8>>;
    pub fn encode_into(&self, pixels: &[u8], w: u32, h: u32, layout: PixelLayout, out: &mut Vec<u8>) -> Result<()>;
}
// Same for LosslessConfig
```

```rust
// Simple — one line, no request visible
let webp = LossyConfig::new(85.0).with_method(4)
    .encode(&pixels, w, h, PixelLayout::Rgba8)?;

// Full control — request layer for metadata, limits, cancellation
let webp = LossyConfig::new(85.0).with_method(4)
    .encode_request(w, h, PixelLayout::Rgba8)
    .with_metadata(&meta)
    .with_limits(&limits)
    .with_stop(&cancel)
    .encode(&pixels)?;
```

**When NOT to split:** Codecs that only support one mode don't need it.
JPEG is lossy-only — just use `EncoderConfig`. GIF is palette-based — just use
`EncoderConfig`. Split only when both modes exist AND have distinct parameter sets.

### Constructor Variants (Not Generic New)

For lossy codecs with multiple color modes, use variant constructors:

```rust
impl LossyConfig {
    /// YCbCr mode — standard, maximum compatibility
    pub fn ycbcr(quality: impl Into<Quality>, subsampling: ChromaSubsampling) -> Self;

    /// XYB mode — perceptual color space, better quality/size
    pub fn xyb(quality: impl Into<Quality>, b_subsampling: XybSubsampling) -> Self;

    /// Grayscale mode — single channel
    pub fn grayscale(quality: impl Into<Quality>) -> Self;
}
```

Makes the color mode decision explicit and visible. Different modes may have different
required parameters (subsampling type varies by mode).

**No `Default` on configs with required semantics.** Quality and color mode represent
fundamental encoding decisions. A default quality of 75 silently produces mediocre
output. Make callers choose.

### Builder/Getter Conventions: `with_` setters, bare-name getters

Use `with_` prefix for consuming builder setters, bare names for getters. This
eliminates ambiguity (`config.progressive()` is always a getter, never a setter).

| Type | Setters | Getters | Rationale |
|------|---------|---------|-----------|
| **Config** | `with_foo(mut self, val) -> Self` | `foo(&self) -> T` | Owned, chainable from constructor. Getters allow inspection. |
| **Request** | `with_foo(mut self, val) -> Self` | None (transient) | Consumed by `encode()`/`build()`. No need to read back. |
| **Encoder/Decoder** | `&mut self` methods (`push`, `add_frame`) | `&self` methods (`info`, `stats`) | Not builders — mutation and queries are separate. |
| **ImageMetadata** | `with_foo(mut self, val) -> Self` | `foo(&self) -> Option<T>` | Small struct, builder + inspection both useful. |

```rust
// Config: with_ setters, bare-name getters
let config = LossyConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .with_progressive(true)
    .with_sharp_yuv(true);
assert!(config.progressive());
assert_eq!(config.subsampling(), ChromaSubsampling::Quarter);

// Request: with_ setters, no getters (consumed immediately)
let request = config.encode_request(w, h, PixelLayout::Rgba8)
    .with_metadata(&meta)
    .with_limits(&limits)
    .with_stop(&cancel);
request.encode(&pixels)?;

// Encoder/Decoder: &mut self mutation, &self inspection
encoder.push(&data, rows, stride)?;
let stats = encoder.stats();
let info = decoder.info();

// ImageMetadata: with_ setters, bare-name getters
let meta = ImageMetadata::new()
    .with_icc_profile(&icc)
    .with_exif(&exif);
assert!(meta.icc_profile().is_some());
```

**Why not bare-name setters?** `config.progressive(true)` looks like a getter call
with a stray argument. `config.progressive()` is ambiguous — getter or setter with
default `true`? The `with_` prefix makes intent unambiguous at every call site.

**Why not `set_` prefix?** `set_foo(&mut self)` implies borrow-based mutation, not
consuming builders. It also doesn't chain from constructors without `let mut`.

### Resource Estimation

Allow callers to check resource requirements before committing:

```rust
impl EncoderConfig {
    /// Typical memory estimate for average content
    pub fn estimate_memory(&self, w: u32, h: u32) -> usize;

    /// Guaranteed upper bound (for resource reservation)
    pub fn estimate_memory_ceiling(&self, w: u32, h: u32) -> usize;
}

// Usage (proxy server scenario)
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
let ceiling = config.estimate_memory_ceiling(4096, 4096);
if ceiling > available_memory {
    return Err("image too large for available resources");
}
```

Estimation is prediction (before work). Limits are enforcement (during work). Both needed.

### Encode Request (Intermediate Layer)

```rust
pub struct EncodeRequest<'a> { /* ... */ }

impl<'a> EncodeRequest<'a> {
    pub fn new(
        config: &'a EncoderConfig,
        width: u32,
        height: u32,
        pixel_layout: PixelLayout,
    ) -> Self;

    // Metadata (factored into its own struct)
    pub fn with_metadata(self, meta: &'a ImageMetadata<'a>) -> Self;

    // Auxiliary streams
    pub fn with_gain_map(self, input: GainMapInput<'a>) -> Self;

    // Controls
    pub fn with_limits(self, limits: &'a Limits) -> Self;
    pub fn with_stop(self, stop: &'a dyn Stop) -> Self;

    // One-shot — "encode" not "finish" (nothing was started)
    pub fn encode(self, pixels: &[u8]) -> Result<Vec<u8>>;
    pub fn encode_into(self, pixels: &[u8], out: &mut Vec<u8>) -> Result<()>;
    #[cfg(feature = "std")]
    pub fn encode_to(self, pixels: &[u8], dest: impl Write) -> Result<()>;

    // Streaming — request produces encoder
    pub fn build(self) -> Result<Encoder<'a>>;
}
```

### Metadata Struct

ICC profiles, EXIF, and XMP are per-image data that belongs with the encode request,
not on the reusable config. Factor into a struct to keep the request clean:

```rust
#[derive(Clone, Debug, Default)]
pub struct ImageMetadata<'a> {
    pub icc_profile: Option<&'a [u8]>,
    pub exif: Option<&'a [u8]>,
    pub xmp: Option<&'a [u8]>,
}

impl<'a> ImageMetadata<'a> {
    pub fn new() -> Self { Self::default() }
    pub fn with_icc_profile(mut self, icc: &'a [u8]) -> Self { self.icc_profile = Some(icc); self }
    pub fn with_exif(mut self, exif: &'a [u8]) -> Self { self.exif = Some(exif); self }
    pub fn with_xmp(mut self, xmp: &'a [u8]) -> Self { self.xmp = Some(xmp); self }
}
```

### Streaming Push API

For memory efficiency with large images:

```rust
pub struct Encoder<'a> { /* ... */ }

impl<'a> Encoder<'a> {
    /// Push rows of pixel data
    pub fn push(&mut self, data: &[u8], rows: usize, stride: usize) -> Result<()>;

    /// Push frames (multi-frame codecs: GIF, animated WebP/JXL)
    pub fn add_frame(&mut self, frame: FrameInput) -> Result<()>;

    /// Streaming completion — "finish" because pushing was "started"
    pub fn finish(self) -> Result<Vec<u8>>;
    pub fn finish_into(self, out: &mut Vec<u8>) -> Result<()>;
    #[cfg(feature = "std")]
    pub fn finish_to(self, dest: impl Write) -> Result<()>;

    pub fn stats(&self) -> &EncodeStats;
}
```

### Naming: encode() vs finish()

- **One-shot** (EncodeRequest): `encode()`, `encode_into()`, `encode_to()`
  Nothing was "started" — you're doing the whole operation.
- **Streaming** (Encoder): `finish()`, `finish_into()`, `finish_to()`
  You pushed rows/frames, now you're completing the stream.

Do NOT use `finish()` for one-shot (nothing was "started") or `encode()` for streaming.

### Quality Abstraction

Support multiple quality scales:

```rust
#[non_exhaustive]
pub enum Quality {
    /// Native quality scale (0-100)
    ApproxJpegli(f32),
    /// Match mozjpeg quality output
    ApproxMozjpeg(u8),
    /// Target SSIMULACRA2 score
    ApproxSsim2(f32),
    /// Target butteraugli distance
    ApproxButteraugli(f32),
}

// Ergonomic conversions
impl From<u8> for Quality { ... }
impl From<f32> for Quality { ... }

// All equivalent ways to specify quality
let config = EncoderConfig::ycbcr(85, ...);                        // u8 -> ApproxJpegli
let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(80), ...);
```

### Cooperative Cancellation

`&dyn Stop` preferred over `S: Stop` generic or `impl Stop` per push:
- No type parameter pollution (`Encoder<'a>` not `Encoder<'a, S>`)
- Vtable cost negligible — cancellation checked a few times per MB
- `&Unstoppable` when caller doesn't care

```rust
/// Re-export from `enough` crate
pub use enough::{Stop, Unstoppable};

// Set once on request, checked throughout encoding
let request = config.encode_request(w, h, layout)
    .with_stop(&cancel_flag);
```

---

## Decoder Design

### Layered API

Use the config/request path for all decoding. Avoid free functions — they can't
accept limits, cancellation, or config, and proliferate with every pixel format.

```rust
// Simple (DecoderConfig::new() uses sensible defaults)
let image = DecoderConfig::new().decode(data, PixelLayout::Rgba8)?;

// Full control
let image = DecoderConfig::new()
    .decode_request(data)
    .with_output_layout(PixelLayout::Rgba8)
    .with_limits(&limits)
    .with_stop(&cancel)
    .decode()?;
```

### Probing: Info Before Decode

Two patterns, both avoiding double-parsing and double-buffering:

#### Static probe (when you have bytes available)

```rust
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
    pub format: BitstreamFormat,
    // format-specific fields
}

impl ImageInfo {
    /// Minimum bytes needed to attempt header parsing.
    /// For fixed-header formats (WebP, GIF, JXL) this is exact.
    /// For variable-header formats (JPEG) this is a typical minimum —
    /// from_bytes() may return NeedMoreData if SOF hasn't appeared yet.
    pub const PROBE_BYTES: usize = 4096;  // format-specific

    /// Parse header from a byte slice. Returns NeedMoreData if
    /// the header is incomplete (variable-length formats).
    pub fn from_bytes(data: &[u8]) -> Result<Self, ProbeError>;
}

pub enum ProbeError {
    NeedMoreData,        // not enough bytes yet
    InvalidFormat,       // not this format at all
    Corrupt(String),     // header present but malformed
}

// Usage with a slice
let info = ImageInfo::from_bytes(&first_4k)?;
if info.width as u64 * info.height as u64 > max_pixels {
    return Err("too large");
}
```

#### Streaming probe (parse header, pause, continue decoding)

The decoder itself has a two-phase API: build parses the header, then you
inspect info before committing to decode. No re-parsing, no double-buffering.

```rust
// Phase 1: Build decoder — parses header, stops before pixel decode
let mut decoder = DecodeRequest::new(&config, &data)
    .with_limits(&limits)
    .build()?;

// Phase 2: Inspect — header already parsed during build()
let info = decoder.info();
if info.width > 8192 { return Err("too wide"); }

// Phase 3: Continue — decodes from where it left off
let output = decoder.decode()?;
```

For incremental network streams:

```rust
let mut stream = StreamingDecoder::new();

// Feed bytes as they arrive
stream.append(&chunk1)?;  // → NeedMoreData
stream.append(&chunk2)?;  // → HeaderReady

// Inspect without re-parsing
let info = stream.info()?;  // available once HeaderReady

// Decision point: proceed or abort
stream.append(&chunk3)?;  // → NeedMoreData
stream.append(&chunk4)?;  // → Complete

// Finish — no double-buffering, decoder already has everything
let (pixels, w, h) = stream.finish_rgba()?;
```

**Key invariant:** Once the header is parsed (during `build()` or after
`HeaderReady`), the decoder holds the parse state. Calling `decode()` or
`finish_*()` continues from that state. The header bytes are never re-read.

### Decode Request (Intermediate Layer)

Telling the decoder the desired output format BEFORE it starts lets it skip
unnecessary conversions (e.g., skip YUV→RGB if caller wants YUV):

```rust
pub struct DecodeRequest<'a> { /* ... */ }

impl<'a> DecodeRequest<'a> {
    pub fn new(config: &'a DecoderConfig, data: &'a [u8]) -> Self;

    // Desired output — lets decoder optimize internal pipeline
    pub fn with_output_layout(self, layout: PixelLayout) -> Self;

    // Controls
    pub fn with_limits(self, limits: &'a Limits) -> Self;
    pub fn with_stop(self, stop: &'a dyn Stop) -> Self;

    // One-shot
    pub fn decode(self) -> Result<DecodeOutput>;
    pub fn decode_into(self, output: &mut [u8]) -> Result<ImageInfo>;

    // Streaming
    pub fn build(self) -> Result<Decoder<'a>>;
}
```

### Zero-Copy Decode Into

For performance-critical paths:

```rust
/// Decode directly into pre-allocated buffer
pub fn decode_into(
    data: &[u8],
    output: &mut [u8],
    stride_bytes: u32,
) -> Result<(u32, u32)>;
```

### Streaming Decode

For progressive display or memory-constrained environments:

```rust
pub struct Decoder<'a> { /* ... */ }

pub enum StreamStatus {
    NeedMoreData,
    HeaderReady,
    Complete,
}

impl<'a> Decoder<'a> {
    pub fn info(&self) -> &ImageInfo;
    pub fn next_frame(&mut self) -> Result<Option<Frame>>;
}

// Incremental input (network streams)
pub struct StreamingDecoder { /* ... */ }

impl StreamingDecoder {
    pub fn new() -> Self;
    pub fn with_config(config: DecoderConfig) -> Self;
    pub fn append(&mut self, data: &[u8]) -> Result<StreamStatus>;
    pub fn info(&self) -> Result<ImageInfo>;
    pub fn is_complete(&self) -> bool;
    pub fn finish_rgba(self) -> Result<(Vec<u8>, u32, u32)>;
}
```

### Decode to Multiple Formats

```rust
// Via PixelLayout on request — one method, any format
let rgba = config.decode_request(data).with_output_layout(PixelLayout::Rgba8).decode()?;
let bgra = config.decode_request(data).with_output_layout(PixelLayout::Bgra8).decode()?;
let yuv  = config.decode_request(data).with_output_layout(PixelLayout::Yuv420).decode()?;
```

No free functions like `decode_rgba()` / `decode_bgra()` — they proliferate with
every pixel format and can't accept limits or cancellation.

---

## Common Patterns

### Pixel Layout Enum

```rust
#[non_exhaustive]
pub enum PixelLayout {
    // 8-bit sRGB
    Rgb8,
    Rgba8,
    Bgr8,
    Bgra8,
    Gray8,
    GrayAlpha8,

    // 16-bit linear
    Rgb16,
    Rgba16,
    Gray16,

    // 32-bit float linear (0.0-1.0)
    RgbF32,
    RgbaF32,
    GrayF32,

    // Planar (JPEG, WebP internal)
    Yuv420,
}

impl PixelLayout {
    pub const fn bytes_per_pixel(&self) -> usize;
    pub const fn is_linear(&self) -> bool;
    pub const fn has_alpha(&self) -> bool;
}
```

Not every codec supports every layout. Request creation returns an error if the
codec doesn't support the requested layout.

**Generics permitted at boundary**: Use `PixelLayout` enum as the primary API to
avoid monomorphizing the entire pipeline per pixel type. Generic convenience methods
at the boundary are fine — they validate byte length, set the layout, and delegate:

```rust
impl<'a> EncodeRequest<'a> {
    pub fn encode_rgb<P: Pixel>(self, pixels: &[P]) -> Result<Vec<u8>> {
        let bytes = bytemuck::cast_slice(pixels);
        self.encode(bytes)
    }
}
```

**Measure before deciding** on generics vs enum internally: build an example with
1 pixel type vs 4 (RGB8, RGBA8, RGB16, RGBF32), compare binary size and compile time.

### Minimum Layout Support: RGBA8 and BGRA8

Every codec must support `Rgba8` and `Bgra8` for both encode and decode, regardless
of whether the format natively supports alpha. This ensures callers can use a single
pixel format across all codecs without branching.

- **Decode**: If the format has no alpha channel (JPEG, lossy WebP), set alpha to 255
  (fully opaque) in the output buffer.
- **Encode**: If the format has no alpha channel, silently ignore the alpha bytes in
  the input. Do not error.

This applies to both one-shot and streaming APIs. Codecs that natively support alpha
(PNG, lossless WebP, GIF, JXL) pass it through as-is.

### Limits (Resource Enforcement)

```rust
/// All fields default to None (no limit).
#[derive(Clone, Debug, Default)]
pub struct Limits {
    pub max_width: Option<u64>,
    pub max_height: Option<u64>,
    pub max_pixels: Option<u64>,        // width * height
    pub max_memory_bytes: Option<u64>,
}
```

- Both encoders AND decoders should accept optional Limits
- Users cannot predict encoder memory — Limits are needed on both sides
- `max_pixels` catches the real problem (1×10M and 3162×3162 use similar memory)
- Estimation predicts cost (before work). Limits enforce bounds (during work). Both needed.

### Error Design

```rust
/// Separate encode/decode errors — different operations have different failure modes.
#[derive(Debug)]
#[non_exhaustive]
pub enum EncodeError {
    InvalidInput { /* ... */ },
    InvalidConfig { /* ... */ },
    LimitExceeded { /* ... */ },
    Cancelled,
    Oom(TryReserveError),
    #[cfg(feature = "std")]
    Io(std::io::Error),
    // format-specific variants
}

/// Use whereat crate for location tracking — invaluable for debugging codec issues.
pub type Result<T> = core::result::Result<T, At<EncodeError>>;
```

### Feature Flags

```rust
// Cargo.toml
[features]
default = ["decode", "encode"]
decode = []
encode = []
std = []          // For io::Write support, encode_to()/finish_to()
streaming = []    // Incremental decode/encode
animation = []    // Multi-frame support
parallel = []     // Multi-threaded encoding
```

### Essential Crates

```toml
[dependencies]
enough = "0.1"           # Cooperative cancellation (Stop trait)
whereat = "0.1"          # Error location tracking (At<E>)
rgb = "0.8"              # Typed pixel structs (RGB8, RGBA8, etc.)
imgref = "1.10"          # Strided 2D image views
bytemuck = "1.14"        # Safe transmute for pixel casting
archmage = "0.5"         # Token-based safe SIMD dispatch
```

---

## Project Standards

### No Legacy APIs

We have no external users — all dependents on crates.io are owned by us. When making
breaking changes, just bump the 0.x version. Do not waste time on deprecation shims,
`#[deprecated]` aliases, or backwards-compatible wrappers. Delete the old API and move on.

### Safety: `#![forbid(unsafe_code)]`

All crates must use `#![forbid(unsafe_code)]` with the default feature set. SIMD code
goes through `archmage` (which generates unsafe internally via proc macros, bypassing
the lint). An optional `unchecked` feature may relax this for performance-critical paths
with trusted input, but the default build must be safe.

### no_std + alloc

Target `no_std` + `alloc` support. At minimum, the crate must compile to
`wasm32-unknown-unknown`. The `std` feature gates IO traits (`Read`, `Write`),
`encode_to()`/`finish_to()` methods, and anything requiring the standard library.
Core encode/decode functionality should work without `std`.

**Do not assume things require `std`.** As of Rust 1.92, almost everything lives in
`core::` — including `core::error::Error`, `core::fmt`, `core::hash`, etc. The `alloc`
crate provides `Vec`, `Box`, `String`, `BTreeMap`, etc. The only things that truly
require `std` are IO traits, filesystem, networking, and `HashMap` (use `hashbrown`
instead). Always test with `cargo build --no-default-features` before claiming something
needs `std`.

For wasm targets that need timing (benchmarks, timeouts), use the
[`wasmtimer`](https://crates.io/crates/wasmtimer) crate instead of `std::time::Instant`.

### CI and Quality

- **Code coverage**: CI must upload coverage to codecov (or coveralls). Aim for meaningful
  coverage of encode/decode paths, not vanity percentage.
- **README badges**: Build status, crates.io version, docs.rs, coverage, license — at the
  top of every README.
- **README usage examples**: Every crate README must include working code examples showing
  basic encode and decode usage. These should be `rust` doc-tested or have equivalent
  integration tests to prevent bitrot.

### Threat Model: Malicious Input on Real-Time Proxies

These codecs run on image proxies that process untrusted input in real time. Every
decode path is an attack surface. Design accordingly:

- **Assume every input is adversarial.** Fuzzed, hand-crafted, or corrupted images
  must never cause panics, OOM, infinite loops, or disproportionate CPU usage.
- **Bound everything.** Memory, CPU time, output dimensions, decompression ratio.
  Limits must be checked before allocation, not after.
- **No amplification.** A 1KB input must not produce 1GB of memory usage or 10 seconds
  of CPU time. Track decompression ratio and abort on suspicious expansion.

### Fuzzing

Every codec must have `cargo-fuzz` targets covering at minimum:
- **Decode fuzzer**: arbitrary bytes → decoder (the primary attack surface)
- **Roundtrip fuzzer**: encode → decode, verify consistency
- **Limits fuzzer**: verify Limits enforcement under adversarial input
- **Streaming fuzzer**: feed bytes incrementally in random chunk sizes

Fuzz targets should be run regularly (not just once). CI should build fuzz targets
to prevent bitrot. Seed corpora should include known-tricky images and prior crash cases.

### Periodic Security Audits

Schedule periodic deep-dive reviews (not just fuzzing) specifically looking for:
- **DoS vectors**: inputs that cause O(n²) or worse behavior, hash flooding, excessive
  backtracking, or algorithmic complexity attacks
- **Memory amplification**: small inputs that trigger large allocations (zip bombs,
  decompression bombs, palette expansion, huge dimensions with tiny compressed data)
- **Worst-case performance paths**: inputs that hit pathological cases in entropy
  decoding, LZW, Huffman tree construction, or color conversion
- **Integer overflow**: dimension multiplication, stride calculation, buffer size computation
- **Infinite loops**: malformed headers that cause parsers to loop without progress

Document findings in each codec's CLAUDE.md under "Known Bugs" or "Security Notes".
Fix critical issues immediately; track others with severity ratings.

## Anti-Patterns

### 1. Giant Constructor

```rust
// BAD: Too many required parameters, order matters
let enc = Encoder::new(width, height, quality, subsampling, progressive, ...)?;

// GOOD: Config builder + request
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter).with_progressive(true);
let encoded = config.encode_request(w, h, PixelLayout::Rgb8).encode(&pixels)?;
```

### 2. Default for Semantic Requirements

```rust
// BAD: What quality does Default give? What color mode?
let config = EncoderConfig::default();

// GOOD: Explicit quality and mode
let config = LossyConfig::ycbcr(85, ChromaSubsampling::Quarter);
```

### 3. Metadata on Reusable Config

```rust
// BAD: ICC profile baked into reusable config — wrong if batch-encoding
//      images with different color profiles
let config = EncoderConfig::new().icc_profile(icc_bytes);

// GOOD: Metadata on request or factored struct
let meta = ImageMetadata::new().icc_profile(&icc_bytes);
let request = config.encode_request(w, h, layout).with_metadata(&meta);
```

### 4. Generic Type Pollution

```rust
// BAD: Generic infects encoder type, complicates boxing/FFI
let enc: RgbEncoder<rgb::RGBA<u8>> = config.encode_from_rgb(w, h)?;

// GOOD: Enum-based, generic at boundary only
let request = config.encode_request(w, h, PixelLayout::Rgba8);
request.encode(&pixels)?;           // enum dispatch inside
request.encode_rgb(&typed_pixels)?; // generic convenience, delegates to enum
```

### 5. Non-Interruptible Operations

```rust
// BAD: No way to cancel a 50MP encode
fn encode(pixels: &[u8]) -> Result<Vec<u8>>;

// GOOD: Cancellation via request
let request = config.encode_request(w, h, layout).with_stop(&cancel_flag);
request.encode(&pixels)?;
```

### 6. Allocating When Caller Has Buffer

```rust
// BAD: Always allocates
fn decode(data: &[u8]) -> Result<Vec<u8>>;

// GOOD: Both options
fn decode(data: &[u8]) -> Result<Vec<u8>>;
fn decode_into(data: &[u8], output: &mut [u8], stride: u32) -> Result<(u32, u32)>;
```

### 7. finish() on One-Shot, encode() on Streaming

```rust
// BAD: Semantic mismatch
let request = EncodeRequest::new(...);
request.finish()?;  // Nothing was "started"!

// BAD: Semantic mismatch
encoder.push(&data)?;
encoder.encode()?;  // You already encoded, now you're just flushing

// GOOD: Names match semantics
request.encode()?;   // One-shot: whole operation
encoder.finish()?;   // Streaming: completing what was started
```

---

## Summary Checklist

### Encoder
- [ ] Three-layer: Config → Request → Encoder
- [ ] Config is dimension-independent and reusable
- [ ] Required parameters (quality, mode) via named constructors, not Default
- [ ] Optional parameters via `with_` builder methods, bare-name getters
- [ ] EncodeRequest binds dimensions, pixel format, metadata, limits, stop
- [ ] Metadata factored into ImageMetadata struct, on request not config
- [ ] Streaming push API for large images via `request.build()` → Encoder
- [ ] One-shot uses `encode()`, streaming uses `finish()`
- [ ] `_to()` variants std-only (IO abstraction, not file IO)
- [ ] Memory estimation before encoding
- [ ] Cooperative cancellation via `&dyn Stop`
- [ ] Limits with `Option<u64>` fields including `max_pixels`
- [ ] Quality abstraction supporting multiple scales

### Decoder
- [ ] Three-layer: Config → Request → Decoder
- [ ] Simple one-shot via config method (not free functions)
- [ ] DecodeRequest specifies desired output format (enables internal optimization)
- [ ] Info retrieval before decode
- [ ] Zero-copy decode_into variants
- [ ] Streaming decode for progressive display / network input
- [ ] Multiple output formats via PixelLayout

### Both
- [ ] `#[non_exhaustive]` error enums
- [ ] `At<>` error wrapping for location tracking
- [ ] Separate `EncodeError` / `DecodeError` types
- [ ] Feature flags for optional functionality
- [ ] `#![forbid(unsafe_code)]` with default features
- [ ] no_std + alloc support (minimum: wasm32 target)
- [ ] `EncodeStats` / allocation tracking
- [ ] Resource estimation on config, Limits enforcement on request
- [ ] No legacy/deprecated API shims — just bump 0.x version
- [ ] CI with codecov coverage upload
- [ ] README badges (build, crates.io, docs.rs, coverage, license)
- [ ] README usage examples (doc-tested or integration-tested)
- [ ] Fuzz targets: decode, roundtrip, limits, streaming
- [ ] Safe for malicious input on real-time proxies (bounded memory/CPU, no amplification)

---

## Reference Implementations

- **Encoder**: [zenjpeg](https://github.com/imazen/zenjpeg) — three-layer pattern, streaming push, quality abstraction
- **Decoder**: [zenwebp](https://github.com/imazen/zenwebp) — EncodeRequest, DecodeRequest, Limits, streaming decoder

## License

CC0 — Public Domain. Use these patterns freely.
