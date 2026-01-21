# Image Codec API Design Guidelines

Best practices for designing image encoder and decoder APIs in Rust, distilled from production codecs including [jpegli-rs](https://github.com/imazen/jpegli-rs) and [webpx](https://github.com/imazen/webpx).

## Core Principles

1. **Config-first, dimensions-second** - Configuration should be reusable across images
2. **Required parameters via constructors, optionals via builders** - No `Default` for configs with required semantics
3. **Multiple entry points for different use cases** - One-shot, streaming, zero-copy
4. **Resource estimation before work** - Memory and compute cost prediction
5. **Cooperative cancellation** - Long operations should be interruptible
6. **Layered complexity** - Simple things simple, complex things possible

---

## Encoder Design

### Config Structure

```rust
/// Dimension-independent, reusable encoder configuration.
///
/// No Default impl - quality and color mode are semantic requirements.
pub struct EncoderConfig {
    quality: Quality,
    color_mode: ColorMode,
    // ... optional settings with sensible defaults
}
```

**Why no `Default`?** Quality and color mode represent fundamental encoding decisions. Forcing explicit choice prevents accidental low-quality output or wrong color space.

### Constructor Variants (Not Generic New)

```rust
impl EncoderConfig {
    /// YCbCr mode - standard, maximum compatibility
    pub fn ycbcr(quality: impl Into<Quality>, subsampling: ChromaSubsampling) -> Self;

    /// XYB mode - perceptual color space, better quality/size
    pub fn xyb(quality: impl Into<Quality>, b_subsampling: XybSubsampling) -> Self;

    /// Grayscale mode - single channel
    pub fn grayscale(quality: impl Into<Quality>) -> Self;
}
```

**Why variant constructors?**
- Makes the color mode decision explicit and visible
- Different modes may have different required parameters (subsampling type varies)
- Self-documenting: `EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)` vs `EncoderConfig::new().color_mode(ColorMode::YCbCr).subsampling(...)`

### Builder Pattern for Optionals

```rust
impl EncoderConfig {
    pub fn progressive(mut self, enable: bool) -> Self { ... }
    pub fn optimize_huffman(mut self, enable: bool) -> Self { ... }
    pub fn icc_profile(mut self, profile: impl Into<Vec<u8>>) -> Self { ... }
    pub fn sharp_yuv(mut self, enable: bool) -> Self { ... }
}

// Usage
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .progressive(true)
    .sharp_yuv(true);
```

### Multiple Entry Points

Provide encoder creation for different input scenarios:

```rust
impl EncoderConfig {
    /// Raw bytes with explicit layout
    pub fn encode_from_bytes(&self, w: u32, h: u32, layout: PixelLayout) -> Result<BytesEncoder>;

    /// Typed pixels (rgb crate types)
    pub fn encode_from_rgb<P: Pixel>(&self, w: u32, h: u32) -> Result<RgbEncoder<P>>;

    /// Pre-converted planar YCbCr (video pipelines)
    pub fn encode_from_ycbcr_planar(&self, w: u32, h: u32) -> Result<YCbCrPlanarEncoder>;
}
```

### Streaming Push API

For memory efficiency with large images:

```rust
pub trait Encoder {
    /// Push rows of pixel data
    fn push_packed(&mut self, data: &[u8], stop: impl Stop) -> Result<()>;

    /// Finish encoding, return output
    fn finish(self) -> Result<Vec<u8>>;
}

// Usage
let mut enc = config.encode_from_bytes(1920, 1080, PixelLayout::Rgb8Srgb)?;
for chunk in pixel_chunks {
    enc.push_packed(&chunk, Unstoppable)?;
}
let jpeg = enc.finish()?;
```

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

### Cooperative Cancellation

Long operations should be interruptible:

```rust
/// Re-export from `enough` crate
pub use enough::{Stop, Unstoppable};

// In encoder
fn push_packed(&mut self, data: &[u8], stop: impl Stop) -> Result<()> {
    for block in blocks {
        stop.check()?;  // Check for cancellation
        self.encode_block(block);
    }
    Ok(())
}

// Usage with timeout
let cancel = Arc::new(AtomicBool::new(false));
// In another thread: cancel.store(true, Ordering::Relaxed);
enc.push_packed(&data, &cancel)?;
```

### Quality Abstraction

Support multiple quality scales:

```rust
#[derive(Clone, Copy)]
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

// Usage - all equivalent ways to specify quality
let config = EncoderConfig::ycbcr(85, ...);                        // u8 -> ApproxJpegli
let config = EncoderConfig::ycbcr(85.0, ...);                      // f32 -> ApproxJpegli
let config = EncoderConfig::ycbcr(Quality::ApproxMozjpeg(80), ...);
```

---

## Decoder Design

### Layered API

Provide both simple functions and a builder for advanced use:

```rust
// Simple one-shot functions
pub fn decode_rgba(data: &[u8]) -> Result<(Vec<u8>, u32, u32)>;
pub fn decode_rgb(data: &[u8]) -> Result<(Vec<u8>, u32, u32)>;

// Typed pixel output
pub fn decode<P: DecodePixel>(data: &[u8]) -> Result<(Vec<P>, u32, u32)>;

// Builder for advanced options
pub struct Decoder<'a> { ... }

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self>;
    pub fn info(&self) -> &ImageInfo;
    pub fn crop(self, left: u32, top: u32, w: u32, h: u32) -> Self;
    pub fn scale(self, w: u32, h: u32) -> Self;
    pub fn decode_rgba(self) -> Result<ImgVec<RGBA8>>;
}
```

### Info Before Decode

Allow inspection without full decode:

```rust
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
    pub format: ImageFormat,
}

impl ImageInfo {
    pub fn from_webp(data: &[u8]) -> Result<Self>;
}

// Usage
let info = ImageInfo::from_webp(&data)?;
if info.width * info.height > MAX_PIXELS {
    return Err("image too large");
}
let (pixels, _, _) = decode_rgba(&data)?;
```

### Zero-Copy Decode Into

For performance-critical paths:

```rust
/// Decode directly into pre-allocated buffer
pub fn decode_rgba_into(
    data: &[u8],
    output: &mut [u8],
    stride_bytes: u32
) -> Result<(u32, u32)>;

/// Typed pixel version
pub fn decode_into<P: DecodePixel>(
    data: &[u8],
    output: &mut [P],
    stride_pixels: u32  // Note: stride unit matches buffer type
) -> Result<(u32, u32)>;
```

**Stride convention**: The stride unit should match the buffer type. Byte buffers use byte strides, pixel buffers use pixel strides.

### Streaming Decode

For progressive display or memory-constrained environments:

```rust
pub struct StreamingDecoder { ... }

pub enum DecodeStatus {
    Complete,
    NeedMoreData,
    Partial(u32),  // rows decoded so far
}

impl StreamingDecoder {
    pub fn new(color_mode: ColorMode) -> Result<Self>;
    pub fn with_buffer(output: &mut [u8], stride: usize, mode: ColorMode) -> Result<Self>;
    pub fn append(&mut self, data: &[u8]) -> Result<DecodeStatus>;
    pub fn get_partial(&self) -> Option<(&[u8], u32, u32)>;
    pub fn finish(self) -> Result<(Vec<u8>, u32, u32)>;
}

// Usage
let mut decoder = StreamingDecoder::new(ColorMode::Rgba)?;
for chunk in network_stream {
    match decoder.append(&chunk)? {
        DecodeStatus::Complete => break,
        DecodeStatus::Partial(rows) => display_partial(decoder.get_partial()),
        DecodeStatus::NeedMoreData => continue,
        _ => {}  // future variants
    }
}
let (pixels, w, h) = decoder.finish()?;
```

### Decode to Multiple Formats

Support common pixel layouts:

```rust
impl Decoder<'_> {
    pub fn decode_rgba(self) -> Result<ImgVec<RGBA8>>;
    pub fn decode_rgb(self) -> Result<ImgVec<RGB8>>;
    pub fn decode_bgra(self) -> Result<ImgVec<BGRA8>>;  // Windows/GPU native
    pub fn decode_bgr(self) -> Result<ImgVec<BGR8>>;   // OpenCV
    pub fn decode_yuv(self) -> Result<YuvPlanes>;      // Video pipelines
}
```

---

## Common Patterns

### Pixel Layout Enum

```rust
pub enum PixelLayout {
    Rgb8Srgb,      // 3 bytes, sRGB gamma
    Bgr8Srgb,      // 3 bytes, BGR order (Windows)
    Rgbx8Srgb,     // 4 bytes, 4th ignored
    Bgrx8Srgb,     // 4 bytes, BGR, 4th ignored
    Gray8Srgb,     // 1 byte grayscale
    Rgb16Linear,   // 6 bytes, linear light
    RgbF32Linear,  // 12 bytes, float linear (HDR)
    YCbCr8,        // 3 bytes, pre-converted
}

impl PixelLayout {
    pub fn bytes_per_pixel(&self) -> usize;
    pub fn is_linear(&self) -> bool;
    pub fn has_alpha(&self) -> bool;
}
```

### Error Design

```rust
#[derive(Debug)]
#[non_exhaustive]  // Allow adding variants without breaking changes
pub enum Error {
    InvalidInput(String),
    InvalidConfig(String),
    DecodeFailed(DecodingError),
    EncodeFailed(EncodingError),
    OutOfMemory,
    Stopped(StopReason),  // Cooperative cancellation
    Io(std::io::Error),
}

pub type Result<T> = std::result::Result<T, At<Error>>;

// Use whereat crate for location tracking
use whereat::*;
return Err(at!(Error::InvalidInput("buffer too small".into())));
```

### Feature Flags

```rust
// Cargo.toml
[features]
default = ["decode", "encode"]
decode = []
encode = []
std = []          # For no_std support
streaming = []    # Incremental decode/encode
animation = []    # Multi-frame support
parallel = []     # Multi-threaded encoding
```

---

## Anti-Patterns to Avoid

### 1. Giant Constructor

```rust
// BAD: Too many required parameters, order matters
let enc = Encoder::new(width, height, quality, subsampling, progressive,
                       optimize_huffman, icc_profile, ...)?;

// GOOD: Config builder
let enc = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter)
    .progressive(true)
    .encode_from_bytes(width, height, layout)?;
```

### 2. Default for Semantic Requirements

```rust
// BAD: What quality does Default give?
let config = EncoderConfig::default();

// GOOD: Explicit quality
let config = EncoderConfig::ycbcr(85, ChromaSubsampling::Quarter);
```

### 3. Mixed Abstraction Levels

```rust
// BAD: Mixing raw bytes and typed pixels in same function
fn encode(pixels: &[u8], format: PixelFormat, ...) -> Result<Vec<u8>>;

// GOOD: Separate entry points
fn encode_from_bytes(data: &[u8], layout: PixelLayout, ...) -> Result<BytesEncoder>;
fn encode_from_rgb<P: Pixel>(pixels: &[P], ...) -> Result<RgbEncoder<P>>;
```

### 4. Non-Interruptible Operations

```rust
// BAD: No way to cancel
fn encode(pixels: &[u8]) -> Result<Vec<u8>>;

// GOOD: Cancellation support
fn push_packed(&mut self, data: &[u8], stop: impl Stop) -> Result<()>;
```

### 5. Allocating When Caller Has Buffer

```rust
// BAD: Always allocates
fn decode(data: &[u8]) -> Result<Vec<u8>>;

// GOOD: Both options
fn decode(data: &[u8]) -> Result<Vec<u8>>;
fn decode_into(data: &[u8], output: &mut [u8], stride: u32) -> Result<(u32, u32)>;
```

---

## Summary Checklist

### Encoder
- [ ] Config struct is dimension-independent and reusable
- [ ] Required parameters (quality, mode) via named constructors
- [ ] Optional parameters via builder methods
- [ ] Multiple entry points for bytes/typed pixels/planar
- [ ] Streaming push API for large images
- [ ] Memory estimation before encoding
- [ ] Cooperative cancellation support
- [ ] Quality abstraction supporting multiple scales

### Decoder
- [ ] Simple one-shot functions for basic use
- [ ] Builder pattern for advanced options (crop, scale)
- [ ] Info retrieval before decode
- [ ] Zero-copy decode_into variants
- [ ] Streaming decode for progressive display
- [ ] Multiple output formats (RGBA, RGB, BGR, YUV)
- [ ] Typed pixel output support

### Both
- [ ] Non-exhaustive error enum
- [ ] Feature flags for optional functionality
- [ ] no_std support where feasible
- [ ] Clear stride conventions documented

---

## Reference Implementations

- **Encoder**: [jpegli-rs](https://github.com/imazen/jpegli-rs) - `jpegli::encoder` module
- **Decoder**: [webpx](https://github.com/imazen/webpx) - `decode.rs`, `streaming.rs`

## License

CC0 - Public Domain. Use these patterns freely.
