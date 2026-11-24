use flate2::{write::ZlibEncoder, Compression};
use std::io::Write;

/// zlib-compress `data`. Film rasters are mostly white, so a fast level is plenty.
pub fn zlib(data: &[u8]) -> Vec<u8> {
    let mut e = ZlibEncoder::new(Vec::with_capacity(data.len() / 16 + 64), Compression::new(3));
    e.write_all(data).expect("in-memory write");
    e.finish().expect("in-memory finish")
}
