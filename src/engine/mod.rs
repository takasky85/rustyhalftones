//! Halftone engine. Pure `std`, no external crates, so it can be tested on its own.
pub mod export;
pub mod layout;
pub mod pages;
pub mod par;
pub mod render;
pub mod spot;
pub mod tonemap;

#[cfg(test)]
mod tests;
