#[derive(Clone, Copy, Debug)]
pub struct PageSize {
    pub name: &'static str,
    pub w_mm: f32,
    pub h_mm: f32,
}

const fn p(name: &'static str, w_mm: f32, h_mm: f32) -> PageSize {
    PageSize { name, w_mm, h_mm }
}

/// Portrait dimensions. Includes ISO, US and common screen-printing film sizes.
pub const PAGE_SIZES: &[PageSize] = &[
    p("Letter (8.5 x 11 in)", 215.9, 279.4),
    p("Legal (8.5 x 14 in)", 215.9, 355.6),
    p("Tabloid (11 x 17 in)", 279.4, 431.8),
    p("12 x 18 in", 304.8, 457.2),
    p("13 x 19 in (Super B)", 330.2, 482.6),
    p("18 x 24 in", 457.2, 609.6),
    p("20 x 24 in", 508.0, 609.6),
    p("24 x 36 in", 609.6, 914.4),
    p("5 x 7 in", 127.0, 177.8),
    p("8 x 10 in", 203.2, 254.0),
    p("11 x 14 in", 279.4, 355.6),
    p("16 x 20 in", 406.4, 508.0),
    p("A0", 841.0, 1189.0),
    p("A1", 594.0, 841.0),
    p("A2", 420.0, 594.0),
    p("A3", 297.0, 420.0),
    p("A4", 210.0, 297.0),
    p("A5", 148.0, 210.0),
    p("A6", 105.0, 148.0),
    p("B4", 250.0, 353.0),
    p("B5", 176.0, 250.0),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Unit {
    Mm,
    Inch,
}

impl Unit {
    pub fn suffix(self) -> &'static str {
        match self {
            Unit::Mm => " mm",
            Unit::Inch => " in",
        }
    }
    pub fn from_mm(self, mm: f32) -> f32 {
        match self {
            Unit::Mm => mm,
            Unit::Inch => mm / 25.4,
        }
    }
    pub fn to_mm(self, v: f32) -> f32 {
        match self {
            Unit::Mm => v,
            Unit::Inch => v * 25.4,
        }
    }
}
