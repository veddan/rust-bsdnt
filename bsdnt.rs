#[feature(macro_rules)];
#[feature(globs)];
#[feature(asm)];

use std::libc::{c_void, c_char, size_t, c_long, c_int};
use std::num::{One, Zero};

#[cfg(target_word_size = "32")]
type word_t = u32;

#[cfg(target_word_size = "64")]
type word_t = u64;

#[cfg(target_word_size = "32")]
type dword_t = u64;

#[cfg(target_word_size = "64")]
// FIXME This should have 16-byte alignment (blocked on mozilla/rust #4578)
type dword_t = (u64, u64);

type len_t = c_long;
type sword_t = c_long;
type bits_t = c_long;

type nn_t = *mut word_t;
type nn_src_t = *word_t;

struct zz_struct {
    n: nn_t,
    size: len_t,
    alloc: len_t
}

type zz_ptr = *mut zz_struct;
type zz_srcptr = *zz_struct;

#[link(name = "bsdnt")]
extern "C" {
    fn nn_cmp_m(a: nn_src_t, b: nn_src_t, m: len_t) -> c_int;

    fn zz_init(r: zz_ptr);
    fn zz_init_fit(r: zz_ptr, m: len_t);
    fn zz_clear(r: zz_ptr);
    fn zz_fit(r: zz_ptr, m: len_t);

    fn zz_equali(r: zz_srcptr, c: sword_t) -> c_int;
    fn zz_cmpi(a: zz_srcptr, b: sword_t) -> c_int;
    fn zz_cmp(a: zz_srcptr, b: zz_srcptr) -> c_int;
    fn zz_cmpabs(a: zz_srcptr, b: zz_srcptr) -> c_int;
    fn zz_is_zero(r: zz_srcptr) -> c_int;

    // fn zz_random

    fn zz_seti(r: zz_ptr, c: sword_t);
    fn zz_set(r: zz_ptr, a: zz_srcptr);
    fn zz_neg(r: zz_ptr, a: zz_srcptr);
    fn zz_swap(a: zz_ptr, b: zz_ptr);
    fn zz_zero(a: zz_ptr);

    fn zz_addi(r: zz_ptr, a: zz_srcptr, c: sword_t);
    fn zz_add(r: zz_ptr, a: zz_srcptr, b: zz_srcptr);
    fn zz_subi(r: zz_ptr, a: zz_srcptr, c: sword_t);
    fn zz_sub(r: zz_ptr, a: zz_srcptr, b: zz_srcptr);
    fn zz_muli(r: zz_ptr, a: zz_srcptr, c: sword_t);
    fn zz_mul(r: zz_ptr, a: zz_srcptr, b: zz_srcptr);
    fn zz_div(q: zz_ptr, a: zz_srcptr, b: zz_srcptr);
    fn zz_divrem(q: zz_ptr, r: zz_ptr, a: zz_srcptr, b: zz_srcptr);
    fn zz_divremi(q: zz_ptr, a: zz_srcptr, b: sword_t) -> sword_t;

    fn zz_gcd(g: zz_ptr, a: zz_srcptr, b: zz_srcptr);

    fn zz_set_str(a: zz_ptr, s: *c_char) -> size_t;
    fn zz_get_str(a: zz_srcptr) -> *mut c_char;
    fn zz_print(a: zz_srcptr);

    fn nn_bit_test(a: nn_src_t, b: bits_t) -> c_int;
}

unsafe fn zz_equal(a: zz_srcptr, b: zz_srcptr) -> bool {
    if (*a).size != (*b).size { return false; }
    if (*a).size == 0 { return true; }
    return nn_cmp_m((*a).n as nn_src_t, (*b).n as nn_src_t, (*a).size.abs()) == 0;
}

macro_rules! binop_new(
    ($a:expr $b:expr $fun:expr) => (
        unsafe {
            let mut res = Bsdnt::new();
            $fun(&mut res.zz, &$a.zz, &$b.zz);
            res
        }
    );
)

pub struct Bsdnt {
    priv zz: zz_struct
}

impl Drop for Bsdnt {
    fn drop(&mut self) { unsafe { zz_clear(&mut self.zz); } }
}

impl Bsdnt {
    pub fn new() -> Bsdnt {
        unsafe {
            let mut zz = std::unstable::intrinsics::uninit();
            zz_init(&mut zz);
            Bsdnt { zz: zz }
        }
    }

    pub fn new_reserve(words: uint) -> Bsdnt {
        unsafe {
            let mut zz = std::unstable::intrinsics::uninit();
            zz_init_fit(&mut zz, words as len_t);
            Bsdnt { zz: zz }
        }
    }

    pub fn reserve(&mut self, words: uint) {
        unsafe { zz_fit(&mut self.zz, words as len_t); }
    }

}

impl Eq for Bsdnt {
    fn eq(&self, other: &Bsdnt) -> bool {
        unsafe { return zz_equal(&self.zz, &other.zz); }
    }
}

impl Ord for Bsdnt {
    fn lt(&self, other: &Bsdnt) -> bool {
        let cmp = unsafe { zz_cmp(&self.zz, &other.zz) };
        cmp < 0
    }
}

// This is pretty bad
impl Orderable for Bsdnt {
    fn min(&self, other: &Bsdnt) -> Bsdnt {
        if self < other { self } else { other }.clone()
    }

    fn max(&self, other: &Bsdnt) -> Bsdnt {
        if self > other { self } else { other }.clone()
    }

    fn clamp(&self, min: &Bsdnt, max: &Bsdnt) -> Bsdnt {
        if self < min { min } else if self > max { max } else { self }.clone()
    }
}

impl Add<Bsdnt, Bsdnt> for Bsdnt {
    fn add(&self, other: &Bsdnt) -> Bsdnt { binop_new!(self other zz_add) }
}

impl Sub<Bsdnt, Bsdnt> for Bsdnt {
    fn sub(&self, other: &Bsdnt) -> Bsdnt { binop_new!(self other zz_sub) }
}

impl Mul<Bsdnt, Bsdnt> for Bsdnt {
    fn mul(&self, other: &Bsdnt) -> Bsdnt { binop_new!(self other zz_mul) }
}

impl Div<Bsdnt, Bsdnt> for Bsdnt {
    fn div(&self, other: &Bsdnt) -> Bsdnt {
        if other.is_zero() { fail!("division by 0"); }
        binop_new!(self other zz_div)
    }
}

impl Neg<Bsdnt> for Bsdnt {
    fn neg(&self) -> Bsdnt {
        unsafe {
            let mut res = Bsdnt::new_reserve(self.zz.size as uint);
            zz_neg(&mut res.zz, &self.zz);
            res
        }
    }
}

impl Rem<Bsdnt, Bsdnt> for Bsdnt {
    // TODO This is less efficient than it could be, possibly implement with nn_???
    fn rem(&self, other: &Bsdnt) -> Bsdnt {
        let (_, rem) = self.div_rem(other);
        rem
    }
}

impl Num for Bsdnt { }

impl Signed for Bsdnt {
    fn abs(&self) -> Bsdnt {
        unsafe {
            if zz_cmpi(&self.zz, 0) >= 0 {
                self.clone()
            } else {
                -self
            }
        }
    }

    fn signum(&self) -> Bsdnt {
        let c = unsafe { zz_cmpi(&self.zz, 0) };
        if c < 0 {
            FromPrimitive::from_int(-1)
        } else if c == 0 {
            FromPrimitive::from_int(0)
        } else {
            FromPrimitive::from_int(1)
        }.unwrap()
    }

    fn is_positive(&self) -> bool {
        unsafe { zz_cmpi(&self.zz, 0) > 0 }
    }

    fn is_negative(&self) -> bool {
        unsafe { zz_cmpi(&self.zz, 0) < 0 }
    }

    fn abs_sub(&self, other: &Bsdnt) -> Bsdnt {
        if self <= other { return Zero::zero(); }
        *self - *other
    }
}

impl ToStr for Bsdnt {
    // This makes an unecessary copy
    fn to_str(&self) -> ~str {
        unsafe {
            let cstr = zz_get_str(&self.zz);
            let rstr = std::str::raw::from_c_str(cstr as *c_char);
            std::libc::free(cstr as *mut c_void);
            rstr
        }
    }
}

impl Integer for Bsdnt {
    fn gcd(&self, other: &Bsdnt) -> Bsdnt { binop_new!(self other zz_gcd) }

    fn div_rem(&self, other: &Bsdnt) -> (Bsdnt, Bsdnt) {
        if other.is_zero() { fail!("division by 0"); }
        unsafe {
            let mut quot = Bsdnt::new();
            let mut rem = Bsdnt::new();
            zz_divrem(&mut quot.zz, &mut rem.zz, &self.zz, &other.zz);
            (quot, rem)
        }
    }

    fn div_floor(&self, other: &Bsdnt) -> Bsdnt { binop_new!(self other zz_div) }

    fn mod_floor(&self, _other :&Bsdnt) -> Bsdnt { fail!("TODO mod_floor"); }

    fn div_mod_floor(&self, _other :&Bsdnt) -> (Bsdnt, Bsdnt) { fail!("TODO div_mod_floor"); }

    fn lcm(&self, other :&Bsdnt) -> Bsdnt {
        unsafe {
            let mut tmpa = self.clone();
            zz_mul(&mut tmpa.zz, &tmpa.zz, &other.zz);
            if zz_cmpi(&tmpa.zz, 0) < 0 {
                zz_neg(&mut tmpa.zz, &tmpa.zz);
            }
            let gcd = self.gcd(other);
            zz_div(&mut tmpa.zz, &tmpa.zz, &gcd.zz);
            tmpa
        }
    }

    fn mod_floor(&self, _other :&Bsdnt) -> Bsdnt { fail!("TODO mod_floor"); }

    fn is_multiple_of(&self, other: &Bsdnt) -> bool {
        !other.is_zero() && (*self % *other).is_zero()
    }

    fn is_even(&self) -> bool {
        if self.is_zero() { return true; }
        unsafe { nn_bit_test(*(&self.zz.n) as nn_src_t, 0) == 0 }
    }

    fn is_odd(&self) -> bool { !self.is_even() }
}

impl One for Bsdnt {
    fn one() -> Bsdnt {
        unsafe {
            let mut res = Bsdnt::new();
            zz_seti(&mut res.zz, 1);
            res
        }
    }
}

impl Zero for Bsdnt {
    fn zero() -> Bsdnt { Bsdnt::new() }

    fn is_zero(&self) -> bool {
        let n = unsafe { zz_is_zero(&self.zz) };
        return n == 1;
    }
}

impl FromStr for Bsdnt {
    fn from_str(s: &str) -> Option<Bsdnt> {
        fn is_digit(c: &u8) -> bool { c >= &('0' as u8) && c <= &('9' as u8) }

        if s.len() == 0 { return None; }
        // `zz_set_str` is a bit weird.
        // It uses only the prefix of `s` matching `\-?[0-9]+` and doesn't report errors.
        // So we validate ourselves.
        let mut it = s.as_bytes().iter();
        let first = it.next().unwrap();  // We know it's not empty
        if first != &('-' as u8) && !is_digit(first) { return None; }
        for c in it {
            if !is_digit(c) { return None; }
        }

        let mut ret = Bsdnt::new();
        unsafe {
            s.with_c_str(|cs| { zz_set_str(&mut ret.zz, cs) });
        }
        return Some(ret);
    }
}

macro_rules! from_primitive(
    ($int_ty:ident $val:expr) => (
        std::$int_ty::to_str_bytes($val, 10, |buf| {
            unsafe {
                // `std::str::raw::from_utf8` doesn't inline, use `transmute` instead
                let s = std::cast::transmute(buf);
                FromStr::from_str(s)
            }
        })
    )
)

impl FromPrimitive for Bsdnt {
    fn from_i64(n: i64) -> Option<Bsdnt> { from_primitive!(i64 n) }

    fn from_u64(n: u64) -> Option<Bsdnt> { from_primitive!(u64 n) }
}

impl Clone for Bsdnt {
    fn clone(&self) -> Bsdnt {
        unsafe {
            let mut ret = Bsdnt::new();
            zz_set(&mut ret.zz, &self.zz);
            ret
        }
    }

    fn clone_from(&mut self, source: &Bsdnt) {
        unsafe { zz_set(&mut self.zz, &source.zz); };
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::num::{Zero, One};  // Two, Three, ...}

    #[test]
    fn test_clone_from() {
        let mut x: Bsdnt = FromPrimitive::from_i64(4000).unwrap();
        let y: Bsdnt = FromPrimitive::from_i64(19000).unwrap();
        assert!(x != y);
        x.clone_from(&y);
        assert!(x == y);
    }

    #[test]
    fn test_clone() {
        let mut x: Bsdnt = from_str("-31232313").unwrap();
        let y = x.clone();
        let z: Bsdnt = from_str("3000").unwrap();
        assert!(x == y);
        x = x + z;
        assert!(x != y);
    }

    #[test]
    fn test_from_str_valid() {
        let x: Bsdnt = FromPrimitive::from_i64(4000).unwrap();
        let y: Bsdnt = FromPrimitive::from_i64(-19000).unwrap();
        let xi: Bsdnt = from_str("4000").unwrap();
        let yi: Bsdnt = from_str("-19000").unwrap();
        assert!(x != yi);
        assert!(y != xi);
        assert!(x == xi);
        assert!(y == yi);
    }

    #[test]
    fn test_from_str_invalid() {
        let xi = from_str::<Bsdnt>("4000hej");
        assert!(xi.is_none());
    }

    #[test]
    fn test_ord() {
        let x: Bsdnt = from_str("-31232313").unwrap();
        let y: Bsdnt = from_str("123432843254375345743857834").unwrap();
        let z: Bsdnt = from_str("321382138321312333333901381").unwrap();
        assert!(x < y && x < z && y < z);
        assert!(x <= x && x <= y && x <= z && y <= z);
        assert!(z > y && z > x && y > x);
        assert!(z >= z && z >= y && z >= x && y >= x);
    }

    #[test]
    fn test_zero_init() {
        let x = Bsdnt::new();
        let y: Bsdnt = from_str("0").unwrap();
        let z: Bsdnt = Zero::zero();
        assert!(x == y);
        assert!(z == y);
    }

    #[test]
    fn test_one() {
        let x: Bsdnt = FromPrimitive::from_i64(1).unwrap();
        assert!(x == One::one());
    }

    #[test]
    #[should_fail]
    fn test_div_zero() {
        let x: Bsdnt = Zero::zero();
        x / x;
    }

    #[test]
    #[should_fail]
    fn test_rem_zero() {
        let x: Bsdnt = Zero::zero();
        x % x;
    }

    #[test]
    fn test_neg() {
        let x: Bsdnt = from_str("32982908").unwrap();
        let y: Bsdnt = from_str("-32982908").unwrap();
        assert!(x.neg() == y);
    }

    #[test]
    fn test_even() {
        let x: Bsdnt = from_str("8").unwrap();
        let y: Bsdnt = from_str("9").unwrap();
        let z: Bsdnt = from_str("-8").unwrap();
        let p: Bsdnt = from_str("-9").unwrap();
        let q: Bsdnt = from_str("0").unwrap();
        let r: Bsdnt = from_str("18446744073709551615123213139").unwrap();
        let s: Bsdnt = from_str("184467440737095516151232131392").unwrap();
        assert!(x.is_even());
        assert!(y.is_odd());
        assert!(z.is_even());
        assert!(p.is_odd());
        assert!(q.is_even());
        assert!(r.is_odd());
        assert!(s.is_even());
    }

    #[test]
    fn test_lcm() {
        let x: Bsdnt = FromPrimitive::from_i64(4).unwrap();
        let y: Bsdnt = FromPrimitive::from_i64(6).unwrap();
        let z: Bsdnt = FromPrimitive::from_i64(12).unwrap();
        assert!(x.lcm(&y) == z);

        let p: Bsdnt = FromPrimitive::from_i64(-5).unwrap();
        let q: Bsdnt = FromPrimitive::from_i64(2).unwrap();
        let r: Bsdnt = FromPrimitive::from_i64(10).unwrap();
        assert!(p.lcm(&q) == r);
    }

    #[test]
    fn test_gcd() {
        let x: Bsdnt = FromPrimitive::from_i64(163231).unwrap();
        let y: Bsdnt = FromPrimitive::from_i64(135749).unwrap();
        assert!(x.gcd(&y) == FromPrimitive::from_i64(151).unwrap());
    }

    #[test]
    fn test_abs() {
        let x: Bsdnt = from_str("32982908").unwrap();
        let y: Bsdnt = from_str("-32982908").unwrap();
        assert!(x == x.abs());
        assert!(x == y.abs());
    }

    #[test]
    fn test_is_multiple_of() {
        let x: Bsdnt = from_str("-9").unwrap();
        let y: Bsdnt = from_str("3").unwrap();
        let z: Bsdnt = from_str("7").unwrap();
        assert!(x.is_multiple_of(&y));
        assert!(!z.is_multiple_of(&y));
        assert!(!z.is_multiple_of(&Zero::zero()));
    }

    #[test]
    fn test_to_str() {
        let a = "32982908";
        let b = "-32982908";
        let c = "0";
        let x: Bsdnt = from_str(a).unwrap();
        let y: Bsdnt = from_str(b).unwrap();
        let z: Bsdnt = from_str(c).unwrap();
        assert!(a == x.to_str());
        assert!(b == y.to_str());
        assert!(c == z.to_str());
    }

    #[test]
    fn test_from_primitive() {
        let a: i64 = 9223372036854775807;
        let b = "9223372036854775807";
        let x: Bsdnt = FromPrimitive::from_i64(a).unwrap();
        let y: Bsdnt = from_str(b).unwrap();
        assert!(x == y);
        assert!(b == x.to_str());
    }

    #[test]
    fn test_from_u64() {
        let a: u64 = 18446744073709551615;
        let b = "18446744073709551615";
        let x: Bsdnt = FromPrimitive::from_u64(a).unwrap();
        let y: Bsdnt = from_str(b).unwrap();
        assert!(x == y);
        assert!(b == x.to_str());
    }
}

#[cfg(test)]
mod bench {
    extern mod extra;

    use super::*;
    use std::iter::range_inclusive;
    use std::num::{One};

    static bignum: &'static str = "347329483248324987312897398216945234732489236493274398127428913\
                                   382190389201839813919208390218903821093219038213128074395657862\
                                   321832190873902183092183092183902810974012743284732894723894790\
                                   312381290389201389021839021803821903892018437549835743897589347\
                                   43289483290489302849032753298573458943758974358974398578943759";

    fn black_box<T>(dummy: T) {
        unsafe { asm!("" : : "r"(&dummy)) }
    }

    fn factorial<T: Integer+FromPrimitive>(n: uint) -> T {
        let mut f: T = One::one();
        for i in range_inclusive(1, n) {
            f = f * FromPrimitive::from_uint(i).unwrap();
        }
        f
    }

    #[bench]
    fn bench_factorial100(b: &mut extra::test::BenchHarness) {
        b.iter(|| { factorial::<Bsdnt>(100); });
    }

    #[bench]
    fn bench_gcd_big_small(b: &mut extra::test::BenchHarness) {
        let x: Bsdnt = from_str(bignum).unwrap();
        let y: Bsdnt = from_str("19").unwrap();
        b.iter(|| { x.gcd(&y); });
    }

    #[bench]
    fn bench_to_str(b: &mut extra::test::BenchHarness) {
        b.iter(|| {
            let n = from_str::<Bsdnt>(bignum);
            black_box(n);
        });
    }
}
