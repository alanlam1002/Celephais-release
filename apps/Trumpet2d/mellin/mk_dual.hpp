// Mellin throat recursion (code round 463; research rounds 1074 / 1089, the forward-mode replay, STAGE 1): the graded series algebra on DUAL
// series -- a value (batch 1) and its tangents along ND directions (one batch copy each: the series algebra's batch axis broadcasts the value
// against them).  Every rule is composed from mk_series' own operations, so the tangent is the derivative of the computation AS EXECUTED:
//   add / sub / scale        linear
//   mul                      d(a b) = a db + da b
//   drad / dth               linear
//   inv                      d(1/a) = -(1/a)^2 da
//   expser                   d exp(a) = exp(a) da
//   logser                   d log(a) = da / a           (the leading exponent and r-power are decisions: frozen)
//   trunc                    the value's validity
//   key_tangent              a piece whose CENTRE moves with the data (g = g(p)): d/dp [x^g sum_m v_m (ln x)^m / m!] adds dg m v_(m-1) to
//                            moment m (the binomial shift's derivative at dl = 0; moment K dropped, as the shift drops it)
// The DECISIONS are the value's: the keys (centres), the validity (top) and the registry.  A tangent is truncated to its value's validity after
// every operation, and its pieces live on the value's keys (a tangent piece on a key the value pruned as an exact zero is kept: the key's
// value is zero, its derivative need not be).  In the exact-merge mode (production) the tangent operations intern no centre the value does not.
// Unit-tested against central finite differences of the same operations (mk_dual_test.cpp).
#pragma once
#include "mk_series.hpp"

namespace mk {

struct DSer {
    Ser v;                                  // the value (batch 1)
    Ser t;                                  // the tangents (batch ND; empty = zero tangent)
};

struct Dual {
    Ctx& C;
    int ND;
    explicit Dual(Ctx& c, int nd) : C(c), ND(nd) {}

    // the tangent restricted to the value's validity (the value's decision)
    Ser fit(const Ser& t, const Ser& v) {
        if (t.empty()) return C.emptyser(v.top);
        return C.trunc(t, v.top);
    }
    Ser tadd(const Ser& a, const Ser& b) {                  // tangent sums (an empty side is zero, whatever its r-power)
        if (a.empty()) return b;
        if (b.empty()) return a;
        return C.add(a, b);
    }
    DSer constant(const Ser& v) { return DSer{v, C.emptyser(v.top)}; }
    // a seed: value v, tangent along direction k = dv (batch 1), zero along the others
    DSer seed(const Ser& v, int k, const Ser& dv) {
        std::vector<std::pair<Key, Piece>> d;
        for (auto& kv : dv.d) {
            Piece p = C.zeros(ND);
            std::copy(kv.second.a.begin(), kv.second.a.end(), p.a.begin() + size_t(k) * C.psz());
            d.emplace_back(kv.first, std::move(p));
        }
        Ser t = C.make(std::move(d), dv.has_e, dv.e, dv.top);
        return DSer{v, fit(t, v)};
    }

    DSer add(const DSer& a, const DSer& b) {
        Ser v = C.add(a.v, b.v);
        return DSer{v, fit(tadd(a.t, b.t), v)};
    }
    DSer scale(const DSer& a, cd c) {
        Ser v = C.scale(a.v, c);
        return DSer{v, fit(a.t.empty() ? a.t : C.scale(a.t, c), v)};
    }
    DSer sub(const DSer& a, const DSer& b) { return add(a, scale(b, -1.0)); }
    DSer mul(const DSer& a, const DSer& b) {
        Ser v = C.mul(a.v, b.v);
        Ser t1 = b.t.empty() || a.v.empty() ? Ser() : C.mul(a.v, b.t);
        Ser t2 = a.t.empty() || b.v.empty() ? Ser() : C.mul(a.t, b.v);
        return DSer{v, fit(tadd(t1, t2), v)};
    }
    DSer drad(const DSer& a) {
        Ser v = C.drad(a.v);
        return DSer{v, fit(a.t.empty() ? a.t : C.drad(a.t), v)};
    }
    DSer dth(const DSer& a) {
        Ser v = C.dth(a.v);
        return DSer{v, fit(a.t.empty() ? a.t : C.dth(a.t), v)};
    }
    DSer inv(const DSer& a) {
        Ser v = C.inv(a.v);
        if (a.t.empty()) return constant(v);
        Ser t = C.scale(C.mul(C.mul(v, v), a.t), -1.0);
        return DSer{v, fit(t, v)};
    }
    DSer expser(const DSer& a) {
        Ser v = C.expser(a.v);
        if (a.t.empty()) return constant(v);
        return DSer{v, fit(C.mul(v, a.t), v)};
    }
    DSer logser(const DSer& a) {
        Ser v = C.logser(a.v);
        if (a.t.empty()) return constant(v);
        return DSer{v, fit(C.mul(C.inv(a.v), a.t), v)};
    }
    DSer trunc(const DSer& a, double top) {
        Ser v = C.trunc(a.v, top);
        return DSer{v, fit(a.t, v)};
    }
    // the tangent of a piece at a MOVING centre: value piece v at key g with dg/dp_k = dg[k] (k < ND) -> the tangent piece (batch ND)
    Piece key_tangent(const Piece& v, const std::vector<double>& dg) const {
        if (v.B != 1) throw std::runtime_error("mk: key_tangent of a batched value");
        Piece o = C.zeros(ND);
        const int K = C.K, NF = C.NF;
        for (int k = 0; k < ND; k++) {
            if (dg[k] == 0.0) continue;
            for (int m = 1; m < K; m++)
                for (int q = 0; q < NF; q++) o.a[(size_t(k) * K + m) * NF + q] += dg[k] * double(m) * v.a[size_t(m - 1) * NF + q];
        }
        return o;
    }
    // the value's piece at key g and the tangent's (zeros if the tangent has none there)
    Piece tan_at(const DSer& a, const Key& g) const {
        int i = a.t.find(g);
        return i >= 0 ? a.t.d[i].second : C.zeros(ND);
    }
};

}  // namespace mk
