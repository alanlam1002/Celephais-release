/*
 * finiteJ_probe_main.cpp -- A1 rung 0: do the SECTION-4 equations mean, inside
 * Kadath, what they mean in sympy?
 *
 * Before any manufactured solution, the chain that carries it has to be checked
 * against an answer already known: the parser accepting 1.2k-3.6k character
 * strings, dr and dt acting on a Domain_polar_shell, add_cst fields behaving as
 * fields under differentiation, and the emitted text meaning what the sympy
 * expression meant.  Four inherited-and-unexercised pieces, and this project's
 * base rate on those is four wrong in six.
 *
 * THE KNOWN ANSWER is the J = 0 Schwarzschild maximal trumpet, which round 102
 * showed annihilates all six equations symbolically:
 *
 *     PS = R/r,  PH = W R/r,  QF = 0,  BR = C r/R^3,  BT = 0,  QB = 0,  JJ = 0
 *
 * with C = 3 sqrt3 M^2 / 4.  So every def must evaluate to zero here, down to
 * the discretisation floor -- and the same defs on a PERTURBED seed must not,
 * which is the control that stops this from being a test that cannot fail.
 */

#include <algorithm>

#include <mpi.h>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include "space/space_polar_trumpet.hpp"
// ⚠ TWO EMISSIONS, ONE BINARY EACH.  The generated headers declare the same
// names in namespace Trumpet, so they cannot both be included; and regenerating
// finiteJ_eqs.hpp in place would make every measurement path in the record
// seven-unknown overnight.  TRUMPET_CHI_UNKNOWN selects the seven-unknown
// emission and builds a SEPARATE target, so the six-unknown pipeline -- which is
// the whole record and the permanent cross-check -- is untouched.
#if defined(TRUMPET_CHI_UNKNOWN) && defined(TRUMPET_Q_REGULAR)
#error "TRUMPET_CHI_UNKNOWN and TRUMPET_Q_REGULAR are separate emissions"
#endif
#ifdef TRUMPET_CHI_UNKNOWN
#include "src/finiteJ_eqs_chi.hpp"
#elif defined(TRUMPET_BULK2)
// ⚠ Round 335 (research round 664, step 2): the q-regular D_r V^th emission with D0081 re-emitted as (r^2 K)^2
// (scripts/finiteJ_emit.py --q-regular --r-into-square D0081, TH2_DRVT_FIX=1).  Its own binary only.
#include "src/finiteJ_eqs_qreg_drvt_r2k.hpp"
#elif defined(TRUMPET_Q_REGULAR) && defined(TRUMPET_DRVT_FIX)
// ⚠ Round 307: the q-regular emission with D_r V^th = d_r V^th + V^th/r
// (scripts/throat_th2.py, TH2_DRVT_FIX=1).  Its own binary only.
#include "src/finiteJ_eqs_qreg_drvt.hpp"
#elif defined(TRUMPET_Q_REGULAR)
// ⚠ Round 279: q = sin^2(theta) q~ in BOTH emissions; the unknown QF is q~.
#include "src/finiteJ_eqs_qreg.hpp"
#else
#include "src/finiteJ_eqs.hpp"
#endif
#if defined(TRUMPET_Q_REGULAR) && defined(TRUMPET_DRVT_FIX_THROAT)
// ⚠ Round 309: the throat rows emitted from deep5/deep6 regenerated with TH2_DRVT_FIX=1
// (throat_emit.py --q-regular --grade1-replace --maximality --grade2-u2).  Own binary only.
#include "src/throatth_eqs_qreg_drvt.hpp"
#elif defined(TRUMPET_Q_REGULAR)
#include "src/throatth_eqs_qreg.hpp"
#else
#include "src/throatth_eqs.hpp"
#endif
#include "Trumpet1d/src/table_io.hpp"

#include <algorithm>
#include <limits>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using Kadath::Array;
using Kadath::Dim_array;
using Kadath::Index;
using Kadath::Point;
using Kadath::Scalar;
using Kadath::System_of_eqs;
using Kadath::Val_domain;

// ⚠ eq_list / eq_int_list are PROTECTED in System_of_eqs and have no accessor.
// A pointer-to-member formed through a derived class reads them without
// touching the library (round 279's --dump-rowmeta needs eq_index -> name).
struct EqListPeek : Kadath::System_of_eqs {
    using L = std::vector<std::tuple<std::string, int, int>>;
    static const L& eqs(const Kadath::System_of_eqs& s)
    { return s.*(&EqListPeek::eq_list); }
    static const L& eqints(const Kadath::System_of_eqs& s)
    { return s.*(&EqListPeek::eq_int_list); }
};

static void emit(const std::string& k, double v)
{
    std::cout << "RESULT " << k << " " << std::setprecision(17) << v << "\n";
}

int main(int argc, char** argv)
{
#ifdef TRUMPET_Q_REGULAR
    // ⚠ Round 236's rule: a build that changes the assembled system says so.
    std::cout << "#  TRUMPET_Q_REGULAR: q = sin^2(theta) q~; the unknown QF is q~"
                 " (and every dumped QF column is q~, not q)\n";
    emit("FJP_q_regular", 1);
#endif
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc < 2) {
        if (rank == 0)
            std::cerr << "usage: finiteJ_probe <backbone.dat> [--ntheta N] "
                         "[--perturb X]\n";
        MPI_Finalize();
        return 2;
    }
    int ntheta = 5;
    double perturb = 0.0;
    bool monolithic = false;
    // --clean: register ONLY the emitted sub-defs and equations.  The smoke,
    // vocabulary and parser-length batteries above them are diagnostics, not
    // part of the object under test, and they put ~45 extra defs -- five of
    // them 100 to 120 operations deep, at a stack limit known not to be
    // reproducible -- into the same System_of_eqs before the acceptance test
    // reads it.  Holding them out is the first thing a bisection on
    // registration order has to do, or "order" and "what else is registered"
    // are varied together.
    bool clean = false;
    // --extra-def "NAME = TEXT", repeatable: registered AFTER everything else
    // and reported at the probe point.  A re-spelling of a sub-def that is
    // already wrong, registered at a different position in the same system,
    // separates "this text evaluates wrong" from "this text evaluates wrong
    // HERE" -- which is the question a bisection on registration order asks.
    std::vector<std::string> extra;
    // --maxdefs K: register only the first K sub-defs (and no equations).  If a
    // sub-def is right at K = its own index and wrong at K = 309, then a LATER
    // registration changed it, and bisecting K names the one that did.
    int maxdefs = -1;
    // --extra-early: register the extras BEFORE anything is read back, instead
    // of after.  The difference between the two isolates whether reading a
    // def's operands is what makes a later sum of them come out right.
    bool extra_early = false;
    // --touch: read every sub-def back on every domain immediately after
    // registering it.  Reading a def is supposed to be an observation; if it
    // changes what a LATER def computes from it, it is not.
    bool touch = false;
    // --read-before NAME, repeatable: read exactly these defs, on the probe
    // domain only, immediately before the --extra-early registrations.  Naming
    // the reads one at a time is how the trigger gets isolated.
    std::vector<std::string> readbefore;
    bool readall = false;
    // --break-contract: register the emission WITHOUT the reads the header
    // requires.  It exists so finiteJ_check() can be shown to catch the breach;
    // a control that has never been seen to fail is not a control.
    bool breakcontract = false;
    // --break-basis: register QB in the basis the emission was NOT analysed
    // under, so the declaration check can be seen to refuse.  --no-basis-check
    // then skips the refusal, which is the only way to read the residuals under
    // the wrong basis; the two together say both what the check does and what
    // the seed can see.
    bool breakbasis = false, nobasischeck = false;
    // --table-noise EPS: multiply the tabulated R and W by 1 + EPS*u, u a
    // deterministic pseudo-random number in [-1, 1].  The residual's response
    // to a perturbation of KNOWN size is the amplification factor, and the
    // floor is then predicted rather than attributed: A times the table's own
    // error, which a 50-digit recomputation puts at 1e-15.
    double tablenoise = 0.0;
    // --seed-perturb EPS --pfield NAME: add a perturbation of KNOWN sup-norm EPS
    // to one unknown (or to all six), in that field's own angular basis, and
    // report the residual it produces.  The ratio is the operator's gain in
    // that direction, and 1/gain turns a residual into a solution-error bound.
    //
    // ⚠ The profile is generic in BOTH directions -- r-dependent and
    // theta-dependent, several harmonics -- because round 103's constant
    // coefficients made every r-derivative vacuous and round 93's unphysical
    // field gave the opposite answer.  And it is built in the field's DECLARED
    // basis: a perturbation in the wrong one would re-create a mixed-basis sum
    // and measure round 111's defect instead of the conditioning.
    double seedperturb = 0.0;
    std::string pfield = "all";
    // --pmix "w1,..,w6": weight the six fields' perturbations independently.
    // Six single-field gains bound the operator's gain from ABOVE; a
    // combination can cancel and be much smaller, and a near-null direction
    // would void any residual-to-solution bound.  This samples combinations.
    std::string pmix;
    // --dump-grid FILE: write the collocation grid Kadath actually built, so
    // the manufactured generator never re-derives a collocation formula.  A
    // duplicated formula is a second place to be wrong, and this one would be
    // wrong silently.
    std::string gridout;
    std::string defsout;   // --dump-defs FILE
    // ⚠ ROUND 157: the six FIELD values, which --dump-defs does not carry (it
    // dumps sub-defs).  The log detector needs q itself, not an equation.
    std::string fieldsout;  // --dump-fields FILE
    // --dump-fields-post: the same dump, AFTER --newton-delta has been applied.
    // ⚠ --dump-fields fires at its own site well before the delta is read, so
    // on a converged chain it writes the SEED and not the state.  The throat's
    // intrinsic geometry (round 252) has to be read off the converged fields,
    // so the dump is repeated at the one place where PS..QB carry the state:
    // immediately after xx_to_vars_delta.
    std::string fieldspost;  // --dump-fields-post FILE
    // --dump-eqvals FILE: every equation residual at every collocation point,
    // so the readout can be a PER-MODE NORM instead of a max.  ⚠ Research round
    // 521: round 212 read FJP_EQTW as a max and round 219 found std::max(x,NaN)
    // returns x, so a max can neither see a NaN nor distinguish "zero" from
    // "cancelling at the grid maximum".  The theta-mode decomposition is done
    // offline against the collocation angles written here.
    std::string eqvalsout;  // --dump-eqvals FILE
#ifdef TRUMPET_CHI_UNKNOWN
    // --chi-seed FILE: plant CH from a --dump-eqvals file's `D P48 ...` lines,
    // i.e. from D0048 evaluated ON A STATE by an earlier run.
    // ⚠ WHY A FILE AND NOT A RE-PLANT IN PLACE.  Round 263 re-planted CH from
    // D0048 after xx_to_vars_delta and it was a no-op; round 264 measured why.
    // The plant DOES reach the Scalar -- a direct Val_domain comparison gives
    // max|CH - D0048| = 0 exactly -- but ECHI still reads the old value, because
    // add_var makes the system hold its own Term_eq copy and neither
    // vars_to_terms() nor std_base() propagates a post-registration write into
    // the already-built def tree.  Seeding BEFORE add_var is the path the system
    // honours, and a file is how a state-dependent value gets there: run once to
    // dump D0048 on the state, run again with --chi-seed.
    std::string chiseed;    // --chi-seed FILE
#endif
    // --jac-balance: impose the BALANCE at every harmonic (research round 528).
    // The shooting reads `E_Phi/Ph + 2 E_sigma` at ell = 0 only; its ell >= 2
    // harmonics are Gate 1, which round 127's state satisfies to 1e-10 and
    // this one violates by +2.3.  The emitted row is division-free,
    //     TBAL = E_Phi(n-3) - Ph * E_q(-2),
    // and it is imposed as `TBAL - SPH * SBAL = 0` with SBAL ONE new scalar
    // unknown: TBAL is Ph times the balance, so `TBAL = Ph * SBAL` is exactly
    // `balance = SBAL` pointwise, while `TBAL = const` would NOT be (Ph is
    // theta-dependent and multiplying by it mixes harmonics).  nt rows against
    // one new column is a NET nt-1 conditions -- the ell >= 2 harmonics -- and
    // SBAL comes back as the shooting residual, read out of the system
    // instead of evaluated offline.
    // --drop-tephi: free TEPHI's ell >= 2 content, keeping only its ell = 0.
    // ⚠ NOT BY REMOVING ROWS.  add_eq_bc projects onto all nt tau modes and
    // there is no way to ask it for one; but giving the ell >= 2 harmonics
    // their own free coefficients has the same effect and IS expressible:
    //     TEPHI - (WP01*MCE01 + ... + WP{nt-1}*MCE{nt-1}) = 0
    // still assembles nt rows, nt-1 of which now merely DEFINE the WP's, so
    // the only condition left on the state is the ell = 0 harmonic.  Rows are
    // unchanged and nt-1 columns are added, which takes the --jac-balance
    // system from over-by-(nt-1) to SQUARE.
    // Justified by round 234's attribution -- TEPHI carries 22x and 34x its
    // share of the excess left-null weight and the balance exactly 0x -- and
    // NOT by a rank, which is not a number at this conditioning.
    bool jacbalance = false;   // --jac-balance
    bool droptephi = false;    // --drop-tephi
    // --jac-axisres: the AXIS RESIDUE of E_sh^theta, as one scalar row
    // (research round 541).  module2_final.py closes its 8x8 ell = 2 solve
    // with the vanishing of the 1/theta pole of the theta-momentum constraint
    // at the axis; the global multiplies E_sh^theta by sin^2 theta (round
    // 514), which clears the pole, and a pole multiplied away is a condition
    // removed, not imposed.
    // ⚠ NO AXIS EVALUATION IS NEEDED, because the throat amplitudes are
    // explicit sums over the SCALAR unknowns G%02d / H%02d times pure angular
    // profiles (throatth_amp_defs).  A Laurent expansion of deep5's UNWEIGHTED
    // E_sh^theta(n-2) about theta = 0 -- theta^-3 and theta^-2 vanish
    // identically, so the pole is exactly simple -- gives
    //     Res = -2 Bh'(0) - n bh(0) [ b1(0) + 6 u1(0) ]
    // and each axis datum is a plain sum of scalars: x(0) = sum_j X_j for the
    // COS_EVEN/COS_ODD families, Bh'(0) = sum_j 2(j+1) G_BT_j for SIN_EVEN.
    // Multiplied through by PS's axis value to clear the one division, the
    // row is the division-free
    //     TRES = -(sum G_PS) [ 4 sum_j (j+1) G_BT_j + n sum H_BR ]
    //            - 3 n (sum G_BR)(sum H_PS)  =  0
    // ONE condition, theta-independent: the residue is a number, not a
    // harmonic, so unlike --jac-balance there is no new unknown and no mode
    // count.  Rows +1, columns +0.
    // ⚠ Validated before it was built, three ways: b1 + 6 u1 == 0 identically
    // in n on round 176/224's closed forms (so the J = 0 backbone satisfies it
    // exactly, not to a tolerance); the five pole-column coefficients match
    // module2_final.py's independent 60-dps numerical extraction exactly,
    // including the irrational sqrt2/3 on c2s; and the combination vanishes on
    // module2's own regular-branch solution.
    bool jacaxisres = false;   // --jac-axisres
    // --grade1-replace: THE ell >= 2 GRADE-1 REPLACEMENT (research rounds
    // 543/575, code rounds 248/250).  Per ell >= 2 mode the six recursion
    // rows are rank 3 of 5 on the grade-1 unknowns, because E_Phi(-3) and
    // E_q(-n-2) are VACUOUS there; min-norm has been setting that two-per-mode
    // freedom to zero.  The fix is a REPLACEMENT, net row change zero:
    //     TEPHI, TEQFN      mode 0 only                      (1 + 1)
    //     TESHR2 = E_sh^r(2n-1), TESIG2 = E_sigma(n-2)
    //                       modes 1..nt-1                   (nt-1 + nt-1)
    // = 2nt, exactly what TEPHI and TEQFN supply at every mode now, so the
    // count stays 6nt - 3.  ⚠ add_eq_bc cannot take a subset of modes, so
    // every one of these goes through add_eq_mode, one coefficient per call
    // (Ope_mode reads the boundary value's cos(2k theta) coefficient k, the
    // same coefficient add_eq_bc's tau projection imposes for a COS_EVEN row).
    // Those are Eq_int rows and land AFTER the field equations in the row
    // order; the count check below is what says the system is still square.
    bool g1replace = false;    // --grade1-replace
    // --balance-l2 (research round 584): THE BALANCE RE-IMPOSED AT ell >= 2, in
    // place of TEPHI's modes 1..nt-1, which are vacuous at linear order.  It is
    // --jac-balance's registration (TBAL - Ph*SBAL at all nt COS_EVEN modes,
    // SBAL one trailing scalar unknown, so the ell = 0 balance comes back as
    // SBAL: READ, not imposed) with TEPHI cut to mode 0 by add_eq_mode.
    //     rows  +nt (TBAL) - (nt-1) (TEPHI's ell >= 2)  = +1
    //     cols  +1  (SBAL)                                     -> SQUARE
    // Round 555 demoted the ell >= 2 balance to a read because imposing it on
    // the floor-contaminated, axis-irregular states broke the outer loop; the
    // floor is fixed by --rhs-subtract and the axis by q = sin^2 q~, so this is
    // meant for finiteJ_probe_qreg and refuses --grade1-replace.
    bool balancel2 = false;    // --balance-l2
    double* sbal_ptr = nullptr;
    // --maximality (research rounds 590/594/596): THE THROAT'S GRADE-n
    // MAXIMALITY M(theta), the r^n coefficient of D_i beta^i on throat_th2's
    // ansatz, emitted by throat_emit.py --maximality from
    // analysis_na/maximality_gradeN.py as TMAX, with B_h = BTH = SBT (the G_BT
    // columns, beta^theta's LEADING amplitude) and u1, b1, q1 from H_PS, H_BR,
    // sin^2 HQF.  Imposed at modes 1..nt-1 by add_eq_mode IN PLACE OF TEQFN's
    // modes 1..nt-1 (identically zero at J = 0); TEQFN keeps mode 0.
    //     rows  -(nt-1) + (nt-1) = 0,  cols +0            -> still SQUARE
    // Only on the balanced q-regular binary: needs --balance-l2, refuses
    // --grade1-replace (which also rewrites TEQFN) and --jac-axisres.
    bool maximality = false;   // --maximality
    // --physical-h (research round 622; code round 294): THE SEED'S GRADE-1
    // NORMALISATION.  Round 176's closed forms take w0 = t.W0 = alpha(r_m),
    // i.e. they are written in rho = (r/r_m)^n; throat_th2, the emitted rows,
    // RECON and the NPS family (r^{g0+n}) are all rho = r^n.  So the seed's H
    // is the physical H times r_m^n (the backbone's own grade-1 coefficient is
    // 1610.5/1611.4/1612.6 x the seed's; 1/r_m^n = 1610.2), and G = F(r_m)/r_m^g0
    // carries the grade-1 term at r_m.  This flag seeds H x r_m^-n and moves
    // the grade-1 term out of G: G <- G - H r_m^n (scripts/rescale_h.py's
    // offline correction, done at the seed).  Seed only: rows, Jacobian and
    // RECON are untouched, and without the flag nothing changes.
    bool physh = false;        // --physical-h
    // --grade2-u2 (research rounds 622/623; code round 295): THE MINIMAL
    // GRADE-2 CLOSURE.  With physical H the rows TESIG, TBAL carry the grade-2
    // amplitude u2 of psi^2 at O(1) (and, through O(j^2), only u2 and H_BT).
    // u2 becomes a THROAT-ONLY unknown U2 = sum_k UUk ACEk (COS_EVEN, nt
    // modes; not a matching unknown), E_sigma(-2) with u2 is imposed at all nt
    // modes as TSIGU, and TESIG, TBAL gain their u2 parts DTSIG, DTBAL
    // (throat_emit.py --grade2-u2).  P2, q2, b2 truncated.
    //     rows +nt (TSIGU), cols +nt (UU)                      -> still SQUARE
    // Seeded at u2 = -5 W^2/24, W = G_PH/G_PS (the J = 0 closed form).  Needs
    // --physical-h, --balance-l2, --maximality; refuses --grade1-replace,
    // --jac-axisres, --drop-tephi, --jac-recursion-early.
    bool g2u2 = false;         // --grade2-u2
    // ⚠ AND A PLANTED LOGARITHM, so the detector can be shown to FIRE.  Round
    // 154 grouped three defects of one family -- checks whose failure mode was
    // unreachable -- and a detector that has never recovered a known signal is
    // the fourth.  --qlog C adds C log(r) to q, which is theta-independent and
    // so lands entirely in the m = 0 mode the detector fits.
    double qlog = 0.0;      // --qlog C
#ifdef TRUMPET_BULK2
    // ⚠ ROUND 335 (research round 664, step 2): the bulk changes, each OFF by default and announced when registered.
    bool eqfnsin2 = false;          // --eqfn-sin2: EQFN imposed as multsint(multsint(EQFN)) (sin^2-weighted tests)
    std::string amaxtable;          // --amax-table FILE: domain-0 rows x W<eq>(r) (no underscore: Kadath reads _ as an index) = 1 / (largest d_rr coefficient)
    bool komar = false;             // --komar: integ(KG) = 4 pi M (M = 1) on the domain-1/2 interface, mode 0
    double komar_m = 1.0;           // --komar-m M
#endif
    std::string jacdump;   // --dump-jacobian FILE (BULK rows only)
    // --dump-rowmeta FILE: Kadath's own per-row metadata (FILE.rows, FILE.cols,
    // dump_tagged_jacobian_metadata_csv) and the eq_index -> expression table
    // (FILE.eqs), so a row is attributed to an EQUATION by the library that
    // assembled it -- round 258's classifier failure was a map built outside.
    std::string rowmeta;
    // ⚠ STEP 1 OF THE BVP BUILD (round 134): the row/column counts must be
    // ATTRIBUTED to named fields and equations, not inferred from the total.
    // These two let a SUBSET be registered, so the per-field column count and
    // the per-equation row count are read off directly instead of divided out
    // of 528*ntheta - 176, which is not even divisible by six.
    std::string jacfields = "all";   // --jac-fields PS,PH,...
    std::string jaceqs = "all";      // --jac-eqs EQTW,ESHR,...
    // ⚠ STEP 2 OF THE BVP BUILD (round 135): interface matching rows ONLY, no
    // physics boundary row.  The system stays RECTANGULAR and nothing is
    // claimed; the point is to exercise the interface machinery against a count
    // known in advance -- 3 interfaces x 2 (value and derivative) x the field
    // angular modes 6*ntheta - 3 = 36*ntheta - 18.
    bool jacinterfaces = false;      // --jac-interfaces
    // ⚠ STEP 3 (round 136): the two outer rows, t_Q = 0 and 2 t_U + t_G = 0.
    // Prediction, stated before running: Phb and qf are both COS_EVEN, so
    // ntheta modes each and 2*ntheta rows.
    bool jacouter = false;           // --jac-outer
    // ⚠ STEP 4's INNER BLOCK (research round 397 ruling 3).  C0 and C1 at
    // r_match for six fields.  The prediction, stated before running, is
    // 12*ntheta - 6 rows -- two conditions per field per ANGULAR MODE, and the
    // mode counts are not ntheta for all six (BT is SIN_EVEN -> nt-2, QB is
    // COS_ODD -> nt-1), which is exactly the correction step 2 needed.
    // ⚠ THIS REGISTERS ROWS ONLY.  The matching's new unknowns -- the throat
    // system's free data -- are NOT registered, because their number is not
    // settled: round 398 measured the deficient row-subspace as SKEW to the
    // parity blocks, so the tower's two functions carry no angular basis to
    // count.  The deficit reported here is therefore the ROW count against the
    // bulk, not the inner block's net contribution, and it is labelled as such.
    bool jacinner = false;           // --jac-inner
    // ⚠ THE BRACKET EXPERIMENT (round 185).  Round 184 changed the BACKGROUND
    // while holding the matching structure fixed and the grading persisted.
    // This changes the STRUCTURE while holding the background fixed.
    //   full  C0 and C1 for six fields         12nt - 6   (the default)
    //   half  C0 only, six fields               6nt - 3   -- research's literal
    //         spec; ⚠ with no matching unknowns this makes the system SQUARE,
    //         so there is no kernel and nothing to grade.  Run anyway, because
    //         "there is nothing to measure" is itself the answer to that form.
    //   net   dr(F) = 0 for PS, PH, QF, BR only     4nt   -- the block that
    //         carries no unknown in the posed system, i.e. the inner block's
    //         NET contribution (round 172).  With no matching unknowns this
    //         reproduces the deficit 2nt - 3 EXACTLY by a different mechanism,
    //         which is the comparison that can actually be made.
    std::string jacinnermode = "full";   // --jac-inner-mode full|half|net|iso
    // six characters, one per field in the order PS PH QF BR BT QB:
    // E = even under the composite isometry -> dr(F) = 0 at the minimal
    // surface, O = odd -> F = 0.  The count is 6nt - 3 for all 64.
    std::string isoparity;               // --iso-parity EEEOOE
    // ⚠ STEP 4's MATCHING UNKNOWNS (research round 414 ruling 1).  The inner
    // rows above register ROWS ONLY, and round 171 measured the consequence:
    // with every specified row in, the assembly has FULL COLUMN RANK, so the
    // posed system's kernel cannot be reached from any dumpable matrix.  The
    // directions that are free live in COLUMNS this probe does not have.
    //
    // --jac-match registers them: 8nt - 6 Kadath scalar unknowns, the throat
    // system's free data, entering the C0/C1 rows they are specified to enter.
    //   grade-0  6nt - 3   one per field per angular mode, in that field's own
    //                      basis, subtracted from the C0 (value) row
    //   tower    2nt - 3   COS_ODD (nt-1) + SIN_EVEN (nt-2), subtracted from
    //                      the C1 (derivative) rows of QB and BT
    // ⚠ THE TOWER'S PLACEMENT IS THE ONE REPRESENTATIONAL CHOICE HERE, and it
    // is not free: round 164 measured the free subspace's column count as
    // 2nt - 3 by INTERSECTION with the parity blocks, and COS_ODD + SIN_EVEN is
    // the only class pair that reproduces it (the other nine admissible pairs
    // give 2nt - 2, 2nt - 1 or 2nt).  Coinciding counts is not the same as the
    // subspace BEING that pair, and that is stated rather than assumed.
    // ⚠ THE BASIS-ASSIGNMENT INSTRUMENT (research round 416 ruling 3).
    // --jac-match-break showed this build CANNOT answer which angular family
    // each matching amplitude belongs to: the plant is reinterpreted because a
    // sum takes its first operand's theta basis (round 111), so a single-mode
    // coefficient array stays single-mode whichever family it is read in.
    //
    // ⚠ SO THIS ROUTES THROUGH NO SUM AT ALL.  It plants a known combination of
    // a family's modes into THE FIELD ITSELF -- pointwise values, transformed
    // by the field's own declared base -- and reads the inner tau rows off the
    // RHS.  No Term_eq addition is involved anywhere, so the mistagging cannot
    // reach it.
    //
    // ⚠ AND THE READING IS A RATIO, NOT A VALUE, because the tau projection may
    // normalise each mode differently and a single plant cannot separate that
    // normalisation from the planted amplitude.  Two plants do: with a_k = 1
    // and a_k = k+1 over the same family, row j must satisfy
    //     rhs_j(B) / rhs_j(A) = j + 1
    // exactly, whatever c_j is.  That identifies WHICH angular mode each row
    // picks off, which is the question.
    //   A   a_k = 1     in each field's DECLARED family
    //   B   a_k = k+1   in each field's DECLARED family
    //   WA/WB  the same two, but BT is given COS_EVEN modes and QB SIN_EVEN
    //          ones -- the WRONG families.  The ratio must then FAIL to be
    //          j + 1, or the instrument is not measuring the family either.
    std::string basisprobe;          // --basis-probe A|B|WA|WB
    bool jacmatch = false;           // --jac-match
    // ⚠ ROUND 193, research round 485.  Round 483 found the series has no rank
    // deficiency -- it has two leading amplitudes its equations do not
    // determine, Bh and Qh -- and round 397's "deficiency 2 at every cut" was
    // that fact mis-read through p-indexing as a dependency.  The tower's
    // 2nt - 3 unknowns were registered against round 397's number, so they may
    // DOUBLE-COUNT a freedom already sitting in Bh and Qh among the leading six.
    // With this flag only the 6nt - 3 grade-0 amplitudes are registered, against
    // the same 12nt - 6 inner rows: 3051 + 27 = 3078 columns against 3078 rows
    // at nt = 5, square before any measurement.
    bool jacmatchnotower = false;    // --jac-match-notower
    bool seedmatch = false;          // --seed-match
    // ⚠ --jac-match-break WAS WRITTEN AS A NEGATIVE CONTROL AND CAME BACK
    // NEGATIVE, WHICH IS THE RESULT.  It gives BT's grade-0 amplitudes a
    // COS_EVEN profile where their row is SIN_EVEN, and the intent was that the
    // column stop being single-row.  It does not: round 172 compared the two
    // dumps column by column and the only difference is at 1e-16.
    //
    // The reason is round 111's: Kadath tags a sum with its FIRST operand's
    // theta basis, so `BT - G * MCE00` reinterprets the profile's COEFFICIENT
    // ARRAY in BT's basis, and a single-mode array stays a single-mode array
    // whichever family it is read in.  So the incidence check below tests
    // BIJECTIVITY and unit coupling -- both real and both necessary -- and it
    // is BLIND to the basis assignment.  Stated here rather than claimed
    // otherwise: this flag is kept because it is what measured that.
    //
    // ⚠ The consequence cuts both ways, and the useful direction is the second:
    // this build does NOT validate which angular family each matching amplitude
    // belongs to, AND the kernel measurement cannot be contaminated by getting
    // it wrong, because the assembled Jacobian is the same either way.
    bool jacmatchbreak = false;      // --jac-match-break
    bool jacrec = false;             // --jac-recursion
    bool jacrecearly = false;        // --jac-recursion-early
    double ampgen = 0.0;             // --amp-generic X
    // ⚠ THE OUTER BLOCK, VALUE AND DERIVATIVE, ALL SIX FIELDS (round 165).
    // Not a proposed BC set: a SPANNING SET of outer-boundary functionals, so
    // that dim(image of ker(bulk+interfaces) under restriction to the outer
    // boundary) can be measured.  That dimension is how much of the interior's
    // undetermined freedom is visible as far-field data, and it is the number
    // the outer specification must be written against -- research round 400
    // ruled that "one condition per field per angular mode" was an assumption.
    bool jacouterfull = false;       // --jac-outer-full
    // ⚠ ROUND 186: WHERE the spanning set is EVALUATED, not what it is.
    // Rounds 184-185 bracketed the 4 + (2nt-7) grading by changing the
    // BACKGROUND with the structure fixed and the STRUCTURE with the background
    // fixed; it survived both.  The item neither session had listed is the one
    // the grading is DEFINED AGAINST -- these functionals.  `-1` keeps the
    // historical behaviour (the outer face of dtop, which on the compact domain
    // is r = infinity); 0..dtop puts the SAME twelve blocks on the outer face of
    // an interior domain, i.e. at a FINITE radius, changing where they live
    // without changing what they are.
    int jacouterfulldom = -1;        // --jac-outer-full-dom <d>
    // ⚠ ROUND 155: the two outer rows registered SEPARATELY, because research
    // round 379 found that multr(QF) = 0 is not what it is named for.  It is
    // not t_Q = 0 (which needs no imposing) and not the asymptotic condition
    // (which must fix the ln r coefficient or the additive constant); it is
    // q = 0 at r = 3.9389, where q is small and not zero -- an approximation
    // imposed as a condition, which a COUNT cannot see.  --jac-outer-rows
    // selects a subset so the two can be told apart.
    std::string jacouterrows = "q,phb";   // --jac-outer-rows q,phb
    // ---- --newton: THE SOLVE (round 201) ---------------------------------
    // ⚠ The registration is EXACTLY --dump-jacobian's: the six fields as
    // variables, the matching amplitudes as variables, the same rows.  Newton
    // adds nothing to the system -- it solves the one that has been measured
    // for eight rounds -- which is why the flag shares the whole path and only
    // the file write is gated on --dump-jacobian.
    bool donewton = false;
    int  newtonmax = 8;
    double newtonprec = 1e-12;
    // --nexp: the throat exponent, hardcoded to sqrt(2) until now.  ⚠ THIS IS
    // NOT THE SHOOTING.  It makes n settable so its effect can be MEASURED;
    // the shooting needs a scalar residual whose zero defines n, and there is
    // none in the p_max = 1 system (see NOTES_finiteJ_bvp.md, round 201).
    double nexp = std::sqrt(2.0);
    std::string newtonfields;   // --newton-fields FILE
    // ---- the J = 0 diagnostic (round 202, research round 498) -------------
    // ⚠ TWO VARIANTS, AND THEY ARE DIFFERENT SYSTEMS.  The ruling says
    // "replace the two vacuous C0 rows on q and q-bar by Dirichlet on their
    // amplitudes", and separately that the two fields "carry effectively
    // NEUMANN ONLY at the inner face".  For g0 = 0 the pair is
    //     C0   F - sum_j G_j M_j = 0        (defines G; imposes nothing)
    //     C1   dr(F) - sum_j G_j dr(M_j) = 0  ->  dr(F) = 0, since dr(r^0) = 0
    // so replacing ONE of them by the amplitude row  sum_j G_j M_j = 0  gives:
    //     --inner-amp-pin  (C0 replaced)  ->  {G = 0, dr(F) = 0}  -- the field
    //         still has NEUMANN ONLY.  This is the ruling's literal text.
    //     --inner-amp-dir  (C1 replaced)  ->  {F(r_m) = 0, G = 0}  -- a genuine
    //         DIRICHLET on the field.  This is what the ruling's purpose needs.
    // Both are count-preserving and touch no columns.  Both are run, because
    // `pin` is the negative control that shows the closure is the Dirichlet and
    // not the pinning of an amplitude that was always free.
    std::string ampdir, amppin;   // --inner-amp-dir / --inner-amp-pin  QF,QB
    // ---- --newton-delta FILE: apply an EXTERNALLY COMPUTED Newton step ----
    // Round 203.  The minimum-norm step has to be computed where the singular
    // values can be compared across LAPACK drivers -- that is how the noise
    // floor is DERIVED rather than chosen -- and that is offline.  This is the
    // hook back in: xx_to_vars_delta is public and is exactly what do_newton
    // calls, so the iteration is
    //     probe --dump-jacobian  ->  offline truncated solve  ->  probe
    //     --newton-delta (accumulated)  --dump-jacobian  ->  ...
    // ⚠ SIGN: xx_to_vars_delta SUBTRACTS (system_of_eqs.cpp:557), so the file
    // carries X solving  J X = b,  not  J X = -b.
    // ⚠ ORDER: applied AFTER every add_var and after the equations are
    // registered, which is where do_newton applies it.  Anywhere earlier and
    // the unknown count it asserts against is not final.
    std::string newtondelta;
    // ---- --eq-prefactor NAME:k,... : multiply an equation by r^k BEFORE the
    // tau projection (round 205, research round 502 ruling 2).
    // ⚠ WHY THIS IS NOT A ROW SCALING.  A radial prefactor applied to the
    // assembled ROW is a diagonal multiplier, and round 205 measured the
    // minimum-norm step to be EXACTLY invariant under one (||dx|| identical to
    // seven digits under a 10^+-7 scaling).  Applied to the EXPRESSION, multr
    // mixes radial modes before the tau projection drops the top coefficient,
    // so it changes WHICH combination is discarded.  That is the only part of
    // the prefactor proposal that can reach this solver, and this flag isolates
    // it.
    // ⚠ INTEGER POWERS ONLY.  The leading grades are -2, -1, -2, -n-3, -2n-2,
    // -2; four are integers and two are not, so E_Phi and E_q cannot be treated
    // this way and are left alone.  A partial test is still a test: if the four
    // move nothing, the two will not either.
    std::string eqpref;
    // ---- --tail-readout: the 1/r coefficients at the r = infinity node ------
    // Round 207, research round 504 ruling (3).  On the compact domain multr
    // raises the decay order, so the VALUE of multr(F) at the r = infinity node
    // IS F's 1/r coefficient -- which is why the outer rows are written that
    // way.  Round 206 tried to get it by fitting r(PH-1) in powers of 1/r and
    // got 7e-4 to 1.1e-3 across fit degrees 2-4, against a floor of alpha =
    // 4.6e-4.  ⚠ A fit cannot resolve it; the coefficient is EXACT in Kadath and
    // is a readout, not a row.  With phv carrying Phibar's node value, multr(PH)
    // is free and this is the theorem-level check the draft describes.
    bool tailout = false;
    // ⚠ ROUND 156: the MIXED outer row  r d_r q + lambda q = 0, which
    // interpolates between the two limits research named.  For q ~ c log r + d
    // the Dirichlet limit (lambda -> infinity) fixes d and the Neumann limit
    // (lambda = 0) fixes c, and round 155 measured the second as nearly implied
    // by the bulk.  Sweeping lambda maps the whole family in one pass, so
    // "no member is well-conditioned" is a measurement rather than two points.
    bool jacmixed = false;
    double jacmixlam = 0.0;          // --jac-outer-mixed LAMBDA
    double outerpert = 0.0;          // --outer-perturb DELTA: add DELTA/r^p to PH
    int outerpow = 1;                // --outer-perturb-pow p
    // ⚠ ROUND 137: the constant gauge family, as MULTIPLICATIVE scalings.
    // CONVENTIONS l.16 absorbs a two-parameter constant family with
    // U(inf) = Q(inf) = F(inf) = 0.  psi^2 = (R/r)e^{2 eps U} and
    // alpha = W e^{eps F}, so a constant U shift scales PS and a constant F
    // shift scales alpha -- and Phb = alpha psi^2 carries both.  These two
    // apply the scalings directly so the family can be walked.
    double scalePS = 1.0, scalePH = 1.0, scaleBR = 1.0;  // --scale-ps/-ph/-br F
    // ⚠ ROUND 146: --esht-nt2 registers the weighted E_sh^theta through one
    // multsint, so its TAU PROJECTION drops from ntheta-1 to ntheta-2.
    // Research round 365's count: the weighted row is COS_ODD and carries
    // ntheta-1 angular modes, while BT -- the field it is paired with -- is
    // SIN_EVEN and carries ntheta-2.  The system imposes one more mode of
    // equation than the field has, once per radial slot, which is the 88.
    // multsint takes COS_ODD to SIN_EVEN (the measured operator/basis lattice),
    // so sin(theta) * E_sh^theta is the SAME equation away from the axis,
    // projected onto ntheta-2 modes, and it imposes nothing at theta = 0.
    // ⚠ It is NOT another weight: round 104 measured multsint . divsint = id at
    // 1.1e-16 and the operand here is a finished, pole-free row, so nothing
    // about the divsint contract changes.  What changes is the projection.
    // ⚠ ADOPTED, research round 366 ruling 2: this is ON, and --no-esht-nt2
    // turns it off for an A/B.  Absent means the ADOPTED value, never the
    // permissive one -- rounds 125, 143 and 144 each cost a round to a flag
    // whose absence meant False, and the convention is now the project's.
    bool eshtnt2 = true;             // --esht-nt2 / --no-esht-nt2
    // ⚠ STEP 4, FIRST PIECE (round 148): S_def, the deformed reference lapse's
    // contribution.  working_draft.tex l.1971-1973 defines
    //   alpha_P^{(n)} := alpha_P^{n/sqrt2},   delta := n/sqrt2 - 1
    // and l.1996-2003 gives
    //   S_def = delta [ Lap3 ln alpha_P + 2 D ln Phb_P . D ln alpha_P
    //                   + D ln(r sin th) . D ln alpha_P ]
    //           + delta^2 (D ln alpha_P)^2
    // l.2004-2006: it "carries no field and so enters the residual and never
    // the Jacobian: the Newton solve is unchanged and only the right-hand side
    // moves", which is why this piece is buildable and checkable with no solve.
    // --sdef DELTA runs it; the two self-checks the draft states are run with
    // it and are the whole point of doing this piece first.
    bool sdef = false;
    double sdefdelta = 0.0;          // --sdef DELTA
    // ⚠ --jj SETS THE SPIN CONSTANT WITHOUT MANUFACTURED DATA.  Until now JJ
    // came from the manufactured file or was 0, and round 126 lost a round to
    // add_cst("JJ", 0.0) while the data was at J = 0.1.  The J = 0 seed with
    // JJ != 0 is a legitimate and useful state: q = 0 there, so the E_q
    // residual IS the source, which is what the t_Q monopole question needs.
    double jjoverride = 0.0;
    bool havejj = false;             // --jj J
    // ⚠ --compact INCLUDES THE COMPACT DOMAIN, which nothing in this thread has
    // exercised (research round 368).  The probe has always stopped at the last
    // finite shell because the seed as WRITTEN is 0/0 there -- R/r and C r/R^3
    // with R = Rr*r and r = infinity.  But the table carries Rr and oor = 1/r
    // as columns, and in those the same seed is
    //     psi^2 = Rr,   Phb = W Rr,   beta~^r = C oor^2 / Rr^3
    // with no 0/0 anywhere: at r = infinity the table gives W = 1, Rr = 1,
    // oor = 0 exactly.  The finite-shell path is left BYTE-IDENTICAL -- it
    // still computes R/rr -- because rewriting it would move every number this
    // thread has published.
    bool compact = false;            // --compact
    // ⚠ ROUND 151: an explicit B-hat^theta / beta~^r profile, so the monopole
    // sweep can carry the note's OWN throat form as a named member instead of
    // scanning blind.  analysis/note/main.tex l.1967-1968:
    //   \hat\mB^\theta &= \alpha_0 \tb_0 r^{\sqrt{2}+1} A \sin\theta \cos\theta
    //   \mB^r          &= \alpha_0 \tb_0 r^{\sqrt{2}+1} ( A \sin^2\theta + B/\sqrt2 )
    // and l.1971: "where $A,B,C, D$ are constant to be determined."
    // ⚠ THE TWO SHARE A AND THE SAME RADIAL POWER, so a BT-only sweep is not
    // the throat solution.  --br-prof exists for exactly that reason and the
    // tied run is reported beside the untied one.
    double btpow = 0.0, btamp = 0.0;     // --bt-prof POW AMP : AMP r^POW sin cos
    int btang = 2;                       // --bt-ang M        : sin(M theta)/2
    // ⚠ --bt-taper: AMP r^POW / (1 + r^2)^POW instead of AMP r^POW.
    // The bare power VANISHES at the throat as required but GROWS without
    // bound outward, and this grid reaches r = 409.99 -- so --bt-prof 2 0.004
    // puts 336 at the outer end against fields of order 1.  Round 180 measured
    // the consequence: the outer-visibility gap collapsed from 1e3-1e4 to
    // O(1) at EVERY amplitude down to 0.004, which reads as "any beta^^theta
    // dissolves the four" and is in fact the profile blowing up where the
    // spectrum is measured.  The taper is ~r^POW inward and ~r^-POW outward,
    // so it vanishes at BOTH ends as a physical beta^^theta must.
    bool bttaper = false;                // --bt-taper
    double brpow = 0.0, bramp = 0.0, brB = 0.0;  // --br-prof POW A B
    // ⚠ ROUND 152: the note's THREE-FIELD throat configuration, applied with
    // its own coefficients rather than as a shape plus a free amplitude.
    // analysis/note/main.tex l.1966-1968:
    //   u            = u_0 + (1/(2 sqrt2)) alpha_0 r^sqrt2 ( A sin^2 th + D )
    //   Bhat^theta   = alpha_0 tb_0 r^{sqrt2+1} A sin th cos th
    //   B^r          = alpha_0 tb_0 r^{sqrt2+1} ( A sin^2 th + B/sqrt2 )
    // l.1969's varphi is NOT applied: it needs C and E, which research did not
    // carry across, and importing them from the earlier version's section is
    // exactly the substitution this project keeps paying for.
    // psi^2 = (R/r) e^{2u}, so the u perturbation enters as PS *= exp(2 du).
    bool throat = false;
    double thA = 0.0, thB = 0.0, thD = 0.0, tha0 = 0.0, thb0 = 0.0;
    // ⚠ ROUND 153: l.1969's varphi, with C and E PASSED IN rather than known.
    //   \vp = (1/sqrt2) alpha_0 Phb_0 r^{2 sqrt2 - 1} ( C sin^2 th + E )
    // Phibar = alpha psi^2 and varphi is its log perturbation, so this enters
    // as PH *= exp(dvp).  It exists so that the linear RESPONSE to C and to E
    // can be measured and the C that would close the A* gap can be SOLVED FOR
    // -- which is a statement about what C would have to be, not a value of C.
    double thC = 0.0, thE = 0.0, thP0 = 0.0;   // --phi-prof C E Phb_0
    // --manufactured FILE: fill the six unknowns from a table and compare the
    // equations against the sources tabulated beside them.  The FIELD SET and
    // the EQUATION NAMES are read from the file's header, so changing either
    // costs a re-run of the generator and no edit here.
    std::string manfile;
    for (int i = 2; i < argc; i++) {
        const std::string k = argv[i];
        if (k == "--ntheta") ntheta = std::stoi(argv[++i]);
        else if (k == "--perturb") perturb = std::stod(argv[++i]);
        else if (k == "--try-monolithic") monolithic = true;
        else if (k == "--clean") clean = true;
        else if (k == "--extra-def") extra.push_back(argv[++i]);
        else if (k == "--maxdefs") maxdefs = std::stoi(argv[++i]);
        else if (k == "--extra-early") extra_early = true;
        else if (k == "--touch") touch = true;
        else if (k == "--read-before") readbefore.push_back(argv[++i]);
        else if (k == "--read-all") readall = true;
        else if (k == "--break-contract") breakcontract = true;
        else if (k == "--break-basis") breakbasis = true;
        else if (k == "--no-basis-check") nobasischeck = true;
        else if (k == "--table-noise") tablenoise = std::stod(argv[++i]);
        else if (k == "--seed-perturb") seedperturb = std::stod(argv[++i]);
        else if (k == "--pfield") pfield = argv[++i];
        else if (k == "--pmix") pmix = argv[++i];
        else if (k == "--dump-grid") gridout = argv[++i];
        else if (k == "--dump-defs") defsout = argv[++i];
        else if (k == "--dump-fields") fieldsout = argv[++i];
        else if (k == "--dump-fields-post") fieldspost = argv[++i];
        else if (k == "--dump-eqvals") eqvalsout = argv[++i];
#ifdef TRUMPET_CHI_UNKNOWN
        else if (k == "--chi-seed") chiseed = argv[++i];
#endif
        else if (k == "--jac-balance") jacbalance = true;
        else if (k == "--balance-l2") { balancel2 = true; jacbalance = true; }
        else if (k == "--maximality") maximality = true;
        else if (k == "--physical-h") physh = true;
        else if (k == "--grade2-u2") g2u2 = true;
        else if (k == "--jac-axisres") jacaxisres = true;
        else if (k == "--grade1-replace") g1replace = true;
        else if (k == "--drop-tephi") droptephi = true;
        else if (k == "--qlog") qlog = std::stod(argv[++i]);
#ifdef TRUMPET_BULK2
        else if (k == "--eqfn-sin2") eqfnsin2 = true;
        else if (k == "--amax-table") amaxtable = argv[++i];
        else if (k == "--komar") komar = true;
        else if (k == "--komar-m") komar_m = std::stod(argv[++i]);
#endif
        else if (k == "--dump-jacobian") jacdump = argv[++i];
        else if (k == "--dump-rowmeta") rowmeta = argv[++i];
        else if (k == "--jac-fields") jacfields = argv[++i];
        else if (k == "--jac-eqs") jaceqs = argv[++i];
        else if (k == "--jac-interfaces") jacinterfaces = true;
        else if (k == "--jac-outer") jacouter = true;
        else if (k == "--jac-inner") jacinner = true;
        else if (k == "--jac-inner-mode") { jacinner = true;
                                            jacinnermode = argv[++i]; }
        else if (k == "--iso-parity") isoparity = argv[++i];
        else if (k == "--basis-probe") basisprobe = argv[++i];
        else if (k == "--jac-match") jacmatch = true;
        else if (k == "--jac-match-notower") { jacmatch = true;
                                               jacmatchnotower = true; }
        else if (k == "--seed-match") seedmatch = true;
        else if (k == "--jac-match-break") { jacmatch = true;
                                             jacmatchbreak = true; }
        else if (k == "--jac-recursion") { jacmatch = true; jacrec = true; }
        else if (k == "--amp-generic") ampgen = std::stod(argv[++i]);
        else if (k == "--jac-recursion-early") { jacmatch = true;
                                    jacrec = true; jacrecearly = true; }
        else if (k == "--jac-outer-full") jacouterfull = true;
        else if (k == "--jac-outer-full-dom") jacouterfulldom = std::stoi(argv[++i]);
        else if (k == "--jac-outer-rows") jacouterrows = argv[++i];
        else if (k == "--newton") donewton = true;
        else if (k == "--newton-max") { donewton = true;
                                        newtonmax = std::stoi(argv[++i]); }
        else if (k == "--newton-prec") { donewton = true;
                                         newtonprec = std::stod(argv[++i]); }
        else if (k == "--nexp") nexp = std::stod(argv[++i]);
        else if (k == "--newton-fields") { donewton = true;
                                           newtonfields = argv[++i]; }
        else if (k == "--inner-amp-dir") ampdir = argv[++i];
        else if (k == "--inner-amp-pin") amppin = argv[++i];
        else if (k == "--newton-delta") newtondelta = argv[++i];
        else if (k == "--eq-prefactor") eqpref = argv[++i];
        else if (k == "--tail-readout") tailout = true;
        else if (k == "--jac-outer-mixed") { jacmixed = true;
                                             jacmixlam = std::stod(argv[++i]); }
        else if (k == "--outer-perturb") outerpert = std::stod(argv[++i]);
        else if (k == "--outer-perturb-pow") outerpow = std::stoi(argv[++i]);
        else if (k == "--scale-ps") scalePS = std::stod(argv[++i]);
        else if (k == "--scale-ph") scalePH = std::stod(argv[++i]);
        else if (k == "--scale-br") scaleBR = std::stod(argv[++i]);
        else if (k == "--esht-nt2") eshtnt2 = true;
        else if (k == "--no-esht-nt2") eshtnt2 = false;
        else if (k == "--sdef") { sdef = true; sdefdelta = std::stod(argv[++i]); }
        else if (k == "--jj") { havejj = true; jjoverride = std::stod(argv[++i]); }
        else if (k == "--compact") compact = true;
        else if (k == "--bt-prof") { btpow = std::stod(argv[++i]);
                                     btamp = std::stod(argv[++i]); }
        else if (k == "--bt-ang") btang = std::stoi(argv[++i]);
        else if (k == "--bt-taper") bttaper = true;
        else if (k == "--phi-prof") { thC = std::stod(argv[++i]);
                                      thE = std::stod(argv[++i]);
                                      thP0 = std::stod(argv[++i]); }
        else if (k == "--throat") { throat = true;
                                    thA = std::stod(argv[++i]);
                                    thB = std::stod(argv[++i]);
                                    thD = std::stod(argv[++i]);
                                    tha0 = std::stod(argv[++i]);
                                    thb0 = std::stod(argv[++i]); }
        else if (k == "--br-prof") { brpow = std::stod(argv[++i]);
                                     bramp = std::stod(argv[++i]);
                                     brB = std::stod(argv[++i]); }
        else if (k == "--manufactured") manfile = argv[++i];
    }
    // ⚠ --physical-h acts only where the grade-1 seed is written (--seed-match
    // with --jac-recursion); anywhere else it would register and do nothing.
    if (g2u2) {
#ifndef THROATTH_HAS_U2
        if (rank == 0)
            std::cerr << "FATAL: --grade2-u2: this binary's throat header has"
                         " no grade-2 u2 rows (throat_emit.py --grade2-u2)\n";
        MPI_Finalize();
        return 1;
#endif
        if (!physh || !balancel2 || !maximality || g1replace || jacaxisres
            || droptephi || jacrecearly) {
            if (rank == 0)
                std::cerr << "FATAL: --grade2-u2 needs --physical-h, "
                             "--balance-l2 and --maximality, and refuses "
                             "--grade1-replace, --jac-axisres, --drop-tephi, "
                             "--jac-recursion-early\n";
            MPI_Finalize();
            return 1;
        }
    }
    if (physh && !(seedmatch && jacrec)) {
        if (rank == 0)
            std::cerr << "FATAL: --physical-h needs --seed-match and "
                         "--jac-recursion (the grade-1 seed it rescales)\n";
        MPI_Finalize();
        return 1;
    }

    // ---- A1's manufactured data (round 125) -------------------------------
    // The generator writes, per collocation point, the six FIELD values and the
    // exact value of each emitted EQUATION on them.  Filling the fields from
    // the table and comparing Kadath's residual against the tabulated source
    // measures the evaluation error and nothing else: the fields are not a
    // solution, and the source is what makes them one.
    std::map<long long, std::vector<double> > mandata;
    long man_installed = 0, man_missing = 0;
    std::vector<std::string> manfields, maneqs;
    double man_n = 0.0, man_J = 0.0;
    if (!manfile.empty()) {
        std::ifstream mf(manfile);
        if (!mf) {
            if (rank == 0) std::cerr << "FATAL: cannot open " << manfile << "\n";
            MPI_Finalize();
            return 1;
        }
        std::string line;
        while (std::getline(mf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream is(line);
            std::string key; is >> key;
            if (key == "fields") { std::string w; while (is >> w) manfields.push_back(w); }
            else if (key == "eqs") { std::string w; while (is >> w) maneqs.push_back(w); }
            else if (key == "n") is >> man_n;
            else if (key == "J") is >> man_J;
            else if (key == "val") {
                int d, i, jj; is >> d >> i >> jj;
                std::vector<double> v; double x;
                while (is >> x) v.push_back(x);
                mandata[(static_cast<long long>(d) * 100000LL
                         + static_cast<long long>(i) * 1000LL + jj)] = v;
            }
        }
        if (rank == 0)
            std::cout << "#   manufactured: " << mandata.size() << " points, "
                      << manfields.size() << " fields, " << maneqs.size()
                      << " equations, n = " << man_n << ", J = " << man_J << "\n";
    }

    TrumpetIO::Table t;
    try {
        t = TrumpetIO::read_table(argv[1]);
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "FATAL: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }
    const int ndom = static_cast<int>(t.doms.size());
    const int dlast = ndom - 1;

    std::vector<double> bounds;
    std::vector<Dim_array> res;
    std::vector<bool> logshell;
    for (int d = 0; d < ndom; d++) {
        logshell.push_back(t.doms[d].kind == "log");
        bounds.push_back(t.doms[d].r_int);
        Dim_array n(2);
        n.set(0) = t.doms[d].nbr;
        n.set(1) = ntheta;
        res.push_back(n);
    }
    Point centre(2);
    centre.set(1) = 0.0;
    centre.set(2) = 0.0;
    Trumpet::Space_polar_trumpet space(CHEB_TYPE, centre, res, bounds, logshell);

    // The compact domain carries r = infinity, where the seed's R/r and C r/R^3
    // are 0/0 in floating point.  The probe therefore runs on the SHELLS only
    // and says so, rather than reporting a NaN as a small number.
    const int dtop = compact ? dlast : dlast - 1;
    if (compact && !clean) {
        // ⚠ the smoke battery reads RR, which is r and is infinite in the
        // compact domain.  Refused rather than run: a battery of inf is not a
        // failure it would report, it is a battery that cannot discriminate.
        if (rank == 0)
            std::cerr << "FATAL: --compact requires --clean; the smoke battery "
                         "reads RR = r, which is infinite there.\n";
        MPI_Finalize();
        return 8;
    }
    std::cout << "# finiteJ_probe  ntheta=" << ntheta << "  shells 0.." << dtop
              << " of " << ndom << " domains  perturb=" << perturb << "\n";
    emit("FJP_ntheta", ntheta);
    emit("FJP_domains_probed", dtop + 1);
    emit("FJP_perturb", perturb);
    emit("FJP_table_noise", tablenoise);

    const double M = t.M;
    const double C = 3.0 * std::sqrt(3.0) * M * M / 4.0;

    Scalar PS(space), PH(space), QF(space), BR(space), BT(space), QB(space);
#ifdef TRUMPET_CHI_UNKNOWN
    // chi promoted to an unknown (research round 554 ruling 1).  Seeded from the
    // product AFTER the defs are registered -- see the CH seed block below --
    // because the product is D0048, which the emission already builds, so the
    // seed is Kadath's own evaluation and not a second transcription of it.
    Scalar CH(space);
#endif
    Scalar RR(space), ST(space), CT(space), C2(space), H2(space), L2(space),
           CX(space), SQ(space), T7(space), ONE(space);
    // Every domain must be allocated even though the system covers only the
    // shells: a Scalar with an unallocated domain segfaults inside add_cst.
    for (int d = 0; d < ndom; d++) {
        const Kadath::Domain* dm = space.get_domain(d);
        Val_domain& vps = PS.set_domain(d);
        Val_domain& vph = PH.set_domain(d);
        Val_domain& vqf = QF.set_domain(d);
        Val_domain& vbr = BR.set_domain(d);
        Val_domain& vbt = BT.set_domain(d);
        Val_domain& vqb = QB.set_domain(d);
#ifdef TRUMPET_CHI_UNKNOWN
        // ⚠ every domain, even those the system does not cover: an unallocated
        // domain segfaults inside add_cst/add_var.
        // ⚠⚠ AND NOT IDENTICALLY ZERO.  Round 261: seeding CH to 0.0 here made
        // the read of `D0085 = dr(CH)` segfault inside finiteJ_register -- an
        // all-zero Val_domain has no coefficient space for dr to differentiate,
        // and the six fields never hit it because all six have nonzero seeds.
        // The value planted here is a PLACEHOLDER; the physical seed is written
        // just below, from the backbone's own closed form.
        CH.set_domain(d) = 1.0;
#endif
        Val_domain& vrr = RR.set_domain(d);
        Val_domain& vst = ST.set_domain(d);
        Val_domain& vct = CT.set_domain(d);
        Val_domain& vc2 = C2.set_domain(d);
        Val_domain& vh2 = H2.set_domain(d);
        Val_domain& vl2 = L2.set_domain(d);
        Val_domain& vcx = CX.set_domain(d);
        Val_domain& vsq = SQ.set_domain(d);
        Val_domain& vt7 = T7.set_domain(d);
        Val_domain& von = ONE.set_domain(d);
        for (Val_domain* v : {&vps, &vph, &vqf, &vbr, &vbt, &vqb, &vrr, &vst, &vct, &vc2, &vh2, &vl2, &vcx, &vsq, &vt7, &von})
            v->allocate_conf();
        Index idx(dm->get_nbr_points());
        do {
            const int i = idx(0);
            if (d == dlast) {                    // the compact domain
                if (!compact) {
                    for (Val_domain* v : {&vps, &vph, &vqf, &vbr, &vbt, &vqb,
                                          &vrr, &vst, &vct, &vc2, &vh2, &vl2, &vcx, &vsq, &vt7, &von})
                        v->set(idx) = 0.0;
                    continue;
                }
                // the SAME seed, written in the table's own columns so that
                // nothing is 0/0 -- see --compact above
                const double Wc = t.pts[d][i].W;
                const double Rc = t.pts[d][i].Rr;
                const double uc = t.pts[d][i].oor;
                const double thc = dm->get_coloc(2)(idx(1));
                vps.set(idx) = Rc;
                vph.set(idx) = Wc * Rc;
                vqf.set(idx) = 0.0;
                vbr.set(idx) = C * uc * uc / (Rc * Rc * Rc);
                vbt.set(idx) = 0.0;
                vqb.set(idx) = 0.0;
                // ⚠ the scaffolding fields are NOT extended.  RR is r, which is
                // infinite here, and the smoke battery that uses it is off under
                // --clean; --compact requires --clean for that reason and the
                // check is below.  ST/CT and friends are angular and carry over.
                vrr.set(idx) = (uc > 0.0) ? 1.0 / uc : 0.0;
                vst.set(idx) = std::sin(thc);
                vct.set(idx) = std::cos(thc);
                vc2.set(idx) = std::cos(2.0 * thc);
                vh2.set(idx) = 0.0;
                vl2.set(idx) = 0.0;
                vcx.set(idx) = 0.0;
                vsq.set(idx) = (1.0 - std::cos(2.0 * thc)) / 2.0;
                vt7.set(idx) = std::sin(2.0 * thc) / 2.0;
                von.set(idx) = 1.0;
                continue;
            }
            const double rr = t.pts[d][i].r;
            // a fixed hash of (d, i) so the pattern is identical run to run and
            // the only thing varying between runs is the amplitude
            auto jitter = [&](int salt) {
                if (tablenoise == 0.0)
                    return 1.0;
                unsigned h = 2166136261u;
                for (int x : {d, i, salt}) {
                    h ^= static_cast<unsigned>(x);
                    h *= 16777619u;
                }
                return 1.0 + tablenoise * (2.0 * (h / 4294967296.0) - 1.0);
            };
            const double W = t.pts[d][i].W * jitter(1);
            const double R = t.pts[d][i].Rr * rr * jitter(2);   // Rr is R/r
            const double th = dm->get_coloc(2)(idx(1));
            // the J = 0 seed, exactly as round 102 verified it symbolically
            vps.set(idx) = R / rr;
            vph.set(idx) = W * R / rr;
            vqf.set(idx) = 0.0;
            vbr.set(idx) = C * rr / (R * R * R);
            vbt.set(idx) = 0.0;
            vqb.set(idx) = 0.0;
            vrr.set(idx) = rr;
            vst.set(idx) = std::sin(th);
            vct.set(idx) = std::cos(th);
            vc2.set(idx) = std::cos(2.0 * th);
            // lap2(r^2 cos2th) = (n^2 - m^2) r^{n-2} cos = 0 exactly
            vh2.set(idx) = rr * rr * std::cos(2.0 * th);
            // lap(r^2 P_2(cos th)) = 0 exactly;  P_2 = (1 + 3cos2th)/4
            vl2.set(idx) = rr * rr * (1.0 + 3.0 * std::cos(2.0 * th)) / 4.0;
            // the TARGET of the cot-theta construction, built pointwise purely
            // as a comparison value -- it is never differentiated
            vcx.set(idx) = (std::sin(th) == 0.0) ? 0.0
                           : (std::cos(th) / std::sin(th)) * std::cos(2.0 * th);
            // A field that VANISHES on the axis, so cot(theta)*X is finite
            // there: X = sin^2 = (1 - cos2th)/2, and cot*X = sin(2th)/2.
            vsq.set(idx) = (1.0 - std::cos(2.0 * th)) / 2.0;
            vt7.set(idx) = std::sin(2.0 * th) / 2.0;
            von.set(idx) = 1.0;
            if (!mandata.empty()) {
                // ⚠ the SIX UNKNOWNS only.  RR/ST/CT/... are the probe's own
                // comparison scaffolding and stay as built; overwriting them
                // would change what the diagnostics mean.
                auto it = mandata.find(static_cast<long long>(d) * 100000LL
                                       + static_cast<long long>(i) * 1000LL
                                       + idx(1));
                if (it != mandata.end() && it->second.size() >= 6) {
                    const std::vector<double>& v = it->second;
                    vps.set(idx) = v[0];  vph.set(idx) = v[1];
                    vqf.set(idx) = v[2];  vbr.set(idx) = v[3];
                    vbt.set(idx) = v[4];  vqb.set(idx) = v[5];
                    man_installed++;
                } else {
                    man_missing++;
                }
            }
        } while (idx.inc());
    }
    // the control: a perturbation of the seed must NOT annihilate the equations
    if (perturb != 0.0) {
        for (int d = 0; d <= dtop; d++)
            PH.set_domain(d) = PH(d) * (1.0 + perturb);
    }
    // ---- THE DECLARED ANGULAR BASES (research round 313 ruling 1) ---------
    //
    // Kadath tags a sum with its FIRST operand's theta basis without checking
    // that the two agree (round 111), so a sum of unlike bases is silently
    // mistagged and every coefficient-space operation taken of it afterwards is
    // wrong.  The emitter's basis lattice sweeps all 4096 declarations of the
    // six unknowns and reports which leave NO mixed sum: QF = COS_EVEN and
    // QB = COS_ODD are forced by the emitted structure alone.
    //
    // beta~^theta is SIN_EVEN (round 314): it carries one theta index, so it is
    // odd under theta -> pi - theta (round 219 S9/S13), and it vanishes on the
    // axis (rounds 247-255), which is a SIN basis; sin(2k theta) is the odd one
    // of the two.  That is the lattice's row 1, reached from a premise the
    // lattice does not share.  beta~^r keeps std_base().
    //
    // ⚠ The J = 0 seed is BLIND to this: beta~^theta = 0 there, so the
    // residuals cannot distinguish it from any other choice.  The derivation
    // and the lattice carry it; the acceptance test does not.  The header
    // records which, and the declaration check below is what keeps the run and
    // the analysis in step.
    // ---- --basis-probe: plant the fields, BEFORE the bases are declared -----
    // ⚠ ORDER.  The plant has to happen before std_base()/std_anti_base() so
    // that the values are transformed by the base under test.  Planting after
    // would transform them with whatever base was already set and the run would
    // report on the wrong object while looking identical.
    if (!basisprobe.empty()) {
        const bool ampB  = (basisprobe == "B" || basisprobe == "WB");
        const bool wrong = (basisprobe == "WA" || basisprobe == "WB");
        Scalar* fp[6] = {&PS, &PH, &QF, &BR, &BT, &QB};
        int fam[6] = {0, 0, 0, 0, 2, 1};   // 0 COS_EVEN, 1 COS_ODD, 2 SIN_EVEN
        const int cnt[6] = {ntheta, ntheta, ntheta, ntheta,
                            ntheta - 2, ntheta - 1};
        if (wrong) { fam[4] = 0; fam[5] = 2; }
        for (int q = 0; q < 6; q++)
            for (int d = 0; d < ndom; d++) {
                const Kadath::Domain* dm = space.get_domain(d);
                Val_domain& v = fp[q]->set_domain(d);
                v.allocate_conf();
                Index ix(dm->get_nbr_points());
                do {
                    const double th = dm->get_coloc(2)(ix(1));
                    double acc = 0.0;
                    for (int k = 0; k < cnt[q]; k++) {
                        const double a = ampB ? double(k + 1) : 1.0;
                        acc += a * (fam[q] == 0 ? std::cos(2.0 * k * th)
                                  : fam[q] == 1 ? std::cos((2.0 * k + 1.0) * th)
                                                : std::sin(2.0 * (k + 1) * th));
                    }
                    v.set(ix) = acc;
                } while (ix.inc());
            }
        emit("FJP_basis_probe_ampB", ampB ? 1.0 : 0.0);
        emit("FJP_basis_probe_wrong", wrong ? 1.0 : 0.0);
        if (rank == 0)
            std::cout << "# ⚠ --basis-probe " << basisprobe
                      << ": the six fields are PLANTED.  No residual, no"
                         " acceptance test and no Jacobian VALUE from this run"
                         " means anything -- only the inner-row RHS does.\n";
    }
    for (Scalar* s : {&PS, &PH, &QF, &BR, &RR, &ST, &CT, &C2, &H2, &L2,
                      &CX, &SQ, &T7, &ONE})
        s->std_base();
#ifdef TRUMPET_CHI_UNKNOWN
    // ⚠ COS_EVEN, and that is the DECLARED basis (FIELD_BASIS['CH'] in
    // finiteJ_emit.py, derived there): chi is a scalar whose defining product is
    // (COS_EVEN)*(COS_EVEN*COS_EVEN + SIN_EVEN*SIN_EVEN), both halves COS_EVEN.
    // It does NOT vanish on the axis -- beta^theta cot theta is finite there --
    // which is why AXIS_ORDER['CH'] is 0 and no divsint may be applied to it.
    // ⚠ THE PHYSICAL CH SEED, from the backbone's own closed form.
    //   chi = -(psi^2/Phibar)[beta^r d_r log(psi^2 r) + beta^th (d_th log psi^2
    //                                                            + cot th)]
    // and on the J = 0 backbone psi^2 = R/r exactly, so psi^2 r = R(r) and
    //   d_r log(psi^2 r) = R'/R = W/r   since dR/dr = R W / r,
    // with beta^th = 0 there.  So  CH = -(PS/PH) BR (W/r), which is a closed
    // form in quantities this run already has -- no quadrature, no fit.
    //
    // ⚠ THAT THIS IS A SECOND TRANSCRIPTION IS THE POINT, NOT A DEFECT.  The
    // first design planted the seed by READING D0048 (the emission's own
    // product) after registration, which would have made the J = 0 control a
    // tautology -- CH would equal the product by construction.  Seeding from the
    // independent closed form instead makes `max|CH - D0048|` a comparison of
    // two evaluations that can disagree, which is what research asked for: the
    // new unknown must reproduce the product on the backbone TO THE TRUNCATION.
    //
    // ⚠ The compactified domain's last node has r = infinity, where W/r -> 0;
    // guarded, because a non-finite seed propagates into every def.
    for (int d = 0; d <= dtop; d++) {
        Val_domain& vc = CH.set_domain(d);
        Index ix(space.get_domain(d)->get_nbr_points());
        do {
            const double rr = t.pts[d][ix(0)].r;
            const double W = t.pts[d][ix(0)].W;
            double v = 0.0;
            if (std::isfinite(rr) && rr > 0.0) {
                const double ps = PS(d)(ix), ph = PH(d)(ix), br = BR(d)(ix);
                if (std::isfinite(ps) && std::isfinite(ph) && ph != 0.0
                    && std::isfinite(br))
                    v = -(ps / ph) * br * (W / rr);
            }
            vc.set(ix) = std::isfinite(v) ? v : 0.0;
        } while (ix.inc());
    }
    // ⚠ --chi-seed OVERRIDES the backbone closed form with D0048 evaluated on a
    // STATE by an earlier run.  Planted HERE, before add_var, which is the only
    // place the system will honour it (see the --chi-seed declaration).
    if (!chiseed.empty()) {
        std::ifstream cf(chiseed);
        if (!cf) {
            if (rank == 0)
                std::cerr << "FATAL: cannot open --chi-seed " << chiseed << "\n";
            MPI_Finalize();
            return 5;
        }
        long got = 0;
        std::string ln;
        while (std::getline(cf, ln)) {
            if (ln.empty() || ln[0] != 'D') continue;
            std::istringstream is(ln);
            std::string tag, nm;
            int d, ir, ith;
            // ⚠ r AND theta ARE READ AS STRINGS AND DISCARDED.  The compactified
            // domain's last node has r = "inf" in the dump (NOTES 10.4), and
            // `>> double` on that token fails on this libstdc++ -- which lost
            // exactly the 13 theta points at that node, 1066 of 1079, and the
            // count check is what caught it.  The values are addressed by
            // (d, ir, ith); the coordinates are decoration.
            std::string rr, tt;
            double v;
            if (!(is >> tag >> nm >> d >> ir >> ith >> rr >> tt >> v)) continue;
            if (nm != "P48" || d < 0 || d > dtop) continue;
            Index ix(space.get_domain(d)->get_nbr_points());
            for (int k = 0; k < ir + ith * space.get_domain(d)->get_nbr_points()(0); k++)
                ix.inc();
            CH.set_domain(d).set(ix) = std::isfinite(v) ? v : 0.0;
            got++;
        }
        // ⚠ Round 236: a flag that silently does nothing produces a number, not
        // an error.  The count is asserted against the grid, not trusted.
        long want = 0;
        for (int d = 0; d <= dtop; d++) {
            const Kadath::Domain* dm = space.get_domain(d);
            want += static_cast<long>(dm->get_nbr_points()(0))
                    * dm->get_nbr_points()(1);
        }
        if (rank == 0)
            std::cout << "#  --chi-seed: " << got << " of " << want
                      << " points planted from " << chiseed
                      << (got == want ? "  OK\n" : "  ⚠ INCOMPLETE\n");
        emit("FJP_chi_seed_points", double(got));
        emit("FJP_chi_seed_want", double(want));
        if (got != want) {
            if (rank == 0)
                std::cerr << "FATAL: --chi-seed planted " << got << " of "
                          << want << " points\n";
            MPI_Finalize();
            return 5;
        }
    }
    CH.std_base();
#endif
    BT.std_anti_base(1);                       // SIN_EVEN, read back below
    if (breakbasis)
        QB.std_base();                         // deliberately not the declared one
    else
        QB.std_anti_base();                    // COS_ODD, read back below

    // is the seed the one that was verified?  R(throat) must be 3M/2.
    {
        double rmin = 1e300, Rmin = 0.0;
        for (std::size_t i = 0; i < t.pts[0].size(); i++)
            if (t.pts[0][i].r < rmin) { rmin = t.pts[0][i].r; Rmin = t.pts[0][i].Rr * rmin; }
        emit("FJP_R_at_inner", Rmin);
        emit("FJP_R_over_1p5M", Rmin / (1.5 * M));
    }

    // ---- THE SEED PERTURBATION -------------------------------------------
    if (seedperturb != 0.0) {
        const std::pair<const char*, Scalar*> unk[] = {
            {"PS", &PS}, {"PH", &PH}, {"QF", &QF},
            {"BR", &BR}, {"BT", &BT}, {"QB", &QB}};
        double supmax = 0.0;
        // pass 1: the raw profile's sup norm over the probed domains
        for (int pass = 0; pass < 2; pass++) {
            for (const auto& f : unk) {
                if (pfield != "all" && pfield != f.first)
                    continue;
                double wgt = 1.0;
                if (!pmix.empty()) {
                    std::stringstream ss(pmix);
                    std::string tok;
                    for (int q = 0; q <= (&f - unk); q++)
                        if (!std::getline(ss, tok, ','))
                            tok = "1";
                    wgt = std::stod(tok);
                }
                const bool isBT = std::string(f.first) == "BT";
                const bool isQB = std::string(f.first) == "QB";
                for (int d = 0; d <= dtop; d++) {
                    const Kadath::Domain* dm = space.get_domain(d);
                    Index ix(dm->get_nbr_points());
                    do {
                        const double rr = t.pts[d][ix(0)].r;
                        // ⚠ THE RADIAL PROFILE IS NOT COMPACT-SAFE.  0.15 r^2
                        // at the compact domain's r = infinity node makes the
                        // pass-0 sup norm infinite, so pass 1 plants
                        // eps * v / inf = 0 EVERYWHERE and NaN at the node
                        // itself -- the flag silently perturbs nothing.  Same
                        // guard the explicit throat profiles already carry.
                        if (!std::isfinite(rr))
                            continue;
                        const double th = dm->get_coloc(2)(ix(1));
                        const double rad = 0.5 + 0.35 * rr + 0.15 * rr * rr;
                        double ang;
                        if (isBT)                       // SIN_EVEN
                            ang = std::sin(2.0 * th) + 0.4 * std::sin(4.0 * th);
                        else if (isQB)                  // COS_ODD
                            ang = std::cos(th) + 0.3 * std::cos(3.0 * th);
                        else                            // COS_EVEN
                            ang = 0.6 + 0.3 * std::cos(2.0 * th)
                                  + 0.1 * std::cos(4.0 * th);
                        const double v = rad * ang;
                        if (pass == 0)
                            supmax = std::max(supmax, std::fabs(v));
                        else
                            f.second->set_domain(d).set(ix) +=
                                wgt * seedperturb * v / supmax;
                    } while (ix.inc());
                }
            }
            if (pass == 0 && supmax == 0.0)
                break;
        }
        emit("FJP_seed_perturb", seedperturb);
        if (rank == 0)
            std::cout << "#   seed perturbed: field " << pfield
                      << ", sup-norm " << seedperturb << "\n";
    }

    // ⚠ Read the bases back rather than trusting the std_base_* call: the
    // whole round turns on which basis each field actually carries, and
    // std_base_*_spher() throws on this space, so "the call exists" is not
    // evidence that it did what is wanted.
    {
        auto bname = [](int b) {
            switch (b) {
                case COS_EVEN: return "COS_EVEN";
                case COS_ODD: return "COS_ODD";
                case SIN_EVEN: return "SIN_EVEN";
                case SIN_ODD: return "SIN_ODD";
                default: return "other";
            }
        };
        const std::pair<const char*, Scalar*> fl[] = {
            {"PS", &PS}, {"PH", &PH}, {"QF", &QF},
            {"BR", &BR}, {"BT", &BT}, {"QB", &QB}, {"ones", &ONE}};
        int mismatch = 0;
        for (const auto& f : fl) {
            const Array<int>* b1 =
                (*f.second)(dtop).get_base().get_base_1d(1);
            const std::string got = b1 ? bname((*b1)(0)) : "none";
            // ⚠ and it must be what the emitter's basis lattice ASSUMED.  That
            // analysis decided every sum ordering in the header; if the run
            // registers a different basis the analysis describes a different
            // system, and nothing downstream means what it says.
            std::string want;
            for (const auto& d : Trumpet::finiteJ_declared_basis())
                if (std::string(d.first) == f.first)
                    want = d.second;
            const bool bad = !want.empty() && want != got;
            if (bad)
                mismatch++;
            if (rank == 0)
                std::cout << "#   registered basis  " << std::left
                          << std::setw(6) << f.first << std::setw(10) << got
                          << (want.empty() ? "" : "declared " + want)
                          << (bad ? "   <-- MISMATCH" : "") << "\n";
        }
        emit("FJP_basis_mismatch", static_cast<double>(mismatch));
        if (mismatch && !nobasischeck) {
            if (rank == 0)
                std::cerr << "FATAL: " << mismatch
                          << " field(s) registered in a basis the emission was "
                             "not analysed under\n";
            MPI_Finalize();
            return 4;
        }
        if (mismatch && rank == 0)
            std::cout << "#   --no-basis-check: continuing under the WRONG "
                         "basis so the residuals can be read there too\n";
    }

    if (!gridout.empty() && rank == 0) {
        std::ofstream g(gridout);
        g << "# collocation grid written by finiteJ_probe --dump-grid\n";
        g << "version 1\nntheta " << ntheta << "\nndom " << ndom
          << "\ndtop " << dtop << "\n";
        g << std::setprecision(17);
        for (int d = 0; d <= dtop; d++) {
            const Kadath::Domain* dm = space.get_domain(d);
            Index ix(dm->get_nbr_points());
            do {
                g << "grid " << d << " " << ix(0) << " " << ix(1) << " "
                  << t.pts[d][ix(0)].r << " " << dm->get_coloc(2)(ix(1)) << "\n";
            } while (ix.inc());
        }
        std::cout << "# grid written to " << gridout << "\n";
    }

    // ---- --jac-match: the angular basis profiles and the unknowns ---------
    // ⚠ DECLARED BEFORE `syst` ON PURPOSE.  add_var(name, double&) stores a
    // POINTER; the storage must outlive the system, and a vector that is later
    // grown would invalidate every one of them silently.  Sized once, here.
    const int NCE = ntheta;        // COS_EVEN   cos(2k th),      k = 0 .. nt-1
    const int NCO = ntheta - 1;    // COS_ODD    cos((2k+1) th),  k = 0 .. nt-2
    const int NSE = ntheta - 2;    // SIN_EVEN   sin(2k th),      k = 1 .. nt-2
    const int NG  = 4 * NCE + NSE + NCO;   // grade-0   = 6nt - 3
    const int NT  = NSE + NCO;             // tower     = 2nt - 3
    // ⚠ THE GRADE-1 BLOCK IS ANOTHER NG, and it is sized here with the rest
    // for the same reason: add_var stores a POINTER, and a vector grown later
    // would invalidate every one of them silently.  Research round 511 ruling
    // (1): G_1 is the same Kadath object as G, extended -- a second block of
    // scalar matching unknowns in the same parity classes, not fields on a
    // 1-D angular space and not add_cst.
    std::vector<double> matchv(NG + NT + NG, 0.0);
    // ⚠ ROUND 195: SEED THE MATCHING AMPLITUDES.  Round 194's residual was read
    // with every G at zero, where the C1 row's residual is dr(F) itself whatever
    // the profile is -- so restoring the radial factor could not have shown up
    // there.  The amplitudes must carry the seed's own values for the residual
    // to mean anything.  At J = 0 the seed is theta-independent, so only mode 0
    // of each COS_EVEN family is nonzero and BT, QB are zero exactly.
    //   G = F(r_m) / r_m^{g0},  matchv laid out PS, PH, QF, BR, BT, QB in the
    //   registration order of the row loop below.
    if (jacmatch && seedmatch) {
        const double rm  = t.pts[0][0].r;
        const double Wm  = t.pts[0][0].W;
        const double Rm  = t.pts[0][0].Rr;
        const double om  = t.pts[0][0].oor;
        const double ne  = nexp;
        matchv[0]        = Rm * rm;                             // PS,  g0 = -1
        matchv[NCE]      = Wm * Rm * std::pow(rm, 1.0 - ne);    // PH,  g0 = n-1
        matchv[2 * NCE]  = 0.0;                                 // QF,  q = 0
        matchv[3 * NCE]  = (C * om * om / (Rm * Rm * Rm)) / rm; // BR,  g0 = +1
        if (rank == 0)
            std::cout << std::setprecision(17)
                      << "# seed-match at r_m = " << rm
                      << "  G_PS = " << matchv[0]
                      << "  G_PH = " << matchv[NCE]
                      << "  G_BR = " << matchv[3 * NCE] << "\n";
        // ⚠ THE GRADE-1 AMPLITUDES, from round 176's monomial curve
        // (throat_wrho.py:188-190), research round 513 ruling 3:
        //     u1 = sqrt2/4 w0    P1 = -sqrt2/3 w0    b1 = -3 sqrt2/2 w0
        //     B2 = Q1 = q1 = 0   at J = 0,   w0 = W0 on the deep layout
        // The UNKNOWNS are not those: the reconstruction in throatth_eqs.hpp
        // reads psi2's Ser as e^{2uh}/r and e^{2uh}/r . 2u1, and Phb's as
        // Ph_/r and Ph_ P1/r, so
        //     H_PS = 2 G_PS u1     H_PH = G_PH P1     H_BR = G_BR b1
        // and the other three are zero.  At J = 0 the seed is spherical, so
        // only mode 0 of each COS_EVEN family carries anything and ACE00 = 1,
        // which makes the mode-0 coefficient the value itself.
        // ⚠ Round 195's lesson: a residual read with the grade-1 amplitudes at
        // zero measures the grade-1 truncation, not the system.
        if (jacrec) {
            const double q2 = std::sqrt(2.0);
            const double w0 = t.W0;   // the layout parameter, not a point value
            // ⚠ ROUND 224: THREE OF THE FIVE FORMS CARRY n, AND THE SHOOTING
            // MOVES n.  Re-deriving throat_wrho.py's STEP 1/STEP 2 with n
            // symbolic (the recursion is rho dR/drho = R W / n, so n is in it
            // from the start) gives
            //     u1 = sqrt2 w0 / 4                       <- n-INDEPENDENT
            //     P1 = -sqrt2 n w0 / (6n - 3 sqrt2)
            //     b1 = 3 w0 (-6 sqrt2 n^4 + 22 n^3 - 15 sqrt2 n^2 + 9n - sqrt2)
            //             / (12 n^4 - 22 sqrt2 n^3 + 30 n^2 - 9 sqrt2 n + 2)
            // and all three reduce to round 176's recorded values at n = sqrt2
            // EXACTLY (checked symbolically).  The e_1 normalisation that makes
            // W's rho^1 coefficient exactly w0 is e1 = sqrt2 w0 / 2, which is
            // itself n-free -- that is why u1 has no n.
            // The effect is small: dP1/dn = w0/3 and db1/dn = 0 at n = sqrt2,
            // so over the shooting's |delta-n| ~ 1e-3 this is a 0.1% change in
            // P1 and a stationary b1.  Implemented because "re-evaluate the
            // seed at each n_k" means this, not because it is large.
            const double ne = nexp;
            const double u1 = q2 / 4.0 * w0;
            const double P1 = -q2 * ne * w0 / (6.0 * ne - 3.0 * q2);
            const double b1 = 3.0 * w0
                * (-6.0 * q2 * ne * ne * ne * ne + 22.0 * ne * ne * ne
                   - 15.0 * q2 * ne * ne + 9.0 * ne - q2)
                / (12.0 * ne * ne * ne * ne - 22.0 * q2 * ne * ne * ne
                   + 30.0 * ne * ne - 9.0 * q2 * ne + 2.0);
            matchv[NG + NT + 0]           = 2.0 * matchv[0] * u1;
            matchv[NG + NT + NCE]         = matchv[NCE] * P1;
            matchv[NG + NT + 3 * NCE]     = matchv[3 * NCE] * b1;
            if (physh) {
                // H_phys = H_seed r_m^-n; G <- G - H_phys r_m^n = G - H_seed.
                // Same order as rescale_h.py: H from the old G, then G.
                const double rmn = std::pow(rm, ne);
                const int fam[3] = {0, NCE, 3 * NCE};   // PS, PH, BR
                for (int f : fam) {
                    const double hs = matchv[NG + NT + f];
                    matchv[NG + NT + f] = hs / rmn;
                    matchv[f] -= hs;
                }
                emit("FJPJ_physical_h", 1.0);
                emit("FJPJ_physical_h_rmn", rmn);
                if (rank == 0)
                    std::cout << std::setprecision(17)
                              << "#  --physical-h: H x r_m^-n (r_m^n = " << rmn
                              << "), G <- G - H r_m^n:  G_PS = " << matchv[0]
                              << "  G_PH = " << matchv[NCE]
                              << "  G_BR = " << matchv[3 * NCE] << "\n";
            }
            emit("FJPJ_seed_grade1", 1.0);
            emit("FJPJ_seed_w0", w0);
            if (rank == 0)
                std::cout << std::setprecision(17)
                          << "# seed-grade1  w0 = " << w0 << "  nexp = " << ne
                          << "  H_PS = " << matchv[NG + NT]
                          << "  H_PH = " << matchv[NG + NT + NCE]
                          << "  H_BR = " << matchv[NG + NT + 3 * NCE] << "\n";
        }
        emit("FJPJ_seed_match", 1.0);
    }
    // ⚠ GENERIC AMPLITUDES -- A RANK INSTRUMENT, AND NOTHING ELSE.  The
    // reconstruction in throatth_eqs.hpp reads the AMPLITUDES, not the fields,
    // so --seed-perturb cannot reach it: it makes qbar nonzero as a field and
    // leaves S_QF and S_QB at exactly zero, which is the quantity the grade-1
    // block is linearised on.  A rank read wants generic data (the throat
    // rank scripts have used fixed seeds since round 77), and this fills every
    // G and H entry with a deterministic value of the requested size.  The
    // state is NOT physical and no residual from it means anything.
    if (jacmatch && ampgen != 0.0) {
        unsigned st = 4242u;
        for (std::size_t i = 0; i < matchv.size(); i++) {
            st = st * 1664525u + 1013904223u;
            const double u = double(st >> 8) / double(1u << 24) - 0.5;
            matchv[i] += ampgen * u;
        }
        emit("FJPJ_amp_generic", ampgen);
        if (rank == 0)
            std::cout << "#  --amp-generic " << ampgen
                      << ": every G and H amplitude perturbed; RANK ONLY\n";
    }
    std::vector<std::unique_ptr<Scalar>> mbase;
    auto mbname = [](const char* pre, int k) {
        char b[16]; std::snprintf(b, sizeof b, "%s%02d", pre, k);
        return std::string(b);
    };
    std::vector<std::string> mbnames;
    if (jacmatch) {
        // fill cos(2k th), cos((2k+1) th), sin(2k th) at the collocation
        // points and tag each with the basis it belongs to.  ⚠ std_base() is
        // COS_EVEN, std_anti_base() COS_ODD and std_anti_base(1) SIN_EVEN --
        // the same three calls the six fields' own declarations use, so the
        // profiles are in the SAME spaces the tau projection reads.
        // ⚠ ROUND 195: THE RADIAL FACTOR, restored.  A theta-only profile is the
        // grade-0 amplitude WITH ITS RADIAL FACTOR DROPPED, which is why dr
        // annihilated it and the C1 row collapsed to dr(F) = 0 (round 194).
        // The factors are read from scripts/throat_th2.py:66-71 -- not from
        // round 428's table, which lists the amplitudes without them:
        //
        //   psi2 = e^{2uh}/r        -> g0 = -1        qf = qh       -> g0 = 0
        //   Phb  = (Ph_/r) rho      -> g0 = n - 1     br = bh r     -> g0 = +1
        //   bt   = Bh rho           -> g0 = n         Qb = Qh       -> g0 = 0
        //
        // and they agree with round 188's INDEPENDENTLY MEASURED grade set
        // (-1, n-1, 0, 1, n, 0) off the assembly's own Ser objects.
        // Stage 1 fixes n = sqrt(2) exactly, per research round 487 ruling (3).
        //
        // ⚠ The profiles are per FIELD now, not per angular family, because the
        // radial factor differs between fields sharing a family.  The plain
        // COS_EVEN family MCE is kept as well, with no radial factor, because
        // --jac-match-break needs it as the wrong-family control (round 172).
        const double NEXP = nexp;
        struct Fam { const char* pre; int n; int kind; double g0; };
        // ⚠ SIX MORE SETS FOR --jac-recursion, and both kinds are needed.
        //   A**  the PURE ANGULAR profile of each family, no radial factor.
        //        The theta-ODE rows live at ONE radius, so a radial factor
        //        there is a constant folded into an unknown that already
        //        carries it.  ACE duplicates MQF and ACO duplicates MQB by
        //        value; they are separate names so a later change to a field's
        //        g0 cannot silently move the angular basis with it.
        //   N**  the GRADE-1 radial profile, r^{g0 + n}: one step of the
        //        rho = r^n series (throat_th2.py's Ser index), which is what
        //        lets C0 and C1 determine G and G_1 TOGETHER.
        const Fam fam[13] = { {"MPS", NCE, 0, -1.0},      {"MPH", NCE, 0, NEXP - 1.0},
                             {"MQF", NCE, 0,  0.0},      {"MBR", NCE, 0,  1.0},
                             {"MBT", NSE, 2,  NEXP},     {"MQB", NCO, 1,  0.0},
                             {"MCE", NCE, 0,  0.0},
                             {"ACE", NCE, 0,  0.0},      {"ASE", NSE, 2,  0.0},
                             {"ACO", NCO, 1,  0.0},
                             {"NPS", NCE, 0, NEXP - 1.0},
                             {"NPH", NCE, 0, 2.0 * NEXP - 1.0},
                             {"NQF", NCE, 0, NEXP} };
        const Fam fam2[3] = { {"NBR", NCE, 0, NEXP + 1.0},
                              {"NBT", NSE, 2, 2.0 * NEXP},
                              {"NQB", NCO, 1, NEXP} };
        std::vector<Fam> allfam(fam, fam + 13);
        allfam.insert(allfam.end(), fam2, fam2 + 3);
        for (const Fam& f : allfam)
            for (int j = 0; j < f.n; j++) {
                const int k = (f.kind == 2) ? j + 1 : j;
                auto sp = std::make_unique<Scalar>(space);
                for (int d = 0; d < ndom; d++) {
                    const Kadath::Domain* dm = space.get_domain(d);
                    Val_domain& v = sp->set_domain(d);
                    v.allocate_conf();
                    Index ix(dm->get_nbr_points());
                    do {
                        const double th = dm->get_coloc(2)(ix(1));
                        const double ang = (f.kind == 0) ? std::cos(2.0 * k * th)
                                         : (f.kind == 1) ? std::cos((2.0*k + 1.0) * th)
                                                         : std::sin(2.0 * k * th);
                        // ⚠ the compact domain carries r = infinity, where
                        // r^{+g0} is inf and r^{-|g0|} is 0.  The profiles are
                        // read ONLY at the inner face of domain 0, so the
                        // compact domain is filled with 0 rather than with an
                        // infinity that would propagate as a NaN.
                        const double rr = t.pts[d][ix(0)].r;
                        // ⚠ A ZERO-GRADE PROFILE IS FINITE AT r = infinity.
                        // The guard below exists because r^{g0} is inf or 0
                        // there; with g0 = 0 there is no radial factor at all,
                        // and zeroing it anyway makes the pure-angular A**
                        // families vanish in the compact domain -- which makes
                        // SPS zero and log(SPS) meaningless in exactly the
                        // domain the theta-ODE defs are also registered in.
                        const double rad = (f.g0 == 0.0) ? 1.0
                                         : ((!std::isfinite(rr)) ? 0.0
                                            : std::pow(rr, f.g0));
                        v.set(ix) = rad * ang;
                    } while (ix.inc());
                }
                if      (f.kind == 0) sp->std_base();
                else if (f.kind == 1) sp->std_anti_base();
                else                  sp->std_anti_base(1);
                // ⚠ NAMED BY j, NOT BY k.  k carries the SIN_EVEN offset for
                // the angular function only; the row loop indexes profiles by j
                // from 0 for every family.  The first version named by k and
                // registered MBT01 where the row asked for MBT00 -- "Unknown
                // operator MBT00" at parse time, which is the good failure: it
                // cannot assemble a wrong system, only refuse to assemble.
                mbnames.push_back(mbname(f.pre, j));
                mbase.push_back(std::move(sp));
            }
    }

    std::cout << "# building system" << std::endl;
    System_of_eqs syst(space, 0, dtop);
    std::cout << "# system built" << std::endl;
    // ⚠ THE BACKBONE'S OWN TAIL, kept as a CONSTANT so the outer row can
    // subtract it.  Research round 345: the row is not "Phb has no 1/r tail" but
    // "Phb's 1/r tail equals Phb_P's", because psi^2 = (R/r)e^{2 eps U} and
    // R -> r + M_P + ..., so R/r carries its own 1/r piece.  Writing the first
    // imposes a condition wrong by exactly the backbone's mass term -- and it
    // would assemble cleanly and solve a different problem.  PHP is filled from
    // the same W*R/r the seed uses, BEFORE any perturbation touches PH.
    // ⚠ PHP IS FILLED BEFORE ANY SCALING TOUCHES PH.  Ordering is the whole
    // point: if PHP were copied afterwards it would capture the scaled field
    // and the outer row's backbone subtraction would silently cancel whatever
    // the scaling did.  Round 137's first version did exactly that.
    Scalar PHP(space);
    for (int d = 0; d < ndom; d++) {
        Val_domain& v = PHP.set_domain(d);
        v.allocate_conf();
        Index ix(space.get_domain(d)->get_nbr_points());
        do { v.set(ix) = (d > dtop) ? 0.0 : PH(d)(ix); } while (ix.inc());
    }
    PHP.std_base();

    // ⚠ alpha_P IS W, AND THAT IS READ OFF THE SEED RATHER THAN ASSUMED.  The
    // seed above sets  vps = R/r  and  vph = W R/r, so Phb_P = alpha_P psi_P^2
    // gives alpha_P = W.  ALP is filled from t.pts[d][i].W directly -- the same
    // table the seed uses -- and BEFORE any perturbation or scaling, for the
    // reason PHP is: a reference captured after the perturbation is not a
    // reference (round 137).
    Scalar ALP(space);
    for (int d = 0; d < ndom; d++) {
        Val_domain& v = ALP.set_domain(d);
        v.allocate_conf();
        Index ix(space.get_domain(d)->get_nbr_points());
        do {
            v.set(ix) = (d > dtop) ? 1.0 : t.pts[d][ix(0)].W;
        } while (ix.inc());
    }
    ALP.std_base();

    // ⚠ THE EXPLICIT THROAT PROFILES, applied AFTER PHP is filled so the outer
    // row's backbone subtraction is unaffected, and after the seed so they ADD
    // to it rather than replace it.
    if (btamp != 0.0 || bramp != 0.0 || brB != 0.0) {
        for (int d = 0; d <= dtop; d++) {
            const Kadath::Domain* dm = space.get_domain(d);
            Val_domain& vb = BT.set_domain(d);
            Val_domain& vr = BR.set_domain(d);
            Index ix(dm->get_nbr_points());
            do {
                const double rr = t.pts[d][ix(0)].r;
                const double tt = dm->get_coloc(2)(ix(1));
                if (!std::isfinite(rr))
                    continue;             // the compact domain's outermost node
                if (btamp != 0.0) {
                    double prof = std::pow(rr, btpow);
                    if (bttaper)
                        prof /= std::pow(1.0 + rr * rr, btpow);
                    vb.set(ix) += btamp * prof * std::sin(btang * tt) / 2.0;
                }
                if (bramp != 0.0 || brB != 0.0)
                    vr.set(ix) += std::pow(rr, brpow)
                                  * (bramp * std::sin(tt) * std::sin(tt)
                                     + brB / std::sqrt(2.0));
            } while (ix.inc());
        }
        emit("FJP_bt_prof_pow", btpow);
        emit("FJP_bt_prof_amp", btamp);
        emit("FJP_bt_prof_ang", btang);
        emit("FJP_bt_prof_taper", bttaper ? 1.0 : 0.0);
        emit("FJP_br_prof_pow", brpow);
        emit("FJP_br_prof_A", bramp);
        emit("FJP_br_prof_B", brB);
    }

    if (throat) {
        const double q2 = std::sqrt(2.0);
        for (int d = 0; d <= dtop; d++) {
            const Kadath::Domain* dm = space.get_domain(d);
            Val_domain& vp = PS.set_domain(d);
            Val_domain& vb = BT.set_domain(d);
            Val_domain& vr = BR.set_domain(d);
            Index ix(dm->get_nbr_points());
            do {
                const double rr = t.pts[d][ix(0)].r;
                if (!std::isfinite(rr))
                    continue;
                const double tt = dm->get_coloc(2)(ix(1));
                const double s2 = std::sin(tt) * std::sin(tt);
                const double du = (1.0 / (2.0 * q2)) * tha0 * std::pow(rr, q2)
                                  * (thA * s2 + thD);
                const double amp = tha0 * thb0 * std::pow(rr, q2 + 1.0);
                vp.set(ix) *= std::exp(2.0 * du);
                vb.set(ix) += amp * thA * std::sin(tt) * std::cos(tt);
                vr.set(ix) += amp * (thA * s2 + thB / q2);
                if (thC != 0.0 || thE != 0.0) {
                    const double dvp = (1.0 / q2) * tha0 * thP0
                                       * std::pow(rr, 2.0 * q2 - 1.0)
                                       * (thC * s2 + thE);
                    PH.set_domain(d).set(ix) *= std::exp(dvp);
                }
            } while (ix.inc());
        }
        emit("FJP_throat_A", thA);
        emit("FJP_throat_B", thB);
        emit("FJP_throat_D", thD);
        emit("FJP_throat_a0", tha0);
        emit("FJP_throat_b0", thb0);
        emit("FJP_throat_C", thC);
        emit("FJP_throat_E", thE);
        emit("FJP_throat_P0", thP0);
    }

    if (qlog != 0.0) {
        for (int d = 0; d <= dtop; d++) {
            const Kadath::Domain* dm = space.get_domain(d);
            Val_domain& v = QF.set_domain(d);
            Index ix(dm->get_nbr_points());
            do {
                const double rr = t.pts[d][ix(0)].r;
                if (std::isfinite(rr) && rr > 0.0)
                    v.set(ix) += qlog * std::log(rr);
            } while (ix.inc());
        }
        emit("FJP_qlog", qlog);
    }

    // the constant-family walk (round 137), AFTER PHP is safe
    if (scalePS != 1.0 || scalePH != 1.0 || scaleBR != 1.0) {
        for (int d = 0; d <= dtop; d++) {
            Val_domain& a = PS.set_domain(d);
            Val_domain& b = PH.set_domain(d);
            Val_domain& c = BR.set_domain(d);
            Index ix(space.get_domain(d)->get_nbr_points());
            do { a.set(ix) *= scalePS; b.set(ix) *= scalePH;
                 c.set(ix) *= scaleBR; } while (ix.inc());
        }
        PS.std_base(); PH.std_base(); BR.std_base();
    }
    if (outerpert != 0.0) {
        // a KNOWN 1/r perturbation on top of the backbone tail
        for (int d = 0; d <= dtop; d++) {
            Val_domain& v = PH.set_domain(d);
            Index ix(space.get_domain(d)->get_nbr_points());
            do {
                double rr = t.pts[d][ix(0)].r, den = 1.0;
                for (int k = 0; k < outerpow; k++) den *= rr;
                v.set(ix) += outerpert / den;
            } while (ix.inc());
        }
        PH.std_base();
    }

    // ⚠ THE SIX UNKNOWNS ARE CONSTANTS FOR EVERY DIAGNOSTIC AND VARIABLES FOR
    // THE JACOBIAN.  add_cst is what makes the residual diagnostics a pure
    // evaluation -- nothing is solved for, so nothing can be silently adjusted.
    // A Jacobian needs them to be unknowns, and that is the ONLY thing
    // --dump-jacobian changes about the registration; the def chain, the bases
    // and the read-back contract are identical either way.
    // ⚠ ONE SWITCH, TWO FLAGS.  --newton needs the same registration as
    // --dump-jacobian (fields as variables); nothing else about the run may
    // differ, or the system solved would not be the system measured.
    const bool jacon = !jacdump.empty() || donewton;
    if (!jacon) {
    // ⚠ DUMPED WHERE THE SYSTEM SEES THEM, not where they are first built.
    // The first version of this sat before PHP -- and so before the throat
    // profiles, before --qlog and before the gauge walk -- so it wrote the bare
    // seed whatever was asked for, and the log detector's selftest recovered
    // 0 from a planted 1.  That is the fourth member of round 154's family in
    // eight rounds, and the selftest is why it was found in one run.
    if (!fieldsout.empty() && rank == 0) {
        std::ofstream ff(fieldsout);
        ff << "# field values written by finiteJ_probe --dump-fields\n";
        ff << "fields PS PH QF BR BT QB\n";
        ff << std::setprecision(17);
        const Scalar* fp[6] = {&PS, &PH, &QF, &BR, &BT, &QB};
        for (int d = 0; d <= dtop; d++) {
            Index ix(space.get_domain(d)->get_nbr_points());
            do {
                ff << "val " << d << " " << ix(0) << " " << ix(1)
                   << " " << t.pts[d][ix(0)].r
                   << " " << space.get_domain(d)->get_coloc(2)(ix(1));
                for (int q = 0; q < 6; q++)
                    ff << " " << (*fp[q])(d)(ix);
                ff << "\n";
            } while (ix.inc());
        }
        std::cout << "# fields written to " << fieldsout << "\n";
    }

        syst.add_cst("PS", PS);
        syst.add_cst("PH", PH);
        syst.add_cst("QF", QF);
        syst.add_cst("BR", BR);
        syst.add_cst("BT", BT);
        syst.add_cst("QB", QB);
#ifdef TRUMPET_CHI_UNKNOWN
        // ⚠ BOTH BRANCHES.  A first version registered CH only in the
        // add_var branch, and the production invocation carries --clean, which
        // takes THIS one -- so CH was never registered, the emitted defs
        // referenced a name the system did not have, and registration
        // segfaulted.  CH is a VARIABLE here too: with the six fields as
        // constants the system is ECHI on CH alone, which is exactly the J = 0
        // control this build exists to run.
        syst.add_var("CH", CH);
        if (rank == 0)
            std::cout << "#  --chi-unknown: CH registered as a VARIABLE"
                         " (--clean branch; the six fields are csts here)\n";
#endif
    } else {
        auto wanted = [](const std::string& list, const char* nm) {
            if (list == "all") return true;
            return (',' + list + ',').find(std::string(",") + nm + ",")
                   != std::string::npos;
        };
        const char* fn[6] = {"PS", "PH", "QF", "BR", "BT", "QB"};
        Scalar* fp[6] = {&PS, &PH, &QF, &BR, &BT, &QB};
        for (int q = 0; q < 6; q++) {
            if (wanted(jacfields, fn[q])) syst.add_var(fn[q], *fp[q]);
            else                          syst.add_cst(fn[q], *fp[q]);
        }
#ifdef TRUMPET_CHI_UNKNOWN
        // ⚠ CH is ALWAYS a variable, and deliberately not subject to
        // --jac-fields.  Its defining equation ECHI is registered for every
        // domain by the bulk loop, so making it a constant would leave ECHI as
        // a condition on nothing -- an over-determined system that reports a
        // residual instead of an error.  Round 236's rule.
        syst.add_var("CH", CH);
        if (rank == 0)
            std::cout << "#  --chi-unknown: CH registered as a VARIABLE"
                         " (COS_EVEN, std_base); ECHI comes from the emission's"
                         " own equation list\n";
#endif
    }
    syst.add_cst("RR", RR);
    syst.add_cst("ST", ST);
    syst.add_cst("CT", CT);
    syst.add_cst("C2", C2);
    syst.add_cst("H2", H2);
    syst.add_cst("L2", L2);
    syst.add_cst("CX", CX);
    syst.add_cst("SQ", SQ);
    syst.add_cst("T7", T7);
    // `ones` is a field the apps register themselves (BH2d/NS2d do the same);
    // the emitted monomials materialise against it.
    syst.add_cst("ones", ONE);
#ifdef TRUMPET_BULK2
    // ⚠ --amax-table: one weight field per equation, W(r) on domain 0 (Chebyshev series in domain 0's ln r, from
    // scripts/python_bulk.py --write-amax), 1 in every other domain.  STATIC: add_cst keeps a reference.
    static std::vector<std::unique_ptr<Scalar>> amaxw;
    std::map<std::string, bool> amaxhas;
    if (!amaxtable.empty()) {
        std::ifstream fin(amaxtable);
        std::string tag;
        double lo = 0, hi = 0;
        int deg = 0;
        fin >> tag >> lo >> hi >> deg;
        if (!fin || tag != "domain0") {
            std::cerr << "FATAL: --amax-table " << amaxtable << " unreadable\n";
            return 1;
        }
        std::string nm;
        while (fin >> nm) {
            std::vector<double> co(deg + 1);
            for (int q = 0; q <= deg; q++) fin >> co[q];
            amaxw.emplace_back(new Scalar(space));
            Scalar& W = *amaxw.back();
            double wmin = 1e300, wmax = -1e300;
            for (int d = 0; d < ndom; d++) {
                const Kadath::Domain* dm = space.get_domain(d);
                Val_domain& vw = W.set_domain(d);
                vw.allocate_conf();
                Index idx(dm->get_nbr_points());
                do {
                    double w = 1.0;
                    if (d == 0) {
                        const double rr = dm->get_radius()(idx);
                        const double x = 2.0 * std::log(rr / lo) / std::log(hi / lo) - 1.0;
                        double b1 = 0, b2 = 0;              // Clenshaw
                        for (int q = deg; q >= 1; q--) { const double t = 2 * x * b1 - b2 + co[q]; b2 = b1; b1 = t; }
                        w = x * b1 - b2 + co[0];
                        wmin = std::min(wmin, w); wmax = std::max(wmax, w);
                    }
                    vw.set(idx) = w;
                } while (idx.inc());
            }
            W.std_base();
            syst.add_cst(("W" + nm).c_str(), W);
            amaxhas[nm] = true;
            if (rank == 0)
                std::cout << "#  --amax-table: W" << nm << " registered, domain 0 range " << wmin << " .. " << wmax
                          << " (1 elsewhere)\n";
        }
    }
#endif
    syst.add_cst("PHP", PHP);
    syst.add_cst("ALP", ALP);
    syst.add_cst("DEL", sdefdelta);
    syst.add_cst("LAM", jacmixlam);
    // ⚠ JJ MUST MATCH THE DATA.  This was hardcoded to 0.0 -- correct for the
    // J = 0 seed, and silently wrong for manufactured data, which is generated
    // at J = 0.1.  Round 126 measured the consequence: D0022 = 6*JJ*sin^3, a def
    // containing NO field at all, disagreed with its twin by its entire value,
    // and that is what a constant mismatch looks like rather than a defect.
    // Round 125 read the whole effect as the emitted text and the twin
    // computing different functions; it was this.
    // ⚠ --jj WINS over the manufactured file's J, and the two are never mixed
    // silently: whichever is used is emitted as FJP_JJ.
    syst.add_cst("JJ", havejj ? jjoverride
                              : (mandata.empty() ? 0.0 : man_J));
    // ⚠ n IS A CONSTANT OF THE SYSTEM, not just a C++ double.  The theta-ODE
    // rows carry the throat exponent explicitly -- every grade is a power of
    // r^n and the recursion relations are n-dependent -- so it has to be a
    // name the parser knows.  Registered beside JJ and from the same nexp the
    // seed-match and the Fam table use, so one --nexp moves all three.
    syst.add_cst("nexp", nexp);
    emit("FJP_JJ", havejj ? jjoverride
                          : (mandata.empty() ? 0.0 : man_J));
    if (rank == 0 && !mandata.empty())
        std::cout << "#   JJ set from the manufactured data: " << man_J << "\n";
    if (tailout) {
        // ⚠ REGISTERED HERE, with the csts, so the defs exist before any
        // equation is parsed -- the same ordering rule the matching profiles
        // follow (round 172: the parser resolves names at add_eq time).
        syst.add_def("TAILPS = multr(PS)");
        syst.add_def("TAILPH = multr(PH)");
        syst.add_def("TAILPHP = multr(PHP)");
        syst.add_def("TAILQF = multr(QF)");
        if (rank == 0) std::cout << "#  --tail-readout: 4 defs registered\n";
    }
    std::cout << "# csts registered" << std::endl;

    // ---- SMOKE: do the operators work at all, and where is the size limit? --
    // Two separate questions, and a 2254-character def failing answers neither
    // on its own.  Small defs with EXACT known answers first.
    if (!clean) {
        struct Sm { const char* def; const char* what; };
        const std::vector<Sm> sm = {
            {"S1 = dr(RR)", "dr(r) = 1"},
            {"S2 = (ST)^2 + (CT)^2", "sin^2 + cos^2 = 1"},
            {"S3 = dt(CT) + ST", "dt(cos) + sin = 0"},
            {"S4 = dt(ST) - CT", "dt(sin) - cos = 0"},
            {"S5 = dt(dt(CT)) + CT", "ddt(cos) + cos = 0"},
            {"S6 = dt(dt(C2)) + 4 * C2", "ddt(cos2th) + 4cos2th = 0  <- IS representable"},
            {"S7 = dt(C2)", "dt(cos2th) = -2 sin2th, max |.| = 2"},
            // ---- THE VOCABULARY BATTERY: which constructs are usable? ----
            {"V1 = divsint(multsint(C2)) - C2", "divsint . multsint = id"},
            {"V2 = divr(multr(C2)) - C2",       "divr . multr = id"},
            {"V3 = lap2(H2)",                   "lap2(r^2 cos2th) = 0 EXACT"},
            {"V4 = lap(L2)",                    "lap(r^2 P_2) = 0 EXACT"},
            {"V5 = multr(divr(C2)) - C2",       "multr . divr = id"},
            {"V6 = lap2(C2) * (RR)^2 + 4 * C2", "lap2(cos2th) = -4cos2th/r^2"},
            // ---- cot(theta) WITHOUT a cos(theta) field.  There is no
            // Ope_mult_cost, but d_th(sin X) = cos X + sin d_th X gives
            //     cot(theta) X = divsint(dt(multsint(X))) - dt(X)
            // in operators that have just been verified individually.
            {"V7 = divsint(dt(multsint(SQ))) - dt(SQ) - T7", "cot(th)*sin^2 = sin2th/2, built"},
            {"V8 = divrsint(multrsint(C2)) - C2", "divrsint . multrsint = id"},
            // ---- cos(theta) * X WITHOUT a division, and so without the axis
            // precondition the cot construction carries.  From
            //     d_th(sin X) = cos X + sin d_th X
            //     =>  cos(theta) X == dt(multsint(X)) - multsint(dt(X))
            // Distribution (round 317) makes monomials combine, so a term can
            // acquire cos with no matching 1/sin, and this is the only way to
            // emit that in a vocabulary with no Ope_mult_cost.
            //
            // ⚠ It is NOT checked against the CT field: round 103 measured that
            // cos(theta) is outside the scalar COS_EVEN span, so CT holds an
            // alias and the comparison would be against garbage.  The check is
            // pointwise against a value computed here, below.
            {"V9 = dt(multsint(ones)) - multsint(dt(ones))", "cos(th) * 1, built"},
            {"V10 = dt(multsint(C2)) - multsint(dt(C2))", "cos(th) * cos2th, built"},
            // ---- research round 297 ruling (3): measure the add_cst exposure
            // rather than argue it.  PS is theta-INDEPENDENT in the seed, the
            // same shape as L0ModelT's 39 coefficient fields.
            {"S8 = dt(PS)", "dt of a theta-independent add_cst field = 0"},
            {"S9 = dt(dt(PS))", "ddt of the same = 0"},
            // ---- T8 bisection: with BT = 0 and BR theta-independent, only
            // these survive.  Each is compared with sympy at the probe point.
            {"P1 = dr(dr(BR))",                   "P1"},
            {"P2 = divr(divr(multr(dr(BR))))",    "P2 = dr(BR)/r"},
            {"P3 = divr(divr(multr(divr(BR))))",  "P3 = BR/r^2"},
            {"P4 = divr(dr(BR))",                 "P4 = dr(BR)/r"},
            {"P5 = divr(divr(BR))",               "P5 = BR/r^2"},
            {"P6 = multr(divr(BR)) - BR",         "P6 = 0"},
            {"P7 = divr(multr(BR)) - BR",         "P7 = 0"},
            // the T8 terms that MUST vanish on this seed (BT = 0, BR theta-indep)
            {"Q1 = dt(dt(BR) + (-1 * (multr(BT))))",   "Q1 = 0"},
            {"Q2 = multr(dt(BT) + divr(BR)) - BR",     "Q2 = 0"},
            {"Q3 = dr(BT) + divr(divr(dt(BR) + (-1 * (multr(BT)))))", "Q3 = Lrt = 0"},
            {"Q4 = divsint(BT)",                       "Q4 = divsint of a zero field"},
            {"Q5 = divsint(dt(multsint(BT))) - dt(BT)", "Q5 = cot(BT) = 0"},
            {"Q6 = dt(BR)",                            "Q6 = 0"},
            {"Q7 = multr(BT)",                         "Q7 = 0"},
            // A1's inner terms, one at a time
            {"Z1 = (-1 * (-1 * (multr(dr(BR))))) - multr(dr(BR))", "Z1 = 0?"},
            {"Z2 = multr(dr(BR))",                     "Z2 = r dr(BR)"},
            {"Z3 = multr(dt(BT) + divr(BR))",          "Z3 = BR"},
            {"Z4 = dt(dt(BR) + (-1 * (multr(BT)))) + ((-1 * (-1 * (multr(dr(BR))))) + (-1 * (multr(dt(BT) + divr(BR)))))", "Z4 = A1 inner"},
            {"Z6 = (-1 * (-1 * (multr(dr(BR))))) + (-1 * (multr(dt(BT) + divr(BR))))", "Z6 = Z2 - Z3"},
            {"Z7 = dt(BR) + BR - BR",   "Z7 = dt(BR) = 0, added to a COS_EVEN field"},
            {"Z8 = dt(BR) + multr(dr(BR)) - multr(dr(BR))", "Z8 = 0, mixing dt into a sum"},
            {"Z9 = dt(PS) + PS - PS",   "Z9 = 0, same with a nonzero smooth field"},
            {"Y1 = (-1 * (multr(dt(BT) + divr(BR)))) + multr(dt(BT) + divr(BR))", "Y1: -1*(op(a+b))"},
            {"Y2 = (0 - (multr(dt(BT) + divr(BR)))) + multr(dt(BT) + divr(BR))", "Y2: 0-(op(a+b))"},
            {"Y3 = ((-1) * (multr(dt(BT) + divr(BR)))) + multr(dt(BT) + divr(BR))", "Y3: (-1)*(op(a+b))"},
            {"Y4 = (-1 * (multr(divr(BR)))) + multr(divr(BR))", "Y4: -1*(op(simple))"},
            {"Y5 = (-1 * (dt(BT) + divr(BR))) + (dt(BT) + divr(BR))", "Y5: -1*(a+b) no op"},
            {"Z10 = Z2 - Z3",  "Z10: the SAME sum via NAMED sub-defs"},
            // ⚠ log inside an expression: each operand right, difference wrong
            {"L1 = dr(log(PS)) - dr(log(PS))",   "L1 = 0?"},
            {"L2 = log(PS) - log(PS)",           "L2 = 0?"},
            {"L3 = dr(log(PH)) - dr(log(PH))",   "L3 = 0?"},
            {"L4 = dr(log(PH)) - 4 * dr(log(PS))", "L4 = the T8 combination inline"},
            {"L5 = log(PS) * PS - log(PS) * PS", "L5 = 0?"},
            {"Z11 = (-1 * (-1 * (multr(dr(BR))))) + (-1 * Z3)", "Z11: one named"},
            {"Z5 = divr(divr(dt(dt(BR) + (-1 * (multr(BT)))) + ((-1 * (-1 * (multr(dr(BR))))) + (-1 * (multr(dt(BT) + divr(BR)))))))", "Z5 = A1 second term"},
            // ---- which EXPRESSION SHAPES does the parser accept? ----
        };
        for (const auto& x : sm) {
            try {
                syst.add_def(x.def);
            } catch (const std::exception& ex) {
                std::cout << "#   " << x.what << " : add_def THREW " << ex.what()
                          << std::endl;
                continue;
            }
            const char* nm = std::string(x.def).substr(0, 2).c_str();
            char id[3] = {x.def[0], x.def[1], 0};
            double mx = 0.0;
            for (int d = 0; d <= dtop; d++) {
                const Val_domain& v = syst.give_val_def_scalar_domain(id, d);
                Index ix(space.get_domain(d)->get_nbr_points());
                do { mx = std::max(mx, std::fabs(v(ix))); } while (ix.inc());
            }
            double want = 0.0;
            if (std::string(id) == "S1" || std::string(id) == "S2") want = 1.0;
            if (std::string(id) == "S7") want = 2.0;
            emit(std::string("FJP_smoke_") + id, std::fabs(mx - want));
            std::cout << "#   " << x.what << "  ->  |value - expected| = "
                      << std::setprecision(3) << std::fabs(mx - want) << std::endl;
            (void)nm;
        }
    }
    // ---- the built cos(theta)*X against a value computed HERE ------------
    if (!clean) {
        double w9 = 0.0, w10 = 0.0;
        for (int d = 0; d <= dtop; d++) {
            const Val_domain& a = syst.give_val_def_scalar_domain("V9", d);
            const Val_domain& b = syst.give_val_def_scalar_domain("V10", d);
            const Kadath::Domain* dm = space.get_domain(d);
            Index ix(dm->get_nbr_points());
            do {
                const double th = dm->get_coloc(2)(ix(1));
                w9 = std::max(w9, std::fabs(a(ix) - std::cos(th)));
                w10 = std::max(w10,
                               std::fabs(b(ix) - std::cos(th) * std::cos(2 * th)));
            } while (ix.inc());
        }
        emit("FJP_multcost_ones", w9);
        emit("FJP_multcost_cos2th", w10);
        if (rank == 0)
            std::cout << "#   cos(th)*X built from dt/multsint, vs cos(th)*X "
                         "computed here:  X=1 " << w9 << "   X=cos2th " << w10
                      << "\n";
    }

    // ---- how long a def can the parser take?  Built from ONE repeated term,
    // so only the length varies.
    if (!clean) {
        std::string acc = "PS";
        for (int k = 1; k <= 120; k++) {   // ~145 is where it dies, and NOT reproducibly
            acc = "(" + acc + " + PS)";
            if (k < 100 || k % 5) continue;
            const std::string d = "L" + std::to_string(k) + " = " + acc;
            std::cout << "#   parser length probe: " << acc.size()
                      << " chars, depth " << k << std::endl << std::flush;
            syst.add_def(d.c_str());
        }
    }

    (void)monolithic;
    // ---- THE ACCEPTANCE TEST.  finiteJ_register() owns the read-state
    // contract -- it registers each def and reads it into configuration space,
    // which is what the evaluator needs and what the probe must not be trusted
    // to remember.  The emission is NOT registered by hand here.
    //
    // ⚠ finiteJ_detail::subdefs() is reached only by the bisection flags below,
    // which break the contract on purpose.
    const auto& SUB = Trumpet::finiteJ_detail::subdefs();
    const int nsub = (maxdefs < 0) ? int(SUB.size())
                                   : std::min<int>(maxdefs, int(SUB.size()));
    emit("FJP_axis_violations",
         static_cast<double>(Trumpet::finiteJ_axis_violations().size()));
    // ⚠ the list holds every mixed sum the ordering pass CONSIDERED, and says
    // of each whether it was swapped or left; counting the list as "reordered"
    // would claim more than it shows.
    {
        int nswap = 0;
        for (const char* o : Trumpet::finiteJ_ordered_sums()) {
            if (std::string(o).find("swapped") != std::string::npos)
                nswap++;
            if (rank == 0)
                std::cout << "#   sum ordering (seed-specific): " << o << "\n";
        }
        emit("FJP_ordered_sums_considered",
             static_cast<double>(Trumpet::finiteJ_ordered_sums().size()));
        emit("FJP_ordered_sums_swapped", static_cast<double>(nswap));
    }
    if (rank == 0)
        for (const char* v : Trumpet::finiteJ_axis_violations())
            std::cout << "#   axis violation carried by the emission: " << v
                      << "\n";
    try {
        if (maxdefs < 0) {
            Trumpet::finiteJ_register(syst, space, 0, dtop, !breakcontract);
        } else {
            // --maxdefs truncates the emission, so the header's registration
            // cannot be used; this path is a diagnostic and says so.
            for (int si = 0; si < nsub; si++) {
                syst.add_def((std::string(SUB[si].name) + " = " + SUB[si].def)
                                 .c_str());
                if (touch)
                    Trumpet::finiteJ_detail::read_def(syst, space, SUB[si].name,
                                                      0, dtop);
            }
        }
    } catch (const std::exception& ex) {
        if (rank == 0)
            std::cerr << "FATAL: registration threw: " << ex.what() << "\n";
        MPI_Finalize();
        return 3;
    }
    emit("FJP_subdefs", static_cast<double>(nsub));

    // ---- STEP 4, FIRST PIECE: S_def and its two self-checks ---------------
    // ⚠ EVERY PIECE IS ITS OWN NAMED DEF, so no sum has an unnamed first
    // operand (round 108's defect) and every operand is read in registration
    // order by add_def.  The expression is transcribed from
    // working_draft.tex l.1996-2003 term by term rather than simplified first.
    //
    // ⚠ ONE TERM IS OMITTED AND THE OMISSION IS MEASURED, NOT ASSUMED.
    // D ln(r sin th) . D X = (1/r) dr(X) + (cot th / r^2) dt(X), and the second
    // piece is dropped here because alpha_P is a function of r alone so
    // dt(ln alpha_P) = 0.  That is exactly the kind of claim this project has
    // been wrong about, so FJP_sdef_dtLA reports max |dt(LA)| over the grid and
    // the block REFUSES if it is not zero.  Emitting the cot form instead would
    // put a divsint on an operand of sin-order 0 into a clean emission, which
    // is what rounds 126 and 144 cost.
    if (sdef) {
        const char* SD[] = {
            "LA = log(ALP)",
            "LPH = log(PHP)",
            "DLA = dr(LA)",
            "DLP = dr(LPH)",
            "TLA = dt(LA)",
            "TLP = dt(LPH)",
            "SA = lap(LA)",
            "SB1 = DLP * DLA",
            "SB2 = TLP * TLA",
            "SB3 = divr(divr(SB2))",
            "SB4 = SB1 + SB3",
            "SB = 2 * SB4",
            "SC = divr(DLA)",
            "SG1 = DLA * DLA",
            "SG2 = TLA * TLA",
            "SG3 = divr(divr(SG2))",
            "SG = SG1 + SG3",
            "SS1 = SA + SB",
            "SS2 = SS1 + SC",
            "SS3 = DEL * SS2",
            "SG4 = DEL * SG",
            "SG5 = DEL * SG4",
            "SDEF = SS3 + SG5",
            "RSD = SDEF * RR",
            "RRSD = RSD * RR",
        };
        for (const char* d : SD)
            syst.add_def(d);
        // ⚠ FETCHING IS NOT READING (round 108): index every def once, in
        // registration order, before anything is read back.
        for (const char* d : SD) {
            std::string nm(d);
            nm = nm.substr(0, nm.find(' '));
            for (int dd = 0; dd <= dtop; dd++) {
                const Val_domain& v =
                    syst.give_val_def_scalar_domain(nm.c_str(), dd);
                Index ix(space.get_domain(dd)->get_nbr_points());
                do { (void) v(ix); } while (ix.inc());
            }
        }
        // ⚠ std::max SILENTLY SWALLOWS NaN, so a sup-norm built from it cannot
        // detect a non-finite field: std::max(m, NaN) returns m, because every
        // comparison with NaN is false.  The first version of this guard used
        // exactly that and reported max |S_def| = 0.295 on a layout whose
        // values were NaN at every point it had skipped.  The non-finite points
        // are COUNTED separately, which is the only way the check can fail.
        long nonfin = 0;
        auto sup = [&](const char* nm, int dd) {
            const Val_domain& v = syst.give_val_def_scalar_domain(nm, dd);
            double m = 0.0;
            Index ix(space.get_domain(dd)->get_nbr_points());
            do {
                const double x = v(ix);
                if (!std::isfinite(x)) { nonfin++; continue; }
                m = std::max(m, std::fabs(x));
            } while (ix.inc());
            return m;
        };
        double dtla = 0.0, dtlp = 0.0, sd = 0.0;
        for (int dd = 0; dd <= dtop; dd++) {
            dtla = std::max(dtla, sup("TLA", dd));
            dtlp = std::max(dtlp, sup("TLP", dd));
            sd = std::max(sd, sup("SDEF", dd));
        }
        emit("FJP_sdef_nonfinite", static_cast<double>(nonfin));
        emit("FJP_sdef_delta", sdefdelta);
        emit("FJP_sdef_dtLA", dtla);
        emit("FJP_sdef_dtLPH", dtlp);
        emit("FJP_sdef_sup", sd);
        if (rank == 0) {
            std::cout << "#\n#   S_def, delta = " << sdefdelta << "\n";
            std::cout << "#   max |dt(ln alpha_P)| = " << dtla
                      << "   max |dt(ln Phb_P)| = " << dtlp
                      << "   (the omitted cot term is proportional to the "
                         "first)\n";
            std::cout << "#   max |S_def| = " << sd << "\n";
        }
        // ⚠ S_def GOES AS 1/r^2 AT THE THROAT (draft l.2007-2009), so a layout
        // whose inner domain REACHES r = 0 cannot carry it: ln alpha_P = ln 0
        // there and every derived quantity is non-finite.  Measured on
        // backbone_stock_res21.dat, whose first radius is exactly 0.  Refused
        // rather than reported, because a table of nan reads as a failed build
        // and this is a statement about the layout.
        if (nonfin || !std::isfinite(sd)) {
            if (rank == 0)
                std::cerr << "FATAL: S_def is not finite at " << nonfin
                          << " point(s) on this layout.  It goes as 1/r^2 at "
                             "the throat, so an inner domain reaching r = 0 "
                             "cannot carry it.\n";
            MPI_Finalize();
            return 7;
        }
        if (dtla != 0.0) {
            if (rank == 0)
                std::cerr << "FATAL: dt(ln alpha_P) is not identically zero, so "
                             "the omitted cot(theta) term is not zero either "
                             "and S_def as built is incomplete\n";
            MPI_Finalize();
            return 6;
        }
        // SELF-CHECK 2: near the throat S_def -> 2 delta (delta + 2) / r^2,
        // draft l.2007-2009.  Reported as r^2 S_def against that constant, at
        // the innermost radii of domain 0, WITH its approach rather than as a
        // single number: ln alpha_P = sqrt2 ln r + O(1) is a leading-order
        // statement and the O(1) is what the approach measures.
        if (rank == 0) {
            const double want = 2.0 * sdefdelta * (sdefdelta + 2.0);
            std::cout << "#   near-throat check: r^2 S_def -> 2 d (d+2) = "
                      << want << "\n";
            std::cout << "#     i   r            r^2 S_def        ratio\n";
            const Val_domain& v =
                syst.give_val_def_scalar_domain("RRSD", 0);
            Index ix(space.get_domain(0)->get_nbr_points());
            const int npr = space.get_domain(0)->get_nbr_points()(0);
            for (int i = 0; i < std::min(6, npr); i++) {
                Index jx(space.get_domain(0)->get_nbr_points());
                jx.set(0) = i; jx.set(1) = 0;
                std::cout << "#     " << std::setw(3) << i << "   "
                          << std::setw(11) << t.pts[0][i].r << "   "
                          << std::setw(14) << v(jx) << "   "
                          << (want != 0.0 ? v(jx) / want : 0.0) << "\n";
            }
            emit("FJP_sdef_throat_target", want);
            Index jx(space.get_domain(0)->get_nbr_points());
            jx.set(0) = 0; jx.set(1) = 0;
            emit("FJP_sdef_r2S_innermost", v(jx));
        }
    }

    // ---- --dump-jacobian: the BULK operator, with NO boundary rows ---------
    // ⚠ WHAT THIS IS AND IS NOT.  These are the six equations registered as
    // equations on every domain and nothing else: no inner match to the
    // rho-series, no outer far-field rows, no compatibility at R = 2M.  So it
    // is NOT P2 and must never be quoted as one -- P2 is a left-null quantity
    // of the FULL BVP and research round 341 is explicit that sigma_min, a
    // right-null quantity, does not answer the blocker question either.
    //
    // What it establishes is one-sided and worth having: adding rows cannot
    // decrease sigma_min, so sigma_min(bulk + BC) >= sigma_min(bulk).  And the
    // counting m - n is reported beside it, because if the bulk is square its
    // left null space is empty and it contributes no obstruction at all, which
    // would put the whole compatibility question in the BC rows.
    if (!basisprobe.empty() && (!jacon || !clean)) {
        // ⚠ The plant destroys the physical fields, so every other output of
        // this run is meaningless.  Refuse rather than let a planted run be
        // mistaken for a real one -- there is no reading of a residual here
        // that would look wrong.
        if (rank == 0)
            std::cerr << "FATAL: --basis-probe requires --clean and "
                         "--dump-jacobian; the fields are planted and nothing "
                         "else this run produces is meaningful.\n";
        MPI_Finalize();
        return 13;
    }
    if (jacon && jacmatch && jacouterfull) {
        // ⚠ The incidence check locates the inner block as the LAST 12nt - 6
        // rows, which is true only because --jac-inner is registered last.
        // --jac-outer-full appends after it, so the two together would make the
        // check report on the wrong rows while still returning a clean number.
        if (rank == 0)
            std::cerr << "FATAL: --jac-match and --jac-outer-full together; the "
                         "inner block would no longer be the last 12nt - 6 rows "
                         "and the incidence check would silently move.\n";
        MPI_Finalize();
        return 12;
    }
    if (jacon && jacmatch && !jacinner) {
        // ⚠ REFUSE rather than register.  The matching unknowns enter the C0/C1
        // rows and nothing else, so without --jac-inner they would be columns
        // no row touches -- the kernel would come back 8nt - 6 and would be
        // reporting this flag combination rather than the system.
        if (rank == 0)
            std::cerr << "FATAL: --jac-match requires --jac-inner; without the "
                         "C0/C1 rows the matching unknowns are unconstrained "
                         "columns and the nullity would measure the flags.\n";
        MPI_Finalize();
        return 11;
    }
    if (jacon) {
        // ---- --jac-match: registration, BEFORE any equation is parsed ------
        // ⚠ ORDER IS LOAD-BEARING.  The parser resolves names at add_eq time,
        // so both the profiles and the unknowns have to exist first.  And the
        // column block they occupy is Kadath's own: do_col_J walks variable
        // domains, then the nvar_double scalars in registration order, then
        // the fields -- so the matching unknowns are columns [0, 8nt-6).
        int match_ncol_before = 0, match_ncol_after = 0;
        if (jacmatch) {
            for (std::size_t i = 0; i < mbase.size(); i++)
                syst.add_cst(mbnames[i].c_str(), *mbase[i]);
            match_ncol_before = syst.get_nbr_unknowns();
            char nm[16];
            for (int g = 0; g < NG; g++) {
                std::snprintf(nm, sizeof nm, "G%02d", g);
                syst.add_var(nm, matchv[g]);
            }
            if (!jacmatchnotower)
                for (int tt = 0; tt < NT; tt++) {
                    std::snprintf(nm, sizeof nm, "T%02d", tt);
                    syst.add_var(nm, matchv[NG + tt]);
                }
            // ⚠ THE GRADE-1 BLOCK, registered AFTER the tower so the column
            // layout of every build without --jac-recursion is untouched.
            if (jacrec)
                for (int g = 0; g < NG; g++) {
                    std::snprintf(nm, sizeof nm, "H%02d", g);
                    syst.add_var(nm, matchv[NG + NT + g]);
                }
            // ⚠ STATIC, because add_var stores a POINTER and this must outlive
            // `syst` exactly as matchv does.  One scalar: the balance's own
            // constant, i.e. n's shooting residual.
            if (jacbalance && jacrec) {
                static double sbal_storage = 0.0;
                sbal_storage = 0.0;
                syst.add_var("SBAL", sbal_storage);
                sbal_ptr = &sbal_storage;
            }
            // ⚠ STATIC, as sbal_storage is.  u2's nt COS_EVEN coefficients,
            // registered right after SBAL, so they sit at columns
            // [2ng + 1, 2ng + 1 + nt), ahead of every field column.
            if (g2u2 && jacrec) {
                static std::vector<double> u2_storage;
                u2_storage.assign(ntheta, 0.0);
                // the J = 0 closed form u2 = -5 W^2/24 with W = G_PH/G_PS,
                // on the (physical-h) seed's grade-0 amplitudes
                const double Wg = matchv[NCE] / matchv[0];
                u2_storage[0] = -5.0 * Wg * Wg / 24.0;
                char un[16];
                for (int q = 0; q < ntheta; q++) {
                    std::snprintf(un, sizeof un, "UU%02d", q);
                    syst.add_var(un, u2_storage[q]);
                }
                emit("FJPJ_grade2u2_cols", ntheta);
                emit("FJPJ_grade2u2_seed", u2_storage[0]);
                if (rank == 0)
                    std::cout << std::setprecision(17)
                              << "#  --grade2-u2: " << ntheta << " unknowns UU00..UU"
                              << ntheta - 1 << " registered after SBAL; seed u2 mode 0"
                                 " = -5 W^2/24 = " << u2_storage[0]
                              << " (W = G_PH/G_PS = " << Wg << ")\n";
            }
            // ⚠ STATIC, as sbal_storage is: add_var keeps a pointer and these
            // must outlive `syst`.
            if (droptephi && jacrec) {
                static std::vector<double> wp_storage;
                wp_storage.assign(ntheta, 0.0);
                char wn[16];
                for (int q = 1; q < ntheta; q++) {
                    std::snprintf(wn, sizeof wn, "WP%02d", q);
                    syst.add_var(wn, wp_storage[q]);
                }
                if (rank == 0)
                    std::cout << "#  --drop-tephi: " << ntheta - 1
                              << " free coefficients WP01..WP"
                              << ntheta - 1 << " for TEPHI's ell >= 2\n";
            }
            match_ncol_after = syst.get_nbr_unknowns();
            emit("FJPJ_match_unknowns", match_ncol_after - match_ncol_before);
            emit("FJPJ_match_grade0", NG);
            emit("FJPJ_match_grade1", jacrec ? NG : 0);
            emit("FJPJ_match_tower", jacmatchnotower ? 0 : NT);
            emit("FJPJ_match_notower", jacmatchnotower ? 1.0 : 0.0);
        }
        auto wanted_eq = [&](const char* nm) {
            if (jaceqs == "all") return true;
            return (',' + jaceqs + ',').find(std::string(",") + nm + ",")
                   != std::string::npos;
        };
        for (int d = 0; d <= dtop; d++)
            for (const char* ename : Trumpet::finiteJ_eq_names())
                if (wanted_eq(ename)) {
                    const bool proj = eshtnt2
                                      && std::string(ename) == "ESHT";
                    std::string lhs = proj
                        ? "multsint(" + std::string(ename) + ")"
                        : std::string(ename);
#ifdef TRUMPET_BULK2
                    if (eqfnsin2 && std::string(ename) == "EQFN") {
                        lhs = "multsint(multsint(" + lhs + "))";
                        if (d == 0 && rank == 0)
                            std::cout << "#  --eqfn-sin2: EQFN registered as " << lhs << " in every domain\n";
                    }
                    if (d == 0 && amaxhas.count(ename)) {
                        lhs = "W" + std::string(ename) + " * " + lhs;
                        if (rank == 0)
                            std::cout << "#  --amax-table: domain 0 " << ename << " registered as " << lhs << "\n";
                    }
#endif
                    // r^k, applied to the EXPRESSION and so before the tau
                    // projection.  k is read from --eq-prefactor NAME:k.
                    int kp = 0;
                    if (!eqpref.empty()) {
                        const std::string want = std::string(ename) + ":";
                        std::size_t at = (',' + eqpref).find(',' + want);
                        if (at != std::string::npos)
                            kp = std::atoi(eqpref.c_str()
                                           + (at + want.size()));
                    }
                    for (int t2 = 0; t2 < kp; t2++)
                        lhs = "multr(" + lhs + ")";
                    if (kp && d == 0 && rank == 0)
                        std::cout << "#  --eq-prefactor " << ename
                                  << " -> r^" << kp << "\n";
#ifdef TRUMPET_CHI_UNKNOWN
                    // ⚠⚠ ECHI IS ZEROTH ORDER AND MUST NOT BE TAU-PROJECTED.
                    // add_eq_inside is documented "assumed to be SECOND order"
                    // (system_of_eqs.hpp:845), and that is exactly why round 262
                    // came out 6*nt rows short: it discarded ECHI's top two
                    // radial coefficients per (domain, angular mode) as if they
                    // were a boundary freedom, when ECHI carries no derivative
                    // of CH at all and every collocation point is a condition.
                    //
                    // Research round 556 ruled "impose every radial coefficient
                    // via add_eq_mode".  add_eq_full IS that, and is the
                    // library's own primitive for it -- documented
                    // "an equation to be solved inside a domain (assumed to be
                    // zeroth order i.e. with no derivatives)"
                    // (system_of_eqs.hpp:1009) -- so it takes one call per
                    // domain instead of nr*nt add_eq_mode calls and cannot get
                    // the coefficient bookkeeping wrong.  The COUNT is the
                    // check, not this comment.
                    if (std::string(ename) == "ECHI") {
                        syst.add_eq_full(d, (lhs + " = 0").c_str());
                        if (d == 0 && rank == 0)
                            std::cout << "#  --chi-unknown: ECHI registered by"
                                         " add_eq_full (zeroth order, NO tau"
                                         " projection) in every domain\n";
                        continue;
                    }
#endif
                    syst.add_eq_inside(d, (lhs + " = 0").c_str());
                }
        if (jacinterfaces) {
            // value and radial derivative of every field, at the outer face of
            // every domain but the last: three interfaces on this layout.
            // ⚠ MUST RESPECT --jac-fields.  Round 135's first per-field check
            // registered all six interfaces whatever was asked for, so every
            // field returned the same 162/306 and the "isolation" isolated
            // nothing.  A test that cannot vary its input reads as agreement.
            const char* fn[6] = {"PS", "PH", "QF", "BR", "BT", "QB"};
            int nif = 0;
            for (int d = 0; d < dtop; d++)
                for (int q = 0; q < 6; q++) {
                    if (!(jacfields == "all"
                          || (',' + jacfields + ',').find(std::string(",") + fn[q] + ",")
                             != std::string::npos))
                        continue;
                    syst.add_eq_matching(d, OUTER_BC, fn[q]);
                    syst.add_eq_matching(d, OUTER_BC,
                                         (std::string("dr(") + fn[q] + ")").c_str());
                    nif += 2;
                }
            emit("FJPJ_interface_conditions", nif);
        }
        if (jacouter) {
            const std::string want = ',' + jacouterrows + ',';
            int nout = 0;
            // ⚠ NAMED t_Q = 0, BUT IT IS NOT THAT.  Research round 379: q_P = 0
            // is a theorem of the conformal gauge, so t_Q needs no imposing;
            // and the asymptotic condition for a 2-D Laplacian must fix the
            // ln r coefficient or the additive constant.  What this row
            // actually imposes is q = 0 AT r = r_out, a Dirichlet condition at
            // a finite radius where q is small and not zero.  It is kept, and
            // separately selectable, so that what it constrains can be
            // measured rather than argued.
            if (want.find(",q,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "multr(QF) = 0");
                nout++;
            }
            // 2 t_U + t_G = 0: Phi-bar = alpha psi^2, so the 1/r coefficient of
            // ln Phi-bar is the sum of the lapse and psi^2 coefficients.  The
            // backbone subtraction is EXPLICIT.  ⚠ Whether this one is also a
            // finite-radius Dirichlet match rather than the global statement it
            // is named for is the same question and is measured the same way.
            if (want.find(",phb,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "multr(PH) = multr(PHP)");
                nout++;
            }
#ifdef TRUMPET_BULK2
            // ⚠ ROUND 335 (research round 664): PH's mode 0 is the VALUE Phibar(inf) = 1; modes >= 1 keep the derivative
            // row multr(PH) = multr(PHP).  Mode by mode (add_eq_mode), in place of `phb`, which must not be listed too.
            if (want.find(",phm,") != std::string::npos) {
                if (want.find(",phb,") != std::string::npos) {
                    std::cerr << "FATAL: phm and phb are alternatives\n";
                    return 1;
                }
                for (int j = 0; j < ntheta; j++) {
                    Index pos_cf(space.get_domain(dtop)->get_nbr_coefs());
                    pos_cf.set(1) = j;
                    if (j == 0) syst.add_eq_mode(dtop, OUTER_BC, "PH", pos_cf, 1.0);
                    else        syst.add_eq_mode(dtop, OUTER_BC, "multr(PH) - multr(PHP)", pos_cf, 0.0);
                }
                if (rank == 0)
                    std::cout << "#  outer phm: PH mode 0 = 1 (value), modes 1.." << ntheta - 1
                              << " multr(PH) = multr(PHP), " << ntheta << " rows by add_eq_mode\n";
                nout++;
            }
#endif
            // ⚠ CONTROLS, and they are the point of the measurement rather
            // than an extra.  "Most of the q row lies inside the bulk row
            // space" means nothing until it is known what a boundary row looks
            // like here: every row supported only on the outer face will have a
            // large component in a row space of codimension 12nt - 6.  These
            // are alternative outer rows of the same shape -- a plain
            // Dirichlet, a Neumann, and one on a different field -- so the q
            // row's fraction can be read against them instead of against 1.
            if (want.find(",qd,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "QF = 0");
                nout++;
            }
            if (want.find(",qn,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "dr(QF) = 0");
                nout++;
            }
            if (want.find(",ps,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "multr(PS) = 0");
                nout++;
            }
            // ⚠ THE FOUR OWED ROWS, IN THEIR SPECIFIED FORM (round 171).
            // NOTES_outer_rows.md specifies Dirichlet AT THE r = infinity NODE
            // -- psi^2(inf) = 1, beta~^r(inf) = 0, Bhat^theta(inf) = 0,
            // qbar(inf) = 0 -- because on the compact domain the value at that
            // node IS the additive constant each m = 0 mode leaves free.
            // ⚠ These are NOT the `ps` and `br` entries below: multr(PS) picks
            // the 1/r COEFFICIENT, which is a different functional.  Added so a
            // complete assembly can be built rather than approximated with the
            // rows that happened to exist.  The RHS is irrelevant to every use
            // these have; only the row space is read.
            if (want.find(",psv,") != std::string::npos) {
                // ⚠ ROUND 195: psi^2(inf) = 1, not 0.  NOTES_outer_rows.md
                // specifies it and the seed carries it (the table gives Rr = 1
                // at the r = infinity node); the 0 was the placeholder the
                // comment above declares, and round 194 measured the seed
                // failing it by exactly 1.  The ROW is unchanged, so every
                // row-space result from rounds 172-193 is untouched.
                // brv, btv and qbv are already 0, which is their specified value.
                syst.add_eq_bc(dtop, OUTER_BC, "PS = 1");
                nout++;
            }
            // ⚠ THE MISSING NODE VALUE (round 206, research round 503 ruling 3).
            // NOTES_outer_rows.md specifies Dirichlet AT THE r = infinity NODE
            // for psi^2, beta~^r, beta^^theta and qbar -- psv, brv, btv, qbv --
            // and there is no entry for Phibar.  That is why releasing
            // multr(PH) cost exactly nt: PH loses its ONLY outer row.
            // Phibar = alpha psi^2 with alpha -> 1 and psi^2 -> 1, so the node
            // value is 1, exactly parallel to psv.  With phv assembled,
            // multr(PH) is free to be the theorem-level check the draft
            // describes instead of a boundary condition.
            if (want.find(",phv,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "PH = 1");
                nout++;
            }
            if (want.find(",brv,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "BR = 0");
                nout++;
            }
            if (want.find(",btv,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "BT = 0");
                nout++;
            }
            if (want.find(",qbv,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "QB = 0");
                nout++;
            }
            if (want.find(",br,") != std::string::npos) {
                syst.add_eq_bc(dtop, OUTER_BC, "multr(BR) = 0");
                nout++;
            }
            if (jacmixed) {
                syst.add_eq_bc(dtop, OUTER_BC,
                               "multr(dr(QF)) + LAM * QF = 0");
                nout++;
                emit("FJPJ_outer_mixed_lambda", jacmixlam);
            }
            emit("FJPJ_outer_conditions", nout);
        }
#ifdef TRUMPET_BULK2
        // ⚠ ROUND 335: the KOMAR row on the domain-1/2 interface (research round 664; radius-independent, round 333).
        // M = (1/4 pi) int (D^i alpha - K^i_j beta^j) dS_i;  Kadath's integ(f) at a boundary is 2 pi r^2 int_0^pi f sin th
        // dth, so with KG = psi^2 d_r alpha - psi^6 e^{2q} (K^r_r beta^r + K^r_th beta^th) the row is integ(KG) = 4 pi M.
        // K_ij = (D_i beta_j + D_j beta_i) / (2 alpha):  K^r_r = (beta^r d_r ln g + beta^th d_th ln g + 2 d_r beta^r) /
        // (2 alpha), K^r_th = (d_th beta^r + r^2 d_r beta^th) / (2 alpha), g = psi^4 e^{2q}, psi^2 = PS, alpha = PH / PS,
        // q = sin^2 th QF.  K^r_phi beta^phi (J^2) is NOT carried (step 2, J = 0; research round 664).
        // integ() is a SCALAR term; `* ones` makes it a field whose mode-0 coefficient is the integral.
        if (komar) {
            const char* kd[] = {
                "KMQ = multsint(multsint(QF))",
                "KMAL = PH / PS",
                "KMALR = dr(KMAL)",
                "KMLGR = 2 * dr(PS) / PS + 2 * dr(KMQ)",
                "KMLGT = 2 * dt(PS) / PS + 2 * dt(KMQ)",
                "KMKRR = (BR * KMLGR + BT * KMLGT + 2 * dr(BR)) / (2 * KMAL)",
                "KMKRT = (dt(BR) + multr(multr(dr(BT)))) / (2 * KMAL)",
                "KMG = PS * KMALR - PS * PS * PS * exp(2 * KMQ) * (KMKRR * BR + KMKRT * BT)"};
            // ⚠ the read-state contract (round 108): each def is READ right after it is registered, as finiteJ_register does
            for (const char* x : kd) {
                syst.add_def(1, x);
                const std::string nm(x, std::string(x).find(' '));
                const Kadath::Val_domain& kv = syst.give_val_def_scalar_domain(nm.c_str(), 1);
                Kadath::Index kix(space.get_domain(1)->get_nbr_points());
                (void)kv(kix);
            }
            Index pos_cf(space.get_domain(1)->get_nbr_coefs());
            syst.add_eq_mode(1, OUTER_BC, "integ(KMG) * ones", pos_cf, 4.0 * M_PI * komar_m);
            if (rank == 0)
                std::cout << "#  --komar: integ(KMG) = 4 pi x " << komar_m << " on the domain-1/2 interface (1 row)\n";
            emit("FJPJ_komar_row", 1);
        }
#endif
        int inner_row_begin = -1, inner_row_end = -1;
        if (jacinner) {
            // C0 and C1 at the inner face of domain 0, for all six fields.
            // add_eq_bc at INNER_BC projects onto each field's own tau basis,
            // so the count is the sum of the six angular-mode counts twice --
            // the 12nt - 6 the specification predicts -- rather than 12*ntheta.
            const char* fn[6] = {"PS", "PH", "QF", "BR", "BT", "QB"};
            // which angular family each field's tau projection lives in, and
            // how many modes it has.  These are the DECLARED bases (round 313):
            // PS/PH/QF/BR COS_EVEN nt, BT SIN_EVEN nt-2, QB COS_ODD nt-1.
            const char* fpre[6] = {"MPS", "MPH", "MQF", "MBR", "MBT", "MQB"};
            const char* npre[6] = {"NPS", "NPH", "NQF", "NBR", "NBT", "NQB"};
            const int   fn_n[6] = {NCE, NCE, NCE, NCE, NSE, NCO};
            // ⚠ the SIN_EVEN offset now lives inside the profile build (k = j+1
            // there), so every family is indexed by j from 0 here.
            const int   fk0 [6] = {0, 0, 0, 0, 0, 0};
            // ⚠ THE TOWER ENTERS ONLY BT's AND QB's C1 ROWS.  4nt C1 rows
            // (PS, PH, QF, BR) carry no unknown at all, and that is exactly the
            // inner block's net contribution 12nt - 6 - (8nt - 6) = 4nt, which
            // is round 164's number reached from the registration rather than
            // from the arithmetic.  If the placement were anywhere else the net
            // would still be 4nt but the count would not be 2nt - 3.
            const bool ftower[6] = {false, false, false, false, true, true};
            // ⚠ get_nbr_conditions() is -1 until the system is assembled, so
            // the span is COMPUTED: the inner block is 12nt - 6 rows and, with
            // --jac-outer-full refused above, it is the last block registered.
            const bool in_half = (jacinnermode == "half");
            const bool in_net  = (jacinnermode == "net");
            // ⚠ ROUND 187: the TWO-ENDED isometry block.  Under the composite
            // r -> M^2/4r, theta -> pi-theta, phi -> -phi (the human, round
            // 463: phi flips J and theta flips J, so together J -> J, which is
            // why the radial inversion alone is not an isometry of a spinning
            // end), each field is even or odd at the minimal surface and gets
            // ONE condition per angular mode either way:
            //     even  ->  dr(F) = 0        odd  ->  F = 0
            // so the block is 6nt - 3 for all 2^6 assignments and the system is
            // SQUARE before any parity is chosen.  ⚠ That is why the observable
            // is the NULLITY and not the count: round 185's `half` was square
            // with nullity 0, and square is not determined.
            const bool in_iso = (jacinnermode == "iso");
            if (in_iso && static_cast<int>(isoparity.size()) != 6) {
                if (rank == 0)
                    std::cerr << "FATAL: --jac-inner-mode iso needs "
                                 "--iso-parity with six characters E/O, got \""
                              << isoparity << "\"\n";
                MPI_Finalize();
                return 15;
            }
            if (in_iso && jacmatch) {
                if (rank == 0)
                    std::cerr << "FATAL: --jac-inner-mode iso with --jac-match;"
                                 " the isometry block replaces the matching"
                                 " unknowns, it does not carry them.\n";
                MPI_Finalize();
                return 16;
            }
            auto ampwant = [](const std::string& list, const char* nm) {
                if (list.empty()) return false;
                return (',' + list + ',').find(std::string(",") + nm + ",")
                       != std::string::npos;
            };
            // ⚠ REFUSE ON g0 != 0.  For those fields C1 is a genuine Robin
            // condition (round 195), so replacing it would REMOVE a real
            // condition and the near-kernel count would be reporting that
            // rather than the diagnosis.  QF and QB are the only two with
            // g0 = 0 and they are the only two the ruling names.
            for (int q = 0; q < 6; q++)
                if ((ampwant(ampdir, fn[q]) || ampwant(amppin, fn[q]))
                    && q != 2 && q != 5) {
                    if (rank == 0)
                        std::cerr << "FATAL: --inner-amp-* names " << fn[q]
                                  << ", whose g0 is not 0; its C1 row is a "
                                     "genuine Robin condition and replacing it "
                                     "would remove a real condition.\n";
                    MPI_Finalize();
                    return 18;
                }
            if ((!ampdir.empty() || !amppin.empty()) && !jacmatch) {
                if (rank == 0)
                    std::cerr << "FATAL: --inner-amp-* without --jac-match; "
                                 "there are no amplitudes to impose on.\n";
                MPI_Finalize();
                return 18;
            }
            int nin = 0, g = 0, tw = 0, namp = 0;
            for (int q = 0; q < 6; q++) {
                std::string c0 = std::string(fn[q]);
                std::string c1 = std::string("dr(") + fn[q] + ")";
                std::string camp;
                if (jacmatch) {
                    // ⚠ 40, not 16: " - G00 * dr(MPS00)" is 18 characters and
                    // the 16-byte buffer TRUNCATED it silently into a string the
                    // parser rejected with "= needed for equations".  The
                    // compiler warned; the build output was tailed two lines and
                    // the warning was missed.
                    char b[40];
                    // ⚠ ROUND 195: THE SAME G ENTERS C1, DIFFERENTIATED.  Round
                    // 194 found C1 collapsing to dr(F) = 0; restoring the
                    // profile's radial factor alone did not change it, because
                    // the grade-0 unknowns were written into C0 ONLY -- C1 took
                    // tower terms and nothing else, so with the tower removed it
                    // carried no unknown at all.  The matching condition is
                    // value AND slope against the SAME amplitude, so G must
                    // appear in both rows; C0 and C1 then eliminate it to round
                    // 436's Robin condition X'S - XS' = 0 with S = r^{g0}M(th).
                    const int gbase = g;
                    for (int j = 0; j < fn_n[q]; j++) {
                        // the break control: BT's grade-0 amplitude gets a
                        // COS_EVEN profile, which its SIN_EVEN row cannot
                        // represent, so the column must stop being single-row.
                        const char* pre = (jacmatchbreak && q == 4) ? "MCE"
                                                                   : fpre[q];
                        const int k = (jacmatchbreak && q == 4) ? j : j + fk0[q];
                        std::snprintf(b, sizeof b, " - G%02d * %s%02d", g++, pre, k);
                        c0 += b;
                    }
                    for (int j = 0; j < fn_n[q]; j++) {
                        const char* pre = (jacmatchbreak && q == 4) ? "MCE"
                                                                    : fpre[q];
                        std::snprintf(b, sizeof b, " - G%02d * dr(%s%02d)",
                                      gbase + j, pre, j + fk0[q]);
                        c1 += b;
                    }
                    // ⚠ C0 AND C1 DETERMINE G AND G_1 TOGETHER (research round
                    // 511 ruling 2').  The grade-1 amplitude enters both rows
                    // against the NEXT profile in the rho = r^n series, so the
                    // 12nt - 6 matching rows carry 12nt - 6 unknowns and their
                    // net contribution to the inner count is ZERO -- which is
                    // what leaves the 6nt - 3 recursion rows as the inner
                    // condition, and what keeps the system square.
                    if (jacrec) {
                        for (int j = 0; j < fn_n[q]; j++) {
                            std::snprintf(b, sizeof b, " - H%02d * %s%02d",
                                          gbase + j, npre[q], j + fk0[q]);
                            c0 += b;
                        }
                        for (int j = 0; j < fn_n[q]; j++) {
                            std::snprintf(b, sizeof b, " - H%02d * dr(%s%02d)",
                                          gbase + j, npre[q], j + fk0[q]);
                            c1 += b;
                        }
                    }
                    // the amplitude row  sum_j G_j M_j = 0, projected onto the
                    // SAME tau basis the C0/C1 rows use, so it is one condition
                    // per angular mode and the count does not move.
                    for (int j = 0; j < fn_n[q]; j++) {
                        std::snprintf(b, sizeof b, "%sG%02d * %s%02d",
                                      j ? " + " : "", gbase + j, fpre[q],
                                      j + fk0[q]);
                        camp += b;
                    }
                    if (ftower[q] && !jacmatchnotower)
                        for (int j = 0; j < fn_n[q]; j++) {
                            std::snprintf(b, sizeof b, " - T%02d * %s%02d",
                                          tw++, fpre[q], j + fk0[q]);
                            c1 += b;
                        }
                }
                if (in_iso) {
                    const char pc = isoparity[q];
                    if (pc != 'E' && pc != 'O') {
                        if (rank == 0)
                            std::cerr << "FATAL: --iso-parity character " << q
                                      << " is '" << pc << "', not E or O\n";
                        MPI_Finalize();
                        return 15;
                    }
                    // even -> dr(F) = 0 ; odd -> F = 0.  Same count either way.
                    syst.add_eq_bc(0, INNER_BC,
                                   ((pc == 'E' ? c1 : c0) + " = 0").c_str());
                    nin++;
                } else if (in_net) {
                    // the 4nt rows that carry no matching unknown
                    if (q < 4) { syst.add_eq_bc(0, INNER_BC,
                                                (c1 + " = 0").c_str()); nin++; }
                } else if (in_half) {
                    syst.add_eq_bc(0, INNER_BC, (c0 + " = 0").c_str()); nin++;
                } else {
                    const bool aD = ampwant(ampdir, fn[q]);
                    const bool aP = ampwant(amppin, fn[q]);
                    if (aD && aP) {
                        if (rank == 0)
                            std::cerr << "FATAL: " << fn[q] << " is in BOTH "
                                         "--inner-amp-dir and --inner-amp-pin; "
                                         "that would replace both rows and the "
                                         "field would carry no inner condition."
                                      << "\n";
                        MPI_Finalize();
                        return 18;
                    }
                    syst.add_eq_bc(0, INNER_BC, ((aP ? camp : c0) + " = 0").c_str());
                    syst.add_eq_bc(0, INNER_BC, ((aD ? camp : c1) + " = 0").c_str());
                    nin += 2;
                    if (aD || aP) namp += fn_n[q];
                }
            }
            // ⚠ REGISTERED HERE, NOT WITH THE MATCHING UNKNOWNS.  Ope_def
            // EVALUATES its expression in the constructor, and the first SUM
            // in the emission segfaults when the defs are added before the
            // bulk equations while the identical string through --extra-def,
            // added after them, is fine.  So the defs go in immediately
            // before the rows that need them, which is also where they
            // belong: nothing earlier reads them.
            if (balancel2 && (g1replace || droptephi || jacrecearly || !jacrec)) {
                if (rank == 0)
                    std::cerr << "FATAL: --balance-l2 needs --jac-recursion at"
                                 " the LATE site, and refuses --grade1-replace"
                                 " and --drop-tephi (all three rewrite TEPHI)\n";
                MPI_Finalize();
                return 23;
            }
            if (maximality && (!balancel2 || g1replace || jacaxisres)) {
                if (rank == 0)
                    std::cerr << "FATAL: --maximality needs --balance-l2 and"
                                 " refuses --grade1-replace and --jac-axisres\n";
                MPI_Finalize();
                return 24;
            }
#ifndef THROATTH_HAS_MAX
            if (maximality) {
                if (rank == 0)
                    std::cerr << "FATAL: --maximality: this binary's throat"
                                 " header carries no TMAX (q-regular only)\n";
                MPI_Finalize();
                return 25;
            }
#endif
            if (g1replace && (jacrecearly || droptephi || !jacrec)) {
                if (rank == 0)
                    std::cerr << "FATAL: --grade1-replace needs --jac-recursion"
                                 " at the LATE site and without --drop-tephi"
                                 " (both rewrite TEPHI's modes)\n";
                MPI_Finalize();
                return 22;
            }
            if (jacrec && jacrecearly) {
                try {
                    // ⚠ DOMAIN 0 ONLY.  These rows live at the inner face
                    // and nowhere else, and the pure-angular profiles are
                    // filled with ZERO at the compact domain's r = infinity
                    // node -- so SPS vanishes there, log(SPS) is -inf, and
                    // reading the chain across all three domains evaluates an
                    // amplitude reconstruction that has no meaning outside
                    // domain 0.
                    Trumpet::throatth_register(syst, space, ntheta,
                                               0, dtop,
                                               std::getenv("THROATTH_NOREAD")
                                                   == nullptr);
                } catch (const std::exception& ex) {
                    if (rank == 0)
                        std::cerr << "FATAL: throatth_register threw: "
                                  << ex.what() << "\n";
                    MPI_Finalize();
                    return 20;
                }
                if (rank == 0)
                    std::cout << "#  --jac-recursion: " << NG
                              << " grade-1 unknowns, "
                              << Trumpet::throatth_rows().size()
                              << " theta-ODE rows registered\n";
            }
            // ⚠ THE RECURSION ROWS -- the inner boundary condition itself
            // (research round 504).  Each is projected onto its own parity's
            // tau basis, so the six contribute nt-1 + nt + nt-2 + nt + nt + nt
            // = 6nt - 3, exactly the grade-1 block they close.  The count is
            // asserted here rather than trusted, because it is the whole
            // reason the system stays square.
            // ⚠ THROATTH_MAX truncates the def table for bisection, and then
            // the rows do not exist -- asking for them aborts on an unknown
            // name, which is what every bisection run was actually dying of
            // before this gate went in.
            if (jacrec && jacrecearly
                && std::getenv("THROATTH_MAX") == nullptr) {
                // ⚠ MEASURED, NOT DECLARED.  The first version summed the
                // emitter's own mode table and asserted that -- which checks
                // the table against itself and passed while the assembly came
                // out one row over.  get_nbr_conditions() before and after is
                // what the system actually did with each row.
                int nrec = 0, k = 0;
                for (const auto& r : Trumpet::throatth_rows()) {
                    const int before = syst.get_nbr_conditions();
                    syst.add_eq_bc(0, INNER_BC,
                                   (std::string(r.first) + " = 0").c_str());
                    const int got = syst.get_nbr_conditions() - before;
                    const int want = Trumpet::throatth_row_modes(k, ntheta);
                    if (rank == 0)
                        std::cout << "#  --jac-recursion: " << r.first
                                  << "  declared " << want << "  assembled "
                                  << got << (got == want ? "" : "   <-- ***")
                                  << "\n";
                    nrec += got;
                    k++;
                }
                if (nrec != NG) {
                    if (rank == 0)
                        std::cerr << "FATAL: the recursion rows project onto "
                                  << nrec << " conditions, not " << NG
                                  << " = 6nt - 3; the system cannot be square."
                                  << "\n";
                    MPI_Finalize();
                    return 21;
                }
                emit("FJPJ_recursion_rows", nrec);
                if (rank == 0)
                    std::cout << "#  --jac-recursion: " << nrec
                              << " conditions from 6 rows (6nt - 3 = " << NG
                              << ")\n";
            }
            emit("FJPJ_inner_conditions", nin);
            emit("FJPJ_amp_rows", namp);
            if (rank == 0 && namp)
                std::cout << "#  --inner-amp: " << namp << " amplitude rows"
                          << (ampdir.empty() ? "" : ("  dir=" + ampdir))
                          << (amppin.empty() ? "" : ("  pin=" + amppin))
                          << "\n";
            if (in_iso && rank == 0)
                std::cout << "# iso parity " << isoparity
                          << "  (E -> dr(F)=0, O -> F=0)\n";
            emit("FJPJ_inner_mode_half", in_half ? 1.0 : 0.0);
            emit("FJPJ_inner_mode_net", in_net ? 1.0 : 0.0);
            if (jacmatch) {
                emit("FJPJ_match_grade0_used", g);
                emit("FJPJ_match_tower_used", tw);
            }
        }
        if (jacouterfull) {
            const char* fn[6] = {"PS", "PH", "QF", "BR", "BT", "QB"};
            const int ofd = (jacouterfulldom < 0) ? dtop : jacouterfulldom;
            if (ofd > dtop) {
                if (rank == 0)
                    std::cerr << "FATAL: --jac-outer-full-dom " << ofd
                              << " is beyond dtop = " << dtop << "\n";
                MPI_Finalize();
                return 14;
            }
            int nof = 0;
            for (int q = 0; q < 6; q++) {
                syst.add_eq_bc(ofd, OUTER_BC,
                               (std::string(fn[q]) + " = 0").c_str());
                syst.add_eq_bc(ofd, OUTER_BC,
                               (std::string("dr(") + fn[q] + ") = 0").c_str());
                nof += 2;
            }
            emit("FJPJ_outer_full_conditions", nof);
            emit("FJPJ_outer_full_dom", ofd);
            // the radius the block now lives at: the outer bound of that
            // domain, which for the compact domain is infinity.
            emit("FJPJ_outer_full_radius",
                 (ofd == ndom - 1) ? std::numeric_limits<double>::infinity()
                                   : bounds[ofd + 1]);
        }
        // ⚠ LATE, AND THAT IS A SEQUENCING RULE, NOT A PREFERENCE.  Ope_def
        // EVALUATES its expression in its constructor, and the first SUM in
        // the emission segfaults at every earlier point tried -- with the
        // matching unknowns, and again immediately before the rows that need
        // it -- while the identical string through --extra-def, after the
        // whole equation set, is fine.  Registered here, the last point
        // before sec_member() is called.  --jac-recursion-early keeps the
        // failing placement reachable as the standing control (round 513
        // ruling 2): if THROATTH_MAX=34 there ever stops segfaulting,
        // something changed and this rule is no longer describing the
        // library it was written against.
        if (jacrec && !jacrecearly) {
            // ⚠ THE MINIMAL REPRODUCTION, env-gated.  Round 213 said "the
            // first + in the emission" and that was WRONG: the amplitude defs
            // are themselves sums over the G block and they register.  This
            // narrows it -- each line is tried in order and the last one
            // printed is the shape that fails.
            if (std::getenv("THROATTH_MINREPRO")) {
                const char* probe[] = {
                    "ZT1 = (PS + PH)",            // two FIELDS
                    "ZT2 = (ACE00 + ACE01)",      // two CSTS
                    "ZT3 = (PS * ACE00)",         // field x cst
                    "ZT4 = (ZT3 + ZT3)",          // sum of two named defs
                    "ZT5 = (G00 * ACE00)",        // UNKNOWN x cst
                    "ZT6 = (ZT5 + ZT5)",          // sum of two unknown-defs
                    "ZT7 = (ZT5 + ZT3)",          // one of each
                    "ZTA = (G00 * ACE00 + G01 * ACE01)",       // 2-term sum
                    "ZTB = (ZTA + ZTA)",                       // sum of those
                    "ZTC = (G00 * ACE00 + G01 * ACE01 + G02 * ACE02)",
                    "ZTD = (ZTC + ZTC)",                       // 3-term
                    "ZTE = (G00 * ACE00 + G01 * ACE01 + G02 * ACE02 + G03 * ACE03)",
                    "ZTF = (ZTE + ZTE)",                       // 4-term
                    "ZTG = (G00 * ACE00 + G01 * ACE01 + G02 * ACE02 + G03 * ACE03 + G04 * ACE04)",
                    "ZTH = (ZTG + ZTG)",                       // 5-term
                    // ⚠ S-PREFIXED NAME, registered the ordinary way: if this
                    // sum dies the trigger is the NAME (is_tensor reading
                    // "SPS" as S with indices P,S), and if it lives the
                    // trigger is throatth_register's add_def(dd, ...) loop.
                    "SQQ = (G00 * ACE00 + G01 * ACE01 + G02 * ACE02 + G03 * ACE03 + G04 * ACE04)",
                    "ZTI = (SQQ + ZTG)",
                    "ZT8 = (SPS + SPH)",          // the emission's shape
                };
                for (const char* q : probe) {
                    std::printf("minrepro %s\n", q);
                    std::fflush(stdout);
                    syst.add_def(q);
                }
                std::printf("minrepro ALL SIX REGISTERED\n");
                std::fflush(stdout);
            }
            // ⚠ ONE EVALUATION FIRST, and it is a probe of the mechanism
            // rather than a superstition: the only placement where these defs
            // register is --extra-def's, which is past sec_member().  If what
            // that buys is a lazily-sized scratch pool -- Ope_add takes TWO
            // OperandScratchLeases at once -- then forcing one evaluation here
            // is the whole fix; if it is not, this costs one assembly and
            // rules the hypothesis out.
            if (std::getenv("THROATTH_NOPRIME") == nullptr) {
                Kadath::Array<double> warm(syst.sec_member());
                (void)warm;
            }
            try {
                // ⚠ DOMAIN 0 ONLY.  These rows live at the inner face
                // and nowhere else, and the pure-angular profiles are
                // filled with ZERO at the compact domain's r = infinity
                // node -- so SPS vanishes there, log(SPS) is -inf, and
                // reading the chain across all three domains evaluates an
                // amplitude reconstruction that has no meaning outside
                // domain 0.
                Trumpet::throatth_register(syst, space, ntheta,
                                   0, dtop,
                                   std::getenv("THROATTH_NOREAD")
                                   == nullptr, g1replace
#ifdef THROATTH_HAS_MAX
                                   , maximality
#endif
#ifdef THROATTH_HAS_U2
                                   , g2u2
#endif
                                   );
            } catch (const std::exception& ex) {
                if (rank == 0)
                std::cerr << "FATAL: throatth_register threw: "
                          << ex.what() << "\n";
                MPI_Finalize();
                return 20;
            }
            if (rank == 0)
                std::cout << "#  --jac-recursion: " << NG
                      << " grade-1 unknowns, "
                      << Trumpet::throatth_rows().size()
                      << " theta-ODE rows registered\n";
        }
        // ⚠ THE RECURSION ROWS -- the inner boundary condition itself
        // (research round 504).  Each is projected onto its own parity's
        // tau basis, so the six contribute nt-1 + nt + nt-2 + nt + nt + nt
        // = 6nt - 3, exactly the grade-1 block they close.  The count is
        // asserted here rather than trusted, because it is the whole
        // reason the system stays square.
        // ⚠ THROATTH_MAX truncates the def table for bisection, and then
        // the rows do not exist -- asking for them aborts on an unknown
        // name, which is what every bisection run was actually dying of
        // before this gate went in.
        if (jacrec && !jacrecearly
            && std::getenv("THROATTH_MAX") == nullptr) {
            int nrec = 0, k = 0;
            // ⚠ MEASURED, NOT DECLARED.  The first version summed the
            // emitter's own mode table and asserted that -- which checks the
            // table against itself, and it passed while the assembly came out
            // one row over.  get_nbr_conditions() before and after is what the
            // system actually did with each row.
            for (const auto& r : Trumpet::throatth_rows()) {
                // ⚠ get_nbr_conditions() IS NOT A RUNNING COUNT -- it came
                // back 2269 before the first row and 0 after, so it recomputes
                // rather than accumulates and cannot measure one row.  The
                // declared table is what is summed here; the REAL check is
                // FJPJ_rows against FJPJ_cols at the end, which is what caught
                // this table being one short.
                std::string eqt = std::string(r.first) + " = 0";
                if (droptephi && std::string(r.first) == "TEPHI") {
                    eqt = r.first;
                    char tb[32];
                    for (int q = 1; q < ntheta; q++) {
                        std::snprintf(tb, sizeof tb, " - WP%02d * MCE%02d",
                                      q, q);
                        eqt += tb;
                    }
                    eqt += " = 0";
                    if (rank == 0)
                        std::cout << "#  --drop-tephi: " << eqt.substr(0, 60)
                                  << " ...\n";
                }
                const std::string rn(r.first);
                if (g2u2 && rn == "TESIG") {
                    // TESIG with its u2 part (E_q(-2) + 2 E_sigma(-2), u2 in)
                    eqt = "TESIG + DTSIG = 0";
                    if (rank == 0)
                        std::cout << "#  --grade2-u2: TESIG registered as "
                                  << eqt << "\n";
                }
                if (balancel2 && rn == "TEPHI") {
                    // mode 0 ONLY: its ell >= 2 is replaced by the balance's.
                    Index pos_cf(space.get_domain(0)->get_nbr_coefs());
                    syst.add_eq_mode(0, INNER_BC, rn.c_str(), pos_cf, 0.0);
                    nrec += 1;
                    emit("FJPJ_balancel2_tephi_cut", ntheta - 1);
                    if (rank == 0)
                        std::cout << "#  --balance-l2: TEPHI at mode 0 only (1"
                                     " condition, was " << ntheta << "); its "
                                  << ntheta - 1 << " ell >= 2 modes go to the"
                                     " balance\n";
                } else if (maximality && rn == "TEQFN") {
                    // mode 0 ONLY: its ell >= 2 modes go to TMAX below.
                    Index pos_cf(space.get_domain(0)->get_nbr_coefs());
                    syst.add_eq_mode(0, INNER_BC, rn.c_str(), pos_cf, 0.0);
                    nrec += 1;
                    if (rank == 0)
                        std::cout << "#  --maximality: TEQFN at mode 0 only (1"
                                     " condition, was " << ntheta << ")\n";
                } else if (g1replace && (rn == "TEPHI" || rn == "TEQFN")) {
                    // mode 0 ONLY: the ell >= 2 content of these two is
                    // vacuous on the grade-1 block and is replaced below.
                    Index pos_cf(space.get_domain(0)->get_nbr_coefs());
                    syst.add_eq_mode(0, INNER_BC, (rn).c_str(), pos_cf, 0.0);
                    nrec += 1;
                    if (rank == 0)
                        std::cout << "#  --grade1-replace: " << rn
                                  << " at mode 0 only (1 condition, was "
                                  << Trumpet::throatth_row_modes(k, ntheta)
                                  << ")\n";
                } else {
                    syst.add_eq_bc(0, INNER_BC, eqt.c_str());
                    nrec += Trumpet::throatth_row_modes(k, ntheta);
                }
                k++;
            }
            if (g1replace) {
                int nadd = 0;
                for (const auto& r : Trumpet::throatth_replace()) {
                    for (int q = 1; q < ntheta; q++) {
                        Index pos_cf(space.get_domain(0)->get_nbr_coefs());
                        pos_cf.set(1) = q;
                        syst.add_eq_mode(0, INNER_BC, r.first, pos_cf, 0.0);
                        nadd++;
                    }
                    if (rank == 0)
                        std::cout << "#  --grade1-replace: " << r.first
                                  << " = " << r.second << " at modes 1.."
                                  << ntheta - 1 << " (" << ntheta - 1
                                  << " conditions)\n";
                }
                nrec += nadd;
                emit("FJPJ_g1replace_rows", nadd);
                emit("FJPJ_g1replace_cut", 2 * (ntheta - 1));
            }
            if (maximality) {
                for (int q = 1; q < ntheta; q++) {
                    Index pos_cf(space.get_domain(0)->get_nbr_coefs());
                    pos_cf.set(1) = q;
                    syst.add_eq_mode(0, INNER_BC, "TMAX", pos_cf, 0.0);
                }
                nrec += ntheta - 1;
                emit("FJPJ_maximality_rows", ntheta - 1);
                emit("FJPJ_maximality_teqfn_cut", ntheta - 1);
                if (rank == 0)
                    std::cout << "#  --maximality: TMAX = M(theta) registered at"
                                 " modes 1.." << ntheta - 1 << " by add_eq_mode ("
                              << ntheta - 1 << " conditions), in place of TEQFN's\n";
            }
            if (balancel2 && nrec == NG - (ntheta - 1)) {
                // the cut is paid for by the balance's nt rows below, less
                // the SBAL column: checked as SQUARE by FJPJ_rows/FJPJ_cols
                nrec += ntheta - 1;
            }
            if (nrec != NG) {
                if (rank == 0)
                std::cerr << "FATAL: the recursion rows project onto "
                          << nrec << " conditions, not " << NG
                          << " = 6nt - 3; the system cannot be square."
                          << "\n";
                MPI_Finalize();
                return 21;
            }
            // ⚠ THE BALANCE ROW, at the LATE site -- that is the one the
            // production invocation reaches.  --jac-recursion-early is what
            // selects the early branch, and registering there gave one new
            // COLUMN and ZERO new rows, which the count caught before any
            // solve was attempted.
            // ⚠ AND IT IS NOT COUNTED WITH get_nbr_conditions(), for the
            // reason recorded twenty lines up: that call RECOMPUTES rather
            // than accumulates and cannot measure one row.  The real check is
            // FJPJ_rows against FJPJ_cols; the declared count is nt because
            // the row is COS_EVEN.
            // ⚠ THE AXIS RESIDUE ROW.  Built here rather than in the
            // emitter because it is not a theta-ODE: it is an algebraic
            // relation among the scalar matching unknowns, and the emitter's
            // whole vocabulary is trig monomials in theta.
            if (jacaxisres) {
                const int NCE = ntheta, NSE = ntheta - 2;
                std::string sgps, sgbr, shps, shbr, sgbt;
                char b[64];
                for (int j = 0; j < NCE; j++) {
                    std::snprintf(b, sizeof b, "%sG%02d", j ? " + " : "", j);
                    sgps += b;
                    std::snprintf(b, sizeof b, "%sH%02d", j ? " + " : "", j);
                    shps += b;
                    std::snprintf(b, sizeof b, "%sG%02d", j ? " + " : "",
                                  3 * ntheta + j);
                    sgbr += b;
                    std::snprintf(b, sizeof b, "%sH%02d", j ? " + " : "",
                                  3 * ntheta + j);
                    shbr += b;
                }
                for (int j = 0; j < NSE; j++) {
                    std::snprintf(b, sizeof b, "%s%d * G%02d",
                                  j ? " + " : "", 4 * (j + 1),
                                  4 * ntheta + j);
                    sgbt += b;
                }
                char nb[64];
                std::snprintf(nb, sizeof nb, "%.17g", nexp);
                // ⚠ NO LEADING UNARY MINUS -- the condition is negated
                // wholesale instead.
                // ⚠ AND IT CANNOT GO THROUGH add_eq_bc.  The expression is
                // PURELY SCALAR: every factor is a G/H unknown, so there is
                // no field to project onto a tau basis and the assembly
                // segfaults (it does register and log first, so the log line
                // is not proof the row took -- the count is).  Multiplying by
                // the cst ACE00 == 1 gives it an angular carrier that is
                // identically theta-independent, so mode 0 IS the residue and
                // every other mode is exactly zero; add_eq_mode then takes
                // that one coefficient and yields ONE row, where add_eq_bc
                // would yield nt of which nt-1 are identically zero.
                std::string eq = "((" + sgps + ") * ((" + sgbt + ") + "
                    + nb + " * (" + shbr + ")) + 3 * " + nb
                    + " * (" + sgbr + ") * (" + shps + ")) * ACE00";
                Index pos_cf(space.get_domain(0)->get_nbr_coefs());
                syst.add_eq_mode(0, INNER_BC, eq.c_str(), pos_cf, 0.0);
                emit("FJPJ_axisres_rows", 1);
                if (rank == 0)
                    std::cout << "#  --jac-axisres: TRES registered at n = "
                              << nb << " by add_eq_mode, 1 scalar condition "
                                 "(rows +1, columns +0); " << eq.substr(0, 60)
                              << " ...\n";
            }
            if (jacbalance && Trumpet::throatth_balance() != nullptr) {
                const std::string bal = std::string(Trumpet::throatth_balance())
                    + (g2u2 ? " + DTBAL" : "") + " - SPH * SBAL = 0";
                syst.add_eq_bc(0, INNER_BC, bal.c_str());
                if (g2u2 && rank == 0)
                    std::cout << "#  --grade2-u2: TBAL registered as " << bal
                              << "\n";
                emit("FJPJ_balance_rows", ntheta);
                if (rank == 0)
                    std::cout << "#  --jac-balance: TBAL - SPH*SBAL registered"
                                 ", declared " << ntheta << " modes (net "
                              << ntheta - 1 << " conditions against 1 new "
                                 "unknown)\n";
            }
#ifdef THROATTH_HAS_U2
            if (g2u2) {
                // E_sigma(-2) with u2, all nt COS_EVEN modes: the nt rows that
                // pay for the nt UU columns.
                syst.add_eq_bc(0, INNER_BC, "TSIGU = 0");
                emit("FJPJ_grade2u2_rows", ntheta);
                if (rank == 0)
                    std::cout << "#  --grade2-u2: TSIGU = E_sigma(-2) (u2 in) = 0"
                                 " registered, declared " << ntheta << " modes"
                                 " (rows +" << ntheta << ", cols +" << ntheta
                              << ")\n";
            }
#endif
            emit("FJPJ_recursion_rows", nrec);
            if (rank == 0)
                std::cout << "#  --jac-recursion: " << nrec
                      << " conditions from 6 rows (6nt - 3 = " << NG
                      << ")\n";
        }
        // ---- --newton-delta: apply the external step BEFORE anything is read
        if (!newtondelta.empty()) {
            const int nu = syst.get_nbr_unknowns();
            std::vector<double> dv(nu, 0.0);
            std::ifstream df(newtondelta);
            if (!df) {
                if (rank == 0)
                    std::cerr << "FATAL: cannot open --newton-delta "
                              << newtondelta << "\n";
                MPI_Finalize();
                return 19;
            }
            std::string tok; int idx; double val; int nread = 0;
            while (df >> tok) {
                if (tok != "d") { std::getline(df, tok); continue; }
                df >> idx >> val;
                if (idx < 0 || idx >= nu) {
                    if (rank == 0)
                        std::cerr << "FATAL: --newton-delta index " << idx
                                  << " outside [0, " << nu << ")\n";
                    MPI_Finalize();
                    return 19;
                }
                dv[idx] = val; nread++;
            }
            if (nread != nu) {
                // ⚠ REFUSE on a short file.  A delta with missing entries is a
                // DIFFERENT step, silently, and the iteration would converge to
                // something nobody asked for.
                if (rank == 0)
                    std::cerr << "FATAL: --newton-delta has " << nread
                              << " entries, needs " << nu << "\n";
                MPI_Finalize();
                return 19;
            }
            // ⚠ THE FIELDS ARE SNAPSHOT FIRST.  "max|dF| / max|F|" is the
            // quantity that was 2-5 for the exact step and must be O(alpha)
            // here, and it has no answer unless the pre-step state is held
            // somewhere xx_to_vars_delta cannot reach.
            Scalar DS0(PS), DH0(PH), DF0(QF), DR0(BR), DT0(BT), DB0(QB);
            Kadath::Array<double> X(nu);
            double dmax = 0.0;
            for (int i = 0; i < nu; i++) {
                X.set(i) = dv[i];
                dmax = std::max(dmax, std::fabs(dv[i]));
            }
            int conte = 0;
            space.xx_to_vars_variable_domains(&syst, X, conte);
            syst.xx_to_vars_delta(X, conte);
            emit("FJPN_delta_applied", 1.0);
            emit("FJPN_delta_max", dmax);
            emit("FJPN_delta_n", nu);
            if (rank == 0)
                std::cout << "#  --newton-delta applied: " << nu
                          << " entries, max|d| = " << dmax << "\n";
            {   // the field-space size of the step just applied
                const char* dn[6] = {"PS", "PH", "QF", "BR", "BT", "QB"};
                const Scalar* p0[6] = {&DS0, &DH0, &DF0, &DR0, &DT0, &DB0};
                const Scalar* p1[6] = {&PS, &PH, &QF, &BR, &BT, &QB};
                if (rank == 0)
                    std::cout << "#  dstep  fld  max|dF|        max|F|         "
                                 "rel\n";
                for (int q = 0; q < 6; q++) {
                    double dm = 0.0, fm = 0.0;
                    for (int d = 0; d <= dtop; d++) {
                        Index ix(space.get_domain(d)->get_nbr_points());
                        do {
                            const double a = (*p0[q])(d)(ix), c2 = (*p1[q])(d)(ix);
                            if (!std::isfinite(a) || !std::isfinite(c2)) continue;
                            fm = std::max(fm, std::fabs(a));
                            dm = std::max(dm, std::fabs(c2 - a));
                        } while (ix.inc());
                    }
                    emit(std::string("FJPN_ddmax_") + dn[q], dm);
                    emit(std::string("FJPN_dfmax_") + dn[q], fm);
                    emit(std::string("FJPN_ddrel_") + dn[q], fm > 0.0 ? dm / fm : 0.0);
                    if (rank == 0) {
                        char line[160];
                        std::snprintf(line, sizeof line,
                                      "%-3s  %-14.6e %-14.6e %-14.6e",
                                      dn[q], dm, fm, fm > 0.0 ? dm / fm : 0.0);
                        std::cout << "#  dstep  " << line << "\n";
                    }
                }
            }
            if (!fieldspost.empty() && rank == 0) {
                std::ofstream ff(fieldspost);
                ff << "# field values AFTER --newton-delta, by --dump-fields-post\n";
                ff << "fields PS PH QF BR BT QB\n";
                ff << std::setprecision(17);
                const Scalar* fp[6] = {&PS, &PH, &QF, &BR, &BT, &QB};
                for (int d = 0; d <= dtop; d++) {
                    Index ix(space.get_domain(d)->get_nbr_points());
                    do {
                        ff << "val " << d << " " << ix(0) << " " << ix(1)
                           << " " << t.pts[d][ix(0)].r
                           << " " << space.get_domain(d)->get_coloc(2)(ix(1));
                        for (int q = 0; q < 6; q++)
                            ff << " " << (*fp[q])(d)(ix);
                        ff << "\n";
                    } while (ix.inc());
                }
                std::cout << "#  --dump-fields-post: written to "
                          << fieldspost << "\n";
            }
        }
        Kadath::Array<double> bb(syst.sec_member());
        const int nrow = syst.get_nbr_conditions();
        const int ncol = syst.get_nbr_unknowns();
        {   // ⚠ the RHS is what distinguishes "no tail" from "the backbone's
            // tail": a row imposing SOME falloff assembles and counts exactly
            // like the right one, and only the residual tells them apart.
            double bmax = 0.0;
            int bnf = 0;
            for (int r = 0; r < nrow; r++) {
                if (!std::isfinite(bb(r))) { bnf++; continue; }
                bmax = std::max(bmax, std::fabs(bb(r)));
            }
            emit("FJPJ_rhs_max", bmax);
            // ⚠ THE DOMAIN TEST (research round 508 ruling 3).  Phibar =
            // alpha psi^2 > 0 and 1/Phibar is in the emission, so a step that
            // drives Phibar through zero leaves the operator's DOMAIN and the
            // assembly is NaN -- which round 209 found only by re-assembling
            // by hand, after reporting the residual.  It is counted here
            // because std::max(x, NaN) returns x: FJPJ_rhs_max cannot see a
            // NaN, exactly as do_newton's own max could not in round 201.
            double phmin = 1e300;
            int phnf = 0;
            for (int d = 0; d <= dtop; d++) {
                Index ip(space.get_domain(d)->get_nbr_points());
                do {
                    const double v = PH(d)(ip);
                    if (!std::isfinite(v)) { phnf++; continue; }
                    phmin = std::min(phmin, v);
                } while (ip.inc());
            }
            emit("FJPJ_min_PH", phmin);
            emit("FJPJ_nonfinite_PH", phnf);
            emit("FJPJ_nonfinite_rhs", bnf);
            emit("FJPJ_in_domain", (phmin > 0.0 && phnf == 0 && bnf == 0)
                                   ? 1.0 : 0.0);
            if (rank == 0)
                std::cout << "#   domain test: min Phibar " << phmin
                          << ", non-finite  Phibar " << phnf
                          << "  rhs " << bnf
                          << (phmin > 0.0 && phnf == 0 && bnf == 0
                              ? "   IN DOMAIN\n" : "   *** OUT OF DOMAIN ***\n");
            emit("FJPJ_outer_perturb", outerpert);
            emit("FJPJ_outer_perturb_pow", outerpow);
        }
        emit("FJPJ_rows", nrow);
        emit("FJPJ_cols", ncol);
        emit("FJPJ_m_minus_n", nrow - ncol);
        if (rank == 0 && !jacdump.empty()) {
            std::ofstream fh(jacdump);
            fh << "# Jacobian of the finite-J BULK operator (no BC rows)  rows "
               << nrow << " cols " << ncol << "\n";
            fh << "# ntheta " << ntheta << " dtop " << dtop << "\n";
            fh << std::setprecision(17);
            for (int r = 0; r < nrow; r++)
                fh << "rhs " << r << " " << bb(r) << "\n";
            if (!basisprobe.empty()) {
                // only the RHS is read; the columns would cost ncol solves and
                // carry no information about the plant.
                emit("FJPJ_nnz", 0.0);
                std::cout << "# basis-probe RHS written to " << jacdump << "\n";
            } else {
            long long nnz = 0;
            // ---- THE INCIDENCE CONTROL for --jac-match ---------------------
            // ⚠ WHAT THIS CHECKS, AND WHY IT IS THE RIGHT CHECK.  Each matching
            // unknown is specified to be the amplitude of ONE angular mode of
            // ONE field, so its Jacobian column must have EXACTLY ONE nonzero,
            // and that row must be an inner row, and the map unknown -> row
            // must be injective.  An amplitude whose profile is not
            // representable in the basis its row is projected onto (round 103
            // measured cos(theta) falling outside the scalar COS_EVEN span)
            // would alias across modes.
            // ⚠ IT DOES NOT TEST THE BASIS ASSIGNMENT.  --jac-match-break was
            // written expecting it to and measured that it does not; see the
            // note at that flag's declaration.  What it tests is that the
            // matching block is an INCIDENCE block: one row per unknown, one
            // unknown per row, coefficient -1, every row inside the inner
            // block.  That is what the kernel measurement rests on.
            const int nmatch = jacmatch ? (match_ncol_after - match_ncol_before)
                                        : 0;
            if (jacmatch) {
                inner_row_begin = nrow - (12 * ntheta - 6);
                inner_row_end = nrow;
                emit("FJPJ_inner_row_begin", inner_row_begin);
                emit("FJPJ_inner_row_end", inner_row_end);
            }
            // ⚠ THE THRESHOLD IS A TOLERANCE, NOT `!= 0.0`.  The first version
            // of this check counted raw nonzeros and reported 8 of 34 columns
            // single-row; every column in fact carried ONE entry of -1 and up
            // to four more at 1e-17, which are the tau transform's roundoff.
            // A check that cannot tell -1 from 1e-17 reports on the arithmetic.
            const double MC_TOL = 1e-10;
            int mc_min = 1 << 30, mc_max = 0, mc_ok = 0, mc_outside = 0;
            double mc_vmin = 1e300, mc_vmax = 0.0;
            std::vector<int> mc_row;
            for (int c = 0; c < ncol; c++) {
                Kadath::Array<double> col(syst.do_col_J(c));
                double cmax = 0.0;
                if (c < nmatch)
                    for (int r = 0; r < nrow; r++)
                        cmax = std::max(cmax, std::fabs(col(r)));
                int ccount = 0, crow = -1;
                for (int r = 0; r < nrow; r++)
                    if (col(r) != 0.0) {
                        fh << "J " << r << " " << c << " " << col(r) << "\n";
                        nnz++;
                        if (c < nmatch && std::fabs(col(r)) > MC_TOL * cmax) {
                            ccount++; if (crow < 0) crow = r;
                        }
                    }
                if (c < nmatch) {
                    mc_min = std::min(mc_min, ccount);
                    mc_max = std::max(mc_max, ccount);
                    mc_vmin = std::min(mc_vmin, cmax);
                    mc_vmax = std::max(mc_vmax, cmax);
                    if (ccount == 1) { mc_ok++; mc_row.push_back(crow); }
                    if (crow >= 0 && (crow < inner_row_begin
                                      || crow >= inner_row_end)) mc_outside++;
                }
            }
            if (jacmatch) {
                std::vector<int> u(mc_row);
                std::sort(u.begin(), u.end());
                u.erase(std::unique(u.begin(), u.end()), u.end());
                emit("FJPJ_match_cols", nmatch);
                emit("FJPJ_match_col_nnz_min", mc_min == (1 << 30) ? -1 : mc_min);
                emit("FJPJ_match_col_nnz_max", mc_max);
                emit("FJPJ_match_single_row_cols", mc_ok);
                emit("FJPJ_match_distinct_rows", int(u.size()));
                emit("FJPJ_match_rows_outside_inner", mc_outside);
                emit("FJPJ_match_col_absmax_min", mc_vmin);
                emit("FJPJ_match_col_absmax_max", mc_vmax);
                std::cout << "#   --jac-match: " << nmatch << " unknown columns, "
                          << mc_ok << " single-row, " << u.size()
                          << " distinct rows, " << mc_outside
                          << " outside the inner block\n";
            }
            emit("FJPJ_nnz", double(nnz));
            emit("FJPJ_density", double(nnz) / (double(nrow) * double(ncol)));
            std::cout << "# bulk Jacobian written to " << jacdump << "\n";
            if (!rowmeta.empty()) {
                std::ofstream rm(rowmeta + ".rows"), cm(rowmeta + ".cols"),
                    em(rowmeta + ".eqs");
                syst.dump_tagged_jacobian_metadata_csv(rm, cm);
                std::vector<Kadath::System_of_eqs::RowMetadata> rmeta;
                syst.classify_equation_row_metadata(rmeta);
                const auto& EL = EqListPeek::eqs(syst);
                const auto& EI = EqListPeek::eqints(syst);
                em << "# kind index dom expression\n";
                for (std::size_t q = 0; q < EL.size(); q++)
                    em << "eq " << q << " " << std::get<1>(EL[q]) << " "
                       << std::get<0>(EL[q]) << "\n";
                for (std::size_t q = 0; q < EI.size(); q++)
                    em << "eqint " << q << " " << std::get<1>(EI[q]) << " "
                       << std::get<0>(EI[q]) << "\n";
                emit("FJPJ_rowmeta_rows", int(rmeta.size()));
                emit("FJPJ_rowmeta_matches", int(rmeta.size()) == nrow ? 1 : 0);
                emit("FJPJ_rowmeta_eqs", int(EL.size()));
                emit("FJPJ_rowmeta_eqints", int(EI.size()));
                std::cout << "#   --dump-rowmeta: " << rmeta.size()
                          << " row tags (" << nrow << " rows), " << EL.size()
                          << " equations + " << EI.size()
                          << " integral equations -> " << rowmeta << ".*\n";
            }
            }
        }

        // ---- --newton: THE SOLVE (round 201) ------------------------------
        // ⚠ WHAT IS READ, AND WHY NOT do_newton's OWN NUMBER.  do_newton
        // reports max |residual| over rows.  The six equations carry no radial
        // prefactor and their leading grades run to r^{-2n-2} (round 190), so
        // at r_m = 5.4e-3 a bulk row's natural size is ~1e11 and the absolute
        // max is a statement about the worst-scaled row, not about the solve.
        // The reading is round 198's RELATIVE C1 diagnostic, per field, before
        // and after, plus the FIELD CHANGE -- because at cond ~ 2.6e18 a
        // backward-stable solve drives the residual down whatever the step is,
        // and only the size and smoothness of the step separates a real
        // correction from amplified roundoff.
        if (donewton) {
            const char* fnm[6]  = {"PS", "PH", "QF", "BR", "BT", "QB"};
            const int   fcnt[6] = {NCE, NCE, NCE, NCE, NSE, NCO};
            const double fg0[6] = {-1.0, nexp - 1.0, 0.0, 1.0, nexp, 0.0};
            const double rm = t.pts[0][0].r;
            const int irb = nrow - (12 * ntheta - 6);
            int gb[6]; { int a = 0; for (int q = 0; q < 6; q++) { gb[q] = a; a += fcnt[q]; } }
            emit("FJPN_r_match", rm);
            emit("FJPN_nexp", nexp);
            emit("FJPN_inner_row_begin", irb);

            // the relative C1 (and C0) diagnostic, read off a residual vector
            auto report = [&](const char* tag, const Kadath::Array<double>& b) {
                if (rank != 0) return;
                std::cout << "#  " << tag
                          << "   fld  |C0 resid|     |C1 resid|     rel C1\n";
                for (int q = 0; q < 6; q++) {
                    const int r0 = irb + 2 * gb[q];
                    double n0 = 0.0, n1 = 0.0, den = 0.0;
                    for (int j = 0; j < fcnt[q]; j++) {
                        n0 = std::max(n0, std::fabs(b(r0 + j)));
                        n1 = std::max(n1, std::fabs(b(r0 + fcnt[q] + j)));
                        // |G_j * dr(r^{g0})| at r_m -- zero for g0 = 0 and for
                        // the three fields whose amplitude vanishes at J = 0,
                        // and those are reported UNDEFINED rather than as 0.
                        den = std::max(den, std::fabs(matchv[gb[q] + j] * fg0[q]
                                                      * std::pow(rm, fg0[q] - 1.0)));
                    }
                    char line[160];
                    if (den > 0.0)
                        std::snprintf(line, sizeof line,
                                      "%-6s %-3s  %-14.6e %-14.6e %.6e",
                                      tag, fnm[q], n0, n1, n1 / den);
                    else
                        std::snprintf(line, sizeof line,
                                      "%-6s %-3s  %-14.6e %-14.6e UNDEFINED (G dr(r^g0) = 0)",
                                      tag, fnm[q], n0, n1);
                    std::cout << "#  " << line << "\n";
                    emit(std::string("FJPN_") + tag + "_C0_" + fnm[q], n0);
                    emit(std::string("FJPN_") + tag + "_C1_" + fnm[q], n1);
                    if (den > 0.0)
                        emit(std::string("FJPN_") + tag + "_rel_" + fnm[q], n1 / den);
                }
            };
            report("pre", bb);

            // ⚠ THE FIELDS ARE COPIED, not re-read.  do_newton updates the
            // registered Scalars in place, so "did it move" has no answer
            // unless the seed is held somewhere the solve cannot reach.
            Scalar PS0(PS), PH0(PH), QF0(QF), BR0(BR), BT0(BT), QB0(QB);
            std::vector<double> G0(matchv);

            double err = 0.0;
            bool ok = false;
            int it = 0;
            bool threw = false;
            try {
                for (it = 0; it < newtonmax && !ok; it++) {
                    ok = syst.do_newton(newtonprec, err);
                    emit("FJPN_err_" + std::to_string(it), err);
                    if (rank == 0)
                        std::cout << "#  newton step " << it
                                  << "  max|resid| = " << std::setprecision(10)
                                  << err << (ok ? "   (converged)" : "") << "\n";
                }
            } catch (const std::exception& e) {
                threw = true;
                if (rank == 0)
                    std::cerr << "FATAL in do_newton: " << e.what() << "\n";
            }
            emit("FJPN_threw", threw ? 1.0 : 0.0);
            emit("FJPN_ok", ok ? 1.0 : 0.0);
            emit("FJPN_iters", it);
            emit("FJPN_err", err);
            if (threw) { MPI_Finalize(); return 17; }

            Kadath::Array<double> b2(syst.sec_member());
            report("post", b2);

            // ---- THE STEP.  Per field, max |dF| and max |dF| / max |F|, over
            // the probed domains only -- the compact domain's r = infinity node
            // is included because that is where the outer rows live.
            const Scalar* f0[6] = {&PS0, &PH0, &QF0, &BR0, &BT0, &QB0};
            const Scalar* f1[6] = {&PS,  &PH,  &QF,  &BR,  &BT,  &QB};
            if (rank == 0)
                std::cout << "#  step   fld  max|dF|        max|F|         "
                             "rel            worst at (dom,ir,ith)\n";
            for (int q = 0; q < 6; q++) {
                double dm = 0.0, fm = 0.0;
                int wd = -1, wr = -1, wt = -1;
                for (int d = 0; d <= dtop; d++) {
                    Index ix(space.get_domain(d)->get_nbr_points());
                    do {
                        const double a = (*f0[q])(d)(ix), b = (*f1[q])(d)(ix);
                        if (!std::isfinite(a) || !std::isfinite(b)) continue;
                        fm = std::max(fm, std::fabs(a));
                        if (std::fabs(b - a) > dm) {
                            dm = std::fabs(b - a); wd = d; wr = ix(0); wt = ix(1);
                        }
                    } while (ix.inc());
                }
                emit(std::string("FJPN_dmax_") + fnm[q], dm);
                emit(std::string("FJPN_fmax_") + fnm[q], fm);
                emit(std::string("FJPN_drel_") + fnm[q], fm > 0.0 ? dm / fm : 0.0);
                if (rank == 0) {
                    char line[200];
                    std::snprintf(line, sizeof line,
                                  "%-3s  %-14.6e %-14.6e %-14.6e (%d,%d,%d)",
                                  fnm[q], dm, fm, fm > 0.0 ? dm / fm : 0.0, wd, wr, wt);
                    std::cout << "#  step   " << line << "\n";
                }
            }
            // ---- THE AMPLITUDES.  These are the six leading unknowns, and a
            // step that is physics moves them by O(alpha) too.
            if (rank == 0)
                std::cout << "#  ampl   fld  max|dG|        max|G0|        rel\n";
            for (int q = 0; q < 6; q++) {
                double dm = 0.0, gm = 0.0;
                for (int j = 0; j < fcnt[q]; j++) {
                    dm = std::max(dm, std::fabs(matchv[gb[q] + j] - G0[gb[q] + j]));
                    gm = std::max(gm, std::fabs(G0[gb[q] + j]));
                }
                emit(std::string("FJPN_dG_") + fnm[q], dm);
                emit(std::string("FJPN_G0_") + fnm[q], gm);
                emit(std::string("FJPN_dGrel_") + fnm[q], gm > 0.0 ? dm / gm : 0.0);
                if (rank == 0) {
                    char line[160];
                    std::snprintf(line, sizeof line, "%-3s  %-14.6e %-14.6e %-14.6e",
                                  fnm[q], dm, gm, gm > 0.0 ? dm / gm : 0.0);
                    std::cout << "#  ampl   " << line << "\n";
                }
            }
            if (rank == 0 && !newtonfields.empty()) {
                std::ofstream nf(newtonfields);
                nf << "# fields AFTER --newton\n";
                nf << "fields PS PH QF BR BT QB\n";
                nf << std::setprecision(17);
                for (int d = 0; d <= dtop; d++) {
                    Index ix(space.get_domain(d)->get_nbr_points());
                    do {
                        nf << "val " << d << " " << ix(0) << " " << ix(1)
                           << " " << t.pts[d][ix(0)].r
                           << " " << space.get_domain(d)->get_coloc(2)(ix(1));
                        for (int q = 0; q < 6; q++) nf << " " << (*f1[q])(d)(ix);
                        nf << "\n";
                    } while (ix.inc());
                }
                nf << "# amplitudes G, in registration order PS PH QF BR BT QB\n";
                for (int g = 0; g < NG; g++)
                    nf << "amp " << g << " " << matchv[g] << " " << G0[g] << "\n";
                std::cout << "# post-Newton fields written to " << newtonfields << "\n";
            }
        }
    }

    // ---- --tail-readout: read the 1/r coefficients at the r = infinity node
    if (tailout) {
        const char* tn[4] = {"TAILPS", "TAILPH", "TAILPHP", "TAILQF"};
        const Kadath::Domain* dm = space.get_domain(dtop);
        const int nr = dm->get_nbr_points()(0);
        if (rank == 0)
            std::cout << "#  tail   def        theta      1/r coefficient\n";
        for (int q = 0; q < 4; q++) {
            const Val_domain& v =
                syst.give_val_def_scalar_domain(tn[q], dtop);
            Index ix(dm->get_nbr_points());
            double mx = 0.0;
            do {
                if (ix(0) != nr - 1) continue;       // the r = infinity node
                const double x = v(ix);
                mx = std::max(mx, std::fabs(x));
                if (rank == 0) {
                    char line[120];
                    std::snprintf(line, sizeof line, "%-10s %-10.6f %+.10e",
                                  tn[q], dm->get_coloc(2)(ix(1)), x);
                    std::cout << "#  tail   " << line << "\n";
                }
            } while (ix.inc());
            emit(std::string("FJP_tail_") + tn[q], mx);
        }
    }

    // ---- --dump-defs FILE: every sub-def's value at every collocation point.
    // The bisection instrument (round 126).  Written from Kadath so the
    // comparison against the emitter's sympy twins happens offline, in one
    // place, in topological order.
    //
    // ⚠ FETCHING IS NOT READING (round 108).  give_val_def_scalar_domain alone
    // forces nothing; the value only exists once operator()(Index) has been
    // called, which the write below does for every point.
    if (!defsout.empty() && rank == 0) {
        std::ofstream df(defsout);
        df << "# sub-def values written by finiteJ_probe --dump-defs\n";
        df << std::setprecision(17);
        for (const char* nm : Trumpet::finiteJ_names()) {
            for (int d = 0; d <= dtop; d++) {
                const Val_domain& v = syst.give_val_def_scalar_domain(nm, d);
                Index idx(space.get_domain(d)->get_nbr_points());
                do {
                    df << "def " << nm << ' ' << d << ' ' << idx(0) << ' '
                       << idx(1) << ' ' << v(idx) << '\n';
                } while (idx.inc());
            }
        }
        std::cout << "# sub-def values written to " << defsout << "\n";
    }

    // ---- THE CONTROL.  Every sum re-expressed with its operands read first.
    if (maxdefs < 0) {
        double readchk = 0.0;
        try {
            readchk = Trumpet::finiteJ_check(syst, space, 0, dtop);
            if (rank == 0)
                std::cout << "# read-state contract kept; worst re-expression "
                             "difference " << readchk << "\n";
        } catch (const std::exception& ex) {
            readchk = 1.0;
            if (rank == 0)
                std::cout << "# read-state contract BROKEN: " << ex.what()
                          << "\n";
        }
        emit("FJP_readcheck", readchk);
    }

    if (rank == 0)
        std::cout << "# all six defs registered\n";

#ifdef TRUMPET_CHI_UNKNOWN
    // ---- THE CH SEED, AND THE J = 0 CONTROL ------------------------------
    // ⚠ THE SEED IS KADATH'S OWN EVALUATION OF THE PRODUCT, not a second
    // transcription of it.  D0048 is the product the emission already builds, so
    // reading it and writing it into CH makes the seed consistent by
    // construction -- and that is exactly what lets ECHI's residual be a
    // CONTROL rather than a tautology: before the plant it must equal the
    // product's own magnitude, after it must be at the truncation.  Both are
    // printed.  If the "after" number is not at the truncation then CH's basis,
    // its registration, or the emitted ECHI text disagree with the product, and
    // nothing downstream is worth building.
    //
    // ⚠ Round 108: FETCHING A Val_domain IS NOT READING IT.  operator()(Index)
    // is what forces configuration space, so the loop indexes every point.
    if (maxdefs < 0) {
        double before = 0.0, after = 0.0, prodmax = 0.0;
        for (int d = 0; d <= dtop; d++) {
            const Val_domain& ve = syst.give_val_def_scalar_domain("ECHI", d);
            Index ix(space.get_domain(d)->get_nbr_points());
            do {
                const double x = ve(ix);
                before = std::max(before, std::isfinite(x) ? std::fabs(x) : 1e300);
            } while (ix.inc());
        }
        for (int d = 0; d <= dtop; d++) {
            const Val_domain& vp = syst.give_val_def_scalar_domain("D0048", d);
            Val_domain& vc = CH.set_domain(d);
            Index ix(space.get_domain(d)->get_nbr_points());
            do {
                const double x = vp(ix);
                vc.set(ix) = std::isfinite(x) ? x : 0.0;
                prodmax = std::max(prodmax, std::isfinite(x) ? std::fabs(x) : 0.0);
            } while (ix.inc());
        }
        CH.std_base();
        // ⚠⚠ WITHOUT THIS THE RE-PLANT IS A NO-OP, and round 263 shipped it as
        // one: max|ECHI| came back identical to ten digits before and after,
        // which I recorded as "a cached read".  It is not a cache.  add_var
        // registers CH as an unknown and the system holds its own Term_eq COPY
        // of it, so writing into the Scalar afterwards changes nothing the
        // equations can see.  vars_to_terms() is the documented sync --
        // "copies the various unknowns into their Term_eq counterparts"
        // (system_of_eqs.hpp:1236).
        // ⚠ AND THIS IS THE SEED FIX RESEARCH ROUND 557 RULED.  This block runs
        // AFTER xx_to_vars_delta, so D0048 here is the product on THE STATE, not
        // on the backbone -- CH starts 60% closer and no chain spends its first
        // iterations re-discovering the seed.
        syst.vars_to_terms();
        // ⚠ DIRECT DIFF, not through ECHI.  Separates "the plant did not take"
        // from "the def read is cached": this compares the two Val_domains
        // itself, so it cannot be answered by a stale ECHI.
        {
            double direct = 0.0, chmax = 0.0;
            for (int d = 0; d <= dtop; d++) {
                const Val_domain& vp = syst.give_val_def_scalar_domain("D0048", d);
                const Val_domain& vc = CH(d);
                Index ix(space.get_domain(d)->get_nbr_points());
                do {
                    const double a = vc(ix), b2 = vp(ix);
                    if (std::isfinite(a)) chmax = std::max(chmax, std::fabs(a));
                    if (std::isfinite(a) && std::isfinite(b2))
                        direct = std::max(direct, std::fabs(a - b2));
                } while (ix.inc());
            }
            if (rank == 0)
                std::cout << "#  ⚠ DIRECT max|CH - D0048| after the plant = "
                          << direct << "   max|CH| = " << chmax << "\n";
            emit("FJP_chi_direct_after", direct);
        }
        for (int d = 0; d <= dtop; d++) {
            const Val_domain& ve = syst.give_val_def_scalar_domain("ECHI", d);
            Index ix(space.get_domain(d)->get_nbr_points());
            do {
                const double x = ve(ix);
                after = std::max(after, std::isfinite(x) ? std::fabs(x) : 1e300);
            } while (ix.inc());
        }
        if (rank == 0) {
            std::cout << "#  --chi-unknown: max|D0048| (Kadath's own product) = "
                      << std::setprecision(10) << prodmax << "\n";
            std::cout << "#  ⚠ J=0 CONTROL  max|ECHI| = max|CH - D0048| = "
                      << before << "   relative " << (prodmax > 0.0
                          ? before / prodmax : -1.0)
                      << "\n     (CH seeded from the INDEPENDENT closed form"
                         " -(PS/PH) BR (W/r); this is the two evaluations"
                         " against each other, and it can fail)\n";
            std::cout << "#  ⚠ AFTER re-planting CH from D0048 ON THE STATE"
                         " (the round-557 seed fix), max|ECHI| = " << after
                      << "   relative " << (prodmax > 0.0 ? after / prodmax
                                            : -1.0)
                      << "\n     (at J = 0 this must fall to the truncation;"
                         " at finite J it is the seed CH now starts from)\n";
        }
        emit("FJP_chi_prodmax", prodmax);
        emit("FJP_chi_echi_before", before);
        emit("FJP_chi_echi_after", after);
        emit("FJP_chi_echi_rel", prodmax > 0.0 ? after / prodmax : -1.0);
    }
#endif

    std::vector<std::string> extra_names;
    // ⚠ The read has to INDEX the Val_domain, not just fetch the reference:
    // operator()(Index) is what forces configuration space.  A fetch alone
    // leaves the def in whatever space it was built in, which is why the first
    // version of this probe found no trigger.
    {
        const int dR = (dtop >= 1) ? 1 : 0;
        auto peek = [&](const char* nm) {
            const Val_domain& v = syst.give_val_def_scalar_domain(nm, dR);
            Index iq(space.get_domain(dR)->get_nbr_points());
            (void)v(iq);
        };
        if (readall)
            for (int si = 0; si < nsub; si++) peek(SUB[si].name);
        for (const auto& nm : readbefore) peek(nm.c_str());
    }
    if (extra_early)
        for (const auto& x : extra) {
            const std::string nm = x.substr(0, x.find(' '));
            try {
                syst.add_def(x.c_str());
                extra_names.push_back(nm);
            } catch (const std::exception& ex) {
                std::cout << "#   extra " << nm << " THREW " << ex.what() << "\n";
            }
        }

    // ---- SUB-DEF LOCALISATION (research round 301 item 1) -----------------
    // Equation-level residuals are the wrong granularity.  Every sub-def is
    // reported AT ONE POINT so the same quantity can be evaluated through sympy
    // on the same seed and the FIRST disagreement found.
    {
        const int dP = (dtop >= 1) ? 1 : 0, iP = 3, jP = 2;
        const Kadath::Domain* dm = space.get_domain(dP);
        Index ix(dm->get_nbr_points());
        for (int k = 0; k < iP + jP * dm->get_nbr_points()(0); k++) ix.inc();
        emit("FJP_pt_dom", dP);
        emit("FJP_pt_r", t.pts[dP][iP].r);
        emit("FJP_pt_R", t.pts[dP][iP].Rr * t.pts[dP][iP].r);
        emit("FJP_pt_W", t.pts[dP][iP].W);
        emit("FJP_pt_th", dm->get_coloc(2)(jP));
        if (!clean)
        for (const char* nm : {"P1", "P2", "P3", "P4", "P5", "P6", "P7",
                               "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7",
                               "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7",
                               "Z8", "Z9", "Y1", "Y2", "Y3", "Y4", "Y5",
                               "Z10", "Z11", "L1", "L2", "L3", "L4", "L5"})
            emit(std::string("FJP_val_") + nm,
                 syst.give_val_def_scalar_domain(nm, dP)(ix));
        for (int si = 0; si < nsub; si++)
            emit(std::string("FJP_val_") + SUB[si].name,
                 syst.give_val_def_scalar_domain(SUB[si].name, dP)(ix));
        if (maxdefs < 0)
            for (const char* nm : Trumpet::finiteJ_eq_names())
                emit(std::string("FJP_val_") + nm,
                     syst.give_val_def_scalar_domain(nm, dP)(ix));
        for (const auto& nm : extra_names)
            emit(std::string("FJP_extra_") + nm,
                 syst.give_val_def_scalar_domain(nm.c_str(), dP)(ix));
        if (!extra_early) {
            for (const auto& x : extra) {
                const std::string nm = x.substr(0, x.find(' '));
                try {
                    syst.add_def(x.c_str());
                } catch (const std::exception& ex) {
                    std::cout << "#   extra " << nm << " THREW " << ex.what()
                              << "\n";
                    continue;
                }
                emit(std::string("FJP_extra_") + nm,
                     syst.give_val_def_scalar_domain(nm.c_str(), dP)(ix));
            }
        }
    }

    // ⚠ Split AXIS from INTERIOR.  The cot(theta) construction is only valid
    // on an operand that vanishes on the axis; if the failures sit at
    // idx(1) == 0 that is the cause, and if they are spread it is not.
    double worst = 0.0, worst_int = 0.0;
    // ⚠ --dump-eqvals writes the residual at EVERY collocation point, with its
    // r and theta, so the verdict can be a per-mode norm.  Opened before the
    // loop and written inside it, reusing the same give_val_def_scalar_domain
    // the max readout uses -- one evaluation, two readouts, no chance of the
    // dump and the printed max being different objects.
    std::ofstream eqf;
    if (!eqvalsout.empty() && rank == 0) {
        eqf.open(eqvalsout);
        eqf << "# equation residuals written by finiteJ_probe --dump-eqvals\n";
        eqf << "version 1\nntheta " << ntheta << "\ndtop " << dtop << "\n";
        eqf << std::setprecision(17);
    }
    std::cout << "#   equation      max|E| axis     max|E| interior\n";
    for (const char* ename : Trumpet::finiteJ_eq_names()) {
        if (maxdefs >= 0) break;
        double mx = 0.0, mxi = 0.0;
        for (int d = 0; d <= dtop; d++) {
            const Val_domain& v = syst.give_val_def_scalar_domain(ename, d);
            const int nth = space.get_domain(d)->get_nbr_points()(1);
            Index idx(space.get_domain(d)->get_nbr_points());
            do {
                const double x = v(idx);
                const double a = std::isfinite(x) ? std::fabs(x) : 1e300;
                mx = std::max(mx, a);
                if (idx(1) != 0 && idx(1) != nth - 1)
                    mxi = std::max(mxi, a);
                if (eqf.is_open())
                    eqf << "E " << ename << " " << d << " " << idx(0) << " "
                        << idx(1) << " " << t.pts[d][idx(0)].r << " "
                        << space.get_domain(d)->get_coloc(2)(idx(1)) << " "
                        << x << "\n";
            } while (idx.inc());
        }
        std::cout << "#   " << std::left << std::setw(13) << ename
                  << std::setprecision(4) << std::setw(16) << mx << mxi << "\n";
        // per-domain, so the floor can be LOCALISED rather than attributed to
        // "the discretisation" as a whole
        for (int d = 0; d <= dtop; d++) {
            const Val_domain& v = syst.give_val_def_scalar_domain(ename, d);
            double md = 0.0;
            Index jd(space.get_domain(d)->get_nbr_points());
            do {
                const double x = v(jd);
                md = std::max(md, std::isfinite(x) ? std::fabs(x) : 1e300);
            } while (jd.inc());
            emit(std::string("FJP_") + ename + "_d" + std::to_string(d), md);
        }
        emit(std::string("FJP_") + ename, mx);
        emit(std::string("FJP_int_") + ename, mxi);
        worst = std::max(worst, mx);
        worst_int = std::max(worst_int, mxi);
    }
    // ---- --extra-def names, on the SAME grid and through the SAME writer ----
    // ⚠ Why this exists (round 259): the five terms of ESIG are internal
    // definitions (D0286, D0293, D0303, D0306, D0311, whose sum is D0312 =
    // ESIG).  Registering them as extras and dumping them here means the term
    // split is the EQUATION'S OWN split, evaluated by Kadath's own operators on
    // the same collocation points as the residual -- not a re-derivation in
    // python that would have to be trusted separately.  The sum of the parts is
    // then checkable against the whole, pointwise, from one run.
    // ⚠ Round 236: every extra name is echoed with its max, so a name that
    // resolved to nothing is an error and not a clean zero.
    if (eqf.is_open())
        for (const auto& nm : extra_names) {
            double mx = 0.0;
            for (int d = 0; d <= dtop; d++) {
                const Val_domain& v =
                    syst.give_val_def_scalar_domain(nm.c_str(), d);
                Index idx(space.get_domain(d)->get_nbr_points());
                do {
                    const double x = v(idx);
                    mx = std::max(mx, std::isfinite(x) ? std::fabs(x) : 1e300);
                    eqf << "D " << nm << " " << d << " " << idx(0) << " "
                        << idx(1) << " " << t.pts[d][idx(0)].r << " "
                        << space.get_domain(d)->get_coloc(2)(idx(1)) << " "
                        << x << "\n";
                } while (idx.inc());
            }
            std::cout << "#   --dump-eqvals extra " << std::left
                      << std::setw(10) << nm << " max|D| = "
                      << std::setprecision(6) << mx << "\n";
        }

    emit("FJP_worst_interior", worst_int);
    emit("FJP_worst", worst);

    // ---- A1: did the manufactured fields actually go IN? -------------------
    // ⚠ Separates "the data never arrived" from "the formulas disagree".  Read
    // back through the same operator()(Index) that forces configuration space,
    // because FETCHING IS NOT READING (round 108).
    if (!mandata.empty()) {
        emit("FJP_man_installed", static_cast<double>(man_installed));
        emit("FJP_man_missing", static_cast<double>(man_missing));
        const char* fnm[6] = {"PS", "PH", "QF", "BR", "BT", "QB"};
        Scalar* fp[6] = {&PS, &PH, &QF, &BR, &BT, &QB};
        double fworst = 0.0;
        for (int q = 0; q < 6; q++) {
            double mx = 0.0;
            for (int d = 0; d <= dtop; d++) {
                const Val_domain& v = (*fp[q])(d);
                Index idx(space.get_domain(d)->get_nbr_points());
                do {
                    auto it = mandata.find(static_cast<long long>(d) * 100000LL
                                           + static_cast<long long>(idx(0)) * 1000LL
                                           + idx(1));
                    if (it == mandata.end() || it->second.size() < 6) continue;
                    const double want = it->second[q];
                    if (!std::isfinite(want)) continue;
                    mx = std::max(mx, std::fabs(v(idx) - want));
                } while (idx.inc());
            }
            emit(std::string("FJP_manfield_") + fnm[q], mx);
            fworst = std::max(fworst, mx);
        }
        emit("FJP_manfield_worst", fworst);
    }

    // ---- A1: Kadath's residual against the exact manufactured source -------
    if (!mandata.empty() && maxdefs < 0) {
        std::cout << "#\n#   A1: |E_kadath - E_exact| on the manufactured "
                     "fields\n";
        std::cout << "#   equation      max abs          max abs (interior)   "
                     "points\n";
        double wa = 0.0, wi = 0.0;
        for (size_t q = 0; q < maneqs.size(); q++) {
            const std::string& ename = maneqs[q];
            double mx = 0.0, mxi = 0.0;
            long npt = 0, nskip = 0;
            for (int d = 0; d <= dtop; d++) {
                const Val_domain& v =
                    syst.give_val_def_scalar_domain(ename.c_str(), d);
                const int nth = space.get_domain(d)->get_nbr_points()(1);
                Index idx(space.get_domain(d)->get_nbr_points());
                do {
                    auto it = mandata.find(static_cast<long long>(d) * 100000LL
                                           + static_cast<long long>(idx(0)) * 1000LL
                                           + idx(1));
                    if (it == mandata.end()
                        || it->second.size() < manfields.size() + maneqs.size()) {
                        nskip++;
                        continue;
                    }
                    const double ex = it->second[manfields.size() + q];
                    const double got = v(idx);
                    // ⚠ a source the generator recorded as non-finite is the
                    // emitted system being singular at that point, not a
                    // missing datum: counted, never silently averaged in.
                    if (!std::isfinite(ex) || !std::isfinite(got)) { nskip++; continue; }
                    const double a = std::fabs(got - ex);
                    mx = std::max(mx, a);
                    if (idx(1) != 0 && idx(1) != nth - 1) mxi = std::max(mxi, a);
                    npt++;
                } while (idx.inc());
            }
            std::cout << "#   " << std::left << std::setw(13) << ename
                      << std::setprecision(6) << std::setw(17) << mx
                      << std::setw(21) << mxi
                      << npt << " used, " << nskip << " skipped\n";
            for (int d = 0; d <= dtop; d++) {
                const Val_domain& v =
                    syst.give_val_def_scalar_domain(ename.c_str(), d);
                double md = 0.0;
                Index jd(space.get_domain(d)->get_nbr_points());
                do {
                    auto it = mandata.find(static_cast<long long>(d) * 100000LL
                                           + static_cast<long long>(jd(0)) * 1000LL
                                           + jd(1));
                    if (it == mandata.end()
                        || it->second.size() < manfields.size() + maneqs.size())
                        continue;
                    const double ex = it->second[manfields.size() + q];
                    const double got = v(jd);
                    if (!std::isfinite(ex) || !std::isfinite(got)) continue;
                    md = std::max(md, std::fabs(got - ex));
                } while (jd.inc());
                emit(std::string("FJP_man_") + ename + "_d" + std::to_string(d), md);
            }
            emit(std::string("FJP_man_") + ename, mx);
            emit(std::string("FJP_manint_") + ename, mxi);
            emit(std::string("FJP_manskip_") + ename, static_cast<double>(nskip));
            wa = std::max(wa, mx);
            wi = std::max(wi, mxi);
        }
        emit("FJP_man_worst", wa);
        emit("FJP_man_worst_interior", wi);
    }

    MPI_Finalize();
    return 0;
}
