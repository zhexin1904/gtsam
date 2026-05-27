/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    CertifiablePoseGraph.cpp
 * @brief   Certifiable SE(d) synchronization (d = 3) via the Riemannian
 *          Staircase, using the SE-Sync translation-explicit cost:
 *            F = Σ_ij κ_ij ‖R_i R_ij − R_j‖_F² + τ_ij ‖t_j − t_i − R_i t_ij‖²
 *          The per-pose rotation R_i is one QCQP variable (3 × D row-orthonormal
 *          matrix); the per-pose translation t_i is another (1 × D row).
 *          Sub-keys are derived from the user's pose key via bit-mask.
 * @author  Zhexin Xu
 *
 * Usage:
 *   ./CertifiablePoseGraph --data=<3D g2o file>
 */

#include <gtsam/constrained/QcqpProblem.h>
#include <gtsam/constrained/QpCost.h>
#include <gtsam/constrained/QuadraticConstraint.h>
#include <gtsam/constrained/RiemannianStaircaseOptimizer.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>

#include <Eigen/SparseCholesky>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>

using namespace gtsam;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Bit-mask key derivation: separates a pose into a rotation sub-key and a
// translation sub-key without touching the user's Symbol indexing. Symbol
// keys fit comfortably under 2^60, so bits 61 and 62 are free.
constexpr Key kRotBit   = Key(1) << 62;
constexpr Key kTransBit = Key(1) << 61;

inline Key RotKeyOf(Key poseKey)   { return poseKey | kRotBit; }
inline Key TransKeyOf(Key poseKey) { return poseKey | kTransBit; }

/// SE-Sync translation-explicit pose factor.
///
/// Emits a single 8×8 QpCost coupling four sub-keys (t_i, t_j, R_i, R_j) and
/// the SO(3) row-orthonormality constraints on each rotation sub-key. Q is
/// built from the per-edge M-matrix derivation (`benchmark/phase2_pgo_sketch.md`):
///   Q = κ·B_rot^T·B_rot + τ·B_trans^T·B_trans
/// where:
///   B_rot   = [0, 0, -R_ij^T, I_3]  (3 × 8)
///   B_trans = [-1, +1, -t_ij^T, 0_3] (1 × 8)
///
/// The factor inherits BetweenFactor<Pose3>'s evaluateError / linearize so the
/// graph behaves normally outside the QCQP path.
class CertifiableSEdBetweenFactor : public BetweenFactor<Pose3> {
 public:
  using BetweenFactor<Pose3>::BetweenFactor;

  void qcqpFactors(NonlinearFactorGraph* costs,
                   NonlinearEqualityConstraints* constraints,
                   size_t K) const override {
    constexpr int d = 3;
    if (K < d) {
      throw std::invalid_argument(
          "CertifiableSEdBetweenFactor::qcqpFactors: K must be >= 3.");
    }
    if (!costs) {
      throw std::invalid_argument(
          "CertifiableSEdBetweenFactor::qcqpFactors: costs is null.");
    }

    // Extract κ (rotation precision) and τ (translation precision) from the
    // 6×6 information matrix. SE-Sync convention:
    //   κ = 3 / (2 * trace(RotInfo^-1))
    //   τ = 3 / trace(TranInfo^-1)
    const auto gaussian =
        std::dynamic_pointer_cast<noiseModel::Gaussian>(this->noiseModel());
    const Matrix6 info =
        gaussian ? gaussian->information() : Matrix6::Identity();
    const double kappa = 3.0 / (2.0 * info.topLeftCorner<3, 3>().inverse().trace());
    const double tau   = 3.0 / info.bottomRightCorner<3, 3>().inverse().trace();

    // Sub-keys for the two endpoints of this edge.
    const Key rot_i   = RotKeyOf(this->key1());
    const Key rot_j   = RotKeyOf(this->key2());
    const Key trans_i = TransKeyOf(this->key1());
    const Key trans_j = TransKeyOf(this->key2());

    // Rotation orthogonality constraints — one set per pose.
    InsertQcqpConstraints<Rot3, d>(rot_i, constraints);
    InsertQcqpConstraints<Rot3, d>(rot_j, constraints);

    // Build the 8×8 Q matrix.
    // Block layout: [t_i (1), t_j (1), R_i (3), R_j (3)]
    const Matrix3 R_ij = this->measured().rotation().matrix();
    const Vector3 t_ij = this->measured().translation();

    // B = [√κ · B_rot ;  √τ · B_trans]  → (4 × 8)
    // Q = B^T · B
    Matrix B = Matrix::Zero(d + 1, 2 + 2 * d);
    const double sqrt_kappa = std::sqrt(kappa);
    const double sqrt_tau   = std::sqrt(tau);

    // B_rot rows (3 rows): -R_ij^T on cols [2..4], +I_3 on cols [5..7].
    for (int r = 0; r < d; ++r) {
      for (int c = 0; c < d; ++c) {
        B(r, 2 + c) = -sqrt_kappa * R_ij(c, r);   // (-R_ij^T)[r,c] = -R_ij[c,r]
      }
      B(r, 2 + d + r) = sqrt_kappa;               // I_3
    }
    // B_trans row (1 row): -1 on col 0, +1 on col 1, -t_ij^T on cols [2..4].
    B(d, 0) = -sqrt_tau;
    B(d, 1) =  sqrt_tau;
    for (int c = 0; c < d; ++c) {
      B(d, 2 + c) = -sqrt_tau * t_ij(c);
    }

    const Matrix Q = B.transpose() * B;
    const SymmetricBlockMatrix blockQ(
        std::vector<DenseIndex>{1, 1, d, d}, Q);
    costs->push_back(std::make_shared<QpCost>(
        KeyVector{trans_i, trans_j, rot_i, rot_j}, blockQ, K));
  }
};

/// Random Pose3 initialization stored at column count D (>= 3).
/// Rotation sub-key gets a (3 × D) row-orthonormal block via the Rot3 QCQP
/// trait; translation sub-key gets a (1 × D) row with the leading 3 columns
/// drawn uniformly from [-translationScale, translationScale].
Values RandomInitial(const std::set<Key>& poseKeys, int D,
                     unsigned int seed,
                     double translationScale = 1.0) {
  Values v;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uni_angle(-kPi, kPi);
  std::uniform_real_distribution<double> uni_trans(-translationScale,
                                                    translationScale);
  for (Key poseKey : poseKeys) {
    // Rotation: random RPY → Rot3 → row-orthonormal (3 × D).
    const Vector3 rpy(uni_angle(rng), uni_angle(rng), uni_angle(rng));
    const Rot3 R = Rot3::RzRyRx(rpy(0), rpy(1), rpy(2));
    InsertQcqpValue<Rot3, 3>(RotKeyOf(poseKey), R, &v);

    // Translation: 1 × D row, leading 3 columns are random t, rest are zero.
    Matrix trow = Matrix::Zero(1, D);
    trow(0, 0) = uni_trans(rng);
    trow(0, 1) = uni_trans(rng);
    trow(0, 2) = uni_trans(rng);
    v.insert(TransKeyOf(poseKey), trow);
  }
  return v;
}

/// Compute SE-Sync's explicit-form cost on a Pose3 solution:
///   F = Σ_ij κ_ij ‖R_i R_ij − R_j‖_F² + τ_ij ‖t_j − t_i − R_i t_ij‖²
/// Iterates the graph (which contains CertifiableSEdBetweenFactor instances)
/// and re-extracts (κ, τ, R_ij, t_ij) from each factor's noise model.
/// Returns (rotation_cost, translation_cost, total).
struct ExplicitCost {
  double rotation;
  double translation;
  double total() const { return rotation + translation; }
};
ExplicitCost EvaluateExplicitCost(const NonlinearFactorGraph& graph,
                                  const Values& poses) {
  ExplicitCost out{0.0, 0.0};
  for (const auto& f : graph) {
    const auto bf = std::dynamic_pointer_cast<BetweenFactor<Pose3>>(f);
    if (!bf) continue;
    if (!poses.exists(bf->key1()) || !poses.exists(bf->key2())) continue;

    const auto gaussian =
        std::dynamic_pointer_cast<noiseModel::Gaussian>(bf->noiseModel());
    const Matrix6 info =
        gaussian ? gaussian->information() : Matrix6::Identity();
    const double kappa = 3.0 / (2.0 * info.topLeftCorner<3, 3>().inverse().trace());
    const double tau   = 3.0 / info.bottomRightCorner<3, 3>().inverse().trace();

    const Pose3 P_i = poses.at<Pose3>(bf->key1());
    const Pose3 P_j = poses.at<Pose3>(bf->key2());
    const Matrix3 R_i = P_i.rotation().matrix();
    const Matrix3 R_j = P_j.rotation().matrix();
    const Vector3 t_i = P_i.translation();
    const Vector3 t_j = P_j.translation();
    const Matrix3 R_ij = bf->measured().rotation().matrix();
    const Vector3 t_ij = bf->measured().translation();

    out.rotation    += kappa * (R_i * R_ij - R_j).squaredNorm();
    out.translation += tau * (t_j - t_i - R_i * t_ij).squaredNorm();
  }
  return out;
}

/// Project Y_rot block of Y to rank d using rotation-only SVD basis V_rot.
/// Returns a Values where each rotation entry is (3 × d) — translations
/// untouched. This avoids contamination by the translation rows in the joint
/// SVD basis (which can be dominated by translation magnitude).
Values ProjectRotationsViaRotSVD(
    const Values& Yp,
    const RiemannianStaircaseOptimizer::Layout& layout,
    int d) {
  // Gather rotation rows into a tall matrix Y_rot.
  size_t nRot = 0;
  for (const auto& [k, sl] : layout.slices) if (sl.rowDim == 3) nRot += 3;
  const Matrix Yfull = layout.stack(Yp);
  Matrix Yrot(nRot, Yfull.cols());
  std::vector<std::pair<Key, size_t>> rotKeyRow;  // key -> starting row in Yrot
  rotKeyRow.reserve(nRot / 3);
  size_t rOff = 0;
  for (const auto& [k, sl] : layout.slices) {
    if (sl.rowDim != 3) continue;
    Yrot.block(rOff, 0, 3, Yfull.cols()) =
        Yfull.block(sl.offset, 0, 3, Yfull.cols());
    rotKeyRow.emplace_back(k, rOff);
    rOff += 3;
  }

  Eigen::JacobiSVD<Matrix> svd(Yrot, Eigen::ComputeFullV);
  // V is (p × p); take leading d columns.
  const Matrix Vd = svd.matrixV().leftCols(d);
  Matrix YrotD = Yrot * Vd;  // (nRot × d)

  // Determine majority det sign and flip last column of YrotD if needed.
  // This sets the consistent gauge BEFORE per-block ClosestTo.
  size_t numNeg = 0, numBlocks = 0;
  for (const auto& [k, row] : rotKeyRow) {
    if (YrotD.block(row, 0, 3, d).leftCols<3>().determinant() < 0) ++numNeg;
    ++numBlocks;
  }
  if (numBlocks > 0 && numNeg > numBlocks / 2) {
    YrotD.col(d - 1) *= -1.0;
  }

  Values out = Yp;
  for (const auto& [k, row] : rotKeyRow) {
    out.update(k, Matrix(YrotD.block(row, 0, 3, d)));
  }
  return out;
}

/// Round the staircase output back to typed Pose3s.
///
/// Gauge alignment for PGO: SVD truncation produces Yd = Y · V where V is
/// orthogonal (gauge-equivalent in BM). For mixed rotation+translation,
/// the per-block ClosestTo's internal det-correction would flip rotations
/// but leave translations alone — breaking the gauge consistency between
/// them. We fix this by checking the majority det sign of rotation blocks
/// and, if negative, flipping column (D-1) of *both* rotation and
/// translation entries so the whole gauge moves together.
std::vector<std::pair<Key, Pose3>> ExtractPoses(const Values& atRankD) {
  Values fixed = atRankD;

  // Decide whether to flip based on rotation-block determinants.
  constexpr int D = 3;
  size_t numNeg = 0, numBlocks = 0;
  for (const auto& [key, M] : fixed.extract<Matrix>()) {
    if (M.rows() != 3 || M.cols() < 3) continue;
    ++numBlocks;
    if (M.leftCols<3>().determinant() < 0) ++numNeg;
  }
  if (numBlocks > 0 && numNeg > numBlocks / 2) {
    // Flip column D-1 of EVERY matrix entry (rotation + translation), so
    // the gauge transformation is uniform across all variable types.
    for (const auto& [key, M] : fixed.extract<Matrix>()) {
      Matrix flipped = M;
      flipped.col(D - 1) *= -1.0;
      fixed.update(key, flipped);
    }
  }

  std::vector<std::pair<Key, Pose3>> poses;
  for (const auto& [key, M] : fixed.extract<Matrix>()) {
    if ((key & kRotBit) == 0) continue;            // not a rotation entry
    const Key poseKey = key & ~(kRotBit | kTransBit);
    const Rot3 R = traits<Rot3>::FromQcqpValue<3>(M);
    const Key trans_key = TransKeyOf(poseKey);
    if (!fixed.exists(trans_key)) continue;
    const Matrix& Trow = fixed.at<Matrix>(trans_key);
    const Vector3 t = Trow.row(0).leftCols<3>().transpose();
    poses.emplace_back(poseKey, Pose3(R, t));
  }
  return poses;
}

/// Re-solve translations from rounded rotations via a sparse linear LS.
/// For fixed R_i, F_trans(t) = Σ τ_ij ‖t_j − t_i − R_i t_ij‖² is convex
/// quadratic; the minimizer is the LS solution of the per-edge residual
///   √τ_ij · (t_j − t_i) = √τ_ij · R_i · t_ij.
/// We anchor pose 0 at the origin to fix the translation gauge.
/// (This is the standard SE-Sync post-rounding step in the explicit form.)
std::vector<std::pair<Key, Pose3>> RefineTranslations(
    const std::vector<std::pair<Key, Pose3>>& roundedPoses,
    const NonlinearFactorGraph& graph) {
  // Index poses 0..n-1, in the order roundedPoses was emitted.
  std::unordered_map<Key, int> idx;
  idx.reserve(roundedPoses.size());
  for (int i = 0; i < (int)roundedPoses.size(); ++i) {
    idx[roundedPoses[i].first] = i;
  }
  const int n = (int)roundedPoses.size();
  if (n == 0) return roundedPoses;

  // Build sparse A and dense b for ‖A t − b‖² where t ∈ R^{3n}.
  // Per edge (i, j): rows = [3·edge_idx .. 3·edge_idx + 2] contribute
  //   √τ · (- I_3 at t_i  ;  + I_3 at t_j)  with RHS  √τ · R_i · t_ij.
  // Pose 0 anchored to origin via an extra ‖t_0‖² penalty row.
  std::vector<Eigen::Triplet<double>> trips;
  std::vector<Vector3> rhs;
  trips.reserve(graph.size() * 6 + 3);
  rhs.reserve(graph.size() + 1);

  int edgeRow = 0;
  for (const auto& f : graph) {
    const auto bf = std::dynamic_pointer_cast<BetweenFactor<Pose3>>(f);
    if (!bf) continue;
    auto it_i = idx.find(bf->key1());
    auto it_j = idx.find(bf->key2());
    if (it_i == idx.end() || it_j == idx.end()) continue;
    const int i = it_i->second, j = it_j->second;
    const auto gaussian =
        std::dynamic_pointer_cast<noiseModel::Gaussian>(bf->noiseModel());
    const Matrix6 info =
        gaussian ? gaussian->information() : Matrix6::Identity();
    const double tau =
        3.0 / info.bottomRightCorner<3, 3>().inverse().trace();
    const double sqrt_tau = std::sqrt(tau);
    const Matrix3 R_i = roundedPoses[i].second.rotation().matrix();
    const Vector3 t_ij = bf->measured().translation();
    for (int c = 0; c < 3; ++c) {
      trips.emplace_back(edgeRow + c, 3 * i + c, -sqrt_tau);
      trips.emplace_back(edgeRow + c, 3 * j + c, +sqrt_tau);
    }
    rhs.push_back(sqrt_tau * R_i * t_ij);
    edgeRow += 3;
  }
  // Gauge anchor: t_0 = 0 with large weight.
  const double anchorWeight = 1e6;
  for (int c = 0; c < 3; ++c) {
    trips.emplace_back(edgeRow + c, c, anchorWeight);
  }
  rhs.push_back(Vector3::Zero());
  edgeRow += 3;

  Eigen::SparseMatrix<double> A(edgeRow, 3 * n);
  A.setFromTriplets(trips.begin(), trips.end());
  Vector b(edgeRow);
  for (int r = 0; r < (int)rhs.size(); ++r) {
    b.segment<3>(3 * r) = rhs[r];
  }

  Eigen::SparseMatrix<double> AtA = A.transpose() * A;
  Vector Atb = A.transpose() * b;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(AtA);
  if (solver.info() != Eigen::Success) {
    return roundedPoses;  // factorization failed, return unrefined
  }
  Vector tOpt = solver.solve(Atb);

  std::vector<std::pair<Key, Pose3>> out;
  out.reserve(n);
  for (int i = 0; i < n; ++i) {
    out.emplace_back(
        roundedPoses[i].first,
        Pose3(roundedPoses[i].second.rotation(), tOpt.segment<3>(3 * i)));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dataPath;
  unsigned int seed = 60;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--data=", 0) == 0) {
      dataPath = arg.substr(7);
    } else if (arg.rfind("--seed=", 0) == 0) {
      seed = static_cast<unsigned int>(std::stoul(arg.substr(7)));
    }
  }
  if (dataPath.empty()) {
    std::cerr << "Usage: " << argv[0] << " --data=<3D g2o file> [--seed=N]\n";
    return 2;
  }

  // Load g2o (3D pose graph), rebuild as a CertifiableSEdBetweenFactor graph.
  auto [poseGraph, _] = readG2o(dataPath, /*is3D=*/true);
  NonlinearFactorGraph graph;
  std::set<Key> poseKeys;
  size_t skipped = 0;
  for (const auto& f : *poseGraph) {
    const auto bf = std::dynamic_pointer_cast<BetweenFactor<Pose3>>(f);
    if (!bf) { ++skipped; continue; }
    graph.emplace_shared<CertifiableSEdBetweenFactor>(
        bf->key1(), bf->key2(), bf->measured(), bf->noiseModel());
    poseKeys.insert(bf->key1());
    poseKeys.insert(bf->key2());
  }
  if (graph.empty()) {
    std::cerr << "No BetweenFactor<Pose3> found in " << dataPath << "\n";
    return 1;
  }
  std::cout << "Loaded " << poseKeys.size() << " poses, " << graph.size()
            << " edges from " << dataPath;
  if (skipped > 0) std::cout << "  (" << skipped << " non-Between skipped)";
  std::cout << "\n";

  // DIAGNOSTIC: kappa, tau for first edge.
  {
    const auto bf =
        std::dynamic_pointer_cast<BetweenFactor<Pose3>>(graph.at(0));
    const auto gaussian =
        std::dynamic_pointer_cast<noiseModel::Gaussian>(bf->noiseModel());
    const Matrix6 info =
        gaussian ? gaussian->information() : Matrix6::Identity();
    const double kappa = 3.0 / (2.0 * info.topLeftCorner<3,3>().inverse().trace());
    const double tau   = 3.0 / info.bottomRightCorner<3,3>().inverse().trace();
    std::cout << "First-edge kappa=" << kappa << " tau=" << tau << "\n";
    std::cout << "First-edge info(0,0)=" << info(0,0)
              << " info(3,3)=" << info(3,3) << "\n\n";
  }

  // Random init at column count D = 3.
  constexpr int D = 3;
  std::cout << "Seed: " << seed << "\n";
  Values initial = RandomInitial(poseKeys, D, seed, /*translationScale=*/1.0);

  // Staircase params — same regime as CRA.
  RiemannianStaircaseParams params;
  params.pMin = D;
  params.verbose = true;
  params.almParams->initialMuEq = 1.0;
  params.almParams->muEqIncreaseRate = 2.0;
  params.almParams->maxIterations = 200;
  params.almParams->absoluteViolationTolerance = 1e-12;
  params.almParams->relativeViolationTolerance = 1e-12;
  params.almParams->absoluteCostTolerance = 1e-14;
  params.almParams->relativeCostTolerance = 1e-14;
  params.almParams->verbose = false;

  // DIAGNOSTIC: F_explicit on the random initial values.
  {
    // Reconstruct Pose3s from initial Values.
    Values initPoses;
    for (Key poseKey : poseKeys) {
      const Matrix& Rmat = initial.at<Matrix>(RotKeyOf(poseKey));
      const Matrix& Trow = initial.at<Matrix>(TransKeyOf(poseKey));
      Rot3 R = traits<Rot3>::FromQcqpValue<3>(Rmat);
      Vector3 t = Trow.row(0).leftCols<3>().transpose();
      initPoses.insert(poseKey, Pose3(R, t));
    }
    const ExplicitCost initCost = EvaluateExplicitCost(graph, initPoses);
    std::cout << "Explicit F (init random): rotation=" << initCost.rotation
              << ", translation=" << initCost.translation
              << ", total=" << initCost.total() << "\n\n";
  }

  RiemannianStaircaseOptimizer rso(graph, initial, params);

  const auto t0 = std::chrono::steady_clock::now();
  const auto result = rso.optimize();
  const double wallSeconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - t0).count();

  // DIAGNOSTIC: rebuild Q and S(lambda) for the converged iterate and check
  // the KKT complementarity ‖S Y‖_F. If Y is truly first-order KKT for BM,
  // this must be zero (within numerical tolerance). A large value means the
  // cert's λ is not the true KKT multiplier and the verified=1 is a false
  // positive.
  if (result.certified) {
    const QcqpProblem qcqpFinal(graph, result.finalRank);
    const auto& cpl = result.minEigenvaluePerLevel;
    (void)cpl;
    // We need the same lambdaEq used to build the cert; re-run inner solver
    // from the converged Y to recover it.
    const auto innerFinal = RiemannianStaircaseOptimizer::runLocalSolver(
        qcqpFinal, result.values, params.almParams);
    const auto layout = RiemannianStaircaseOptimizer::Layout::From(innerFinal.Y);
    Eigen::SparseMatrix<double> S =
        RiemannianStaircaseOptimizer::buildCertificate(
            qcqpFinal, layout, innerFinal.lambdaEq);
    const Matrix Ymat = layout.stack(innerFinal.Y);
    const Matrix SY = S * Ymat;
    std::cout << "‖S·Y‖_F = " << SY.norm()
              << "    (KKT complementarity residual)\n";
  }

  // DIAGNOSTIC: maximum constraint violation at the converged BM solution.
  // Theorem 1(a) requires h(Y) = 0 exactly. ALM enforces softly, so if
  // violations are non-trivial here, the extracted lambda is biased and the
  // cert is unreliable.
  if (result.certified) {
    QcqpProblem qcqpAtFinalRank(graph, result.finalRank);
    double maxViolation = 0.0;
    for (const auto& f : qcqpAtFinalRank.eConstraints()) {
      Vector e = f->whitenedError(result.values);
      maxViolation = std::max(maxViolation, e.cwiseAbs().maxCoeff());
    }
    std::cout << "Max constraint violation |h(Y)|_∞ = " << maxViolation << "\n";
  }

  // DIAGNOSTIC: singular values of converged Y, split by row type.
  if (result.certified) {
    const Matrix Yfull = result.layout.stack(result.values);
    Eigen::JacobiSVD<Matrix> svdAll(Yfull);
    std::cout << "Y_full singular values: ";
    for (int i = 0; i < svdAll.singularValues().size(); ++i)
      std::cout << svdAll.singularValues()(i) << " ";
    std::cout << "\n";

    // Split into rotation rows (rowDim=3) and translation rows (rowDim=1).
    size_t nRot = 0, nTrans = 0;
    for (const auto& [key, slice] : result.layout.slices) {
      if (slice.rowDim == 3) nRot += 3;
      else if (slice.rowDim == 1) nTrans += 1;
    }
    Matrix Yrot(nRot, Yfull.cols()), Ytrans(nTrans, Yfull.cols());
    size_t rOff = 0, tOff = 0;
    for (const auto& [key, slice] : result.layout.slices) {
      if (slice.rowDim == 3) {
        Yrot.block(rOff, 0, 3, Yfull.cols()) =
            Yfull.block(slice.offset, 0, 3, Yfull.cols());
        rOff += 3;
      } else if (slice.rowDim == 1) {
        Ytrans.row(tOff) = Yfull.row(slice.offset);
        ++tOff;
      }
    }
    Eigen::JacobiSVD<Matrix> svdR(Yrot);
    Eigen::JacobiSVD<Matrix> svdT(Ytrans);
    std::cout << "Y_rot   singular values: ";
    for (int i = 0; i < svdR.singularValues().size(); ++i)
      std::cout << svdR.singularValues()(i) << " ";
    std::cout << "\n";
    std::cout << "Y_trans singular values: ";
    for (int i = 0; i < svdT.singularValues().size(); ++i)
      std::cout << svdT.singularValues()(i) << " ";
    std::cout << "\n";
  }

  Values rounded;          // pose set with t from SVD truncation
  Values roundedRefined;   // pose set with t from post-rounding linear LS
  Values roundedRotSVD;    // rotation projected via rotation-only SVD + t LS
  if (result.rounded) {
    Values atRankD = result.layout.unstack(result.rounded->Yd);
    auto roundedPoses = ExtractPoses(atRankD);
    for (auto& [poseKey, pose] : roundedPoses) {
      rounded.insert(poseKey, pose);
    }
    auto refined = RefineTranslations(roundedPoses, graph);
    for (auto& [poseKey, pose] : refined) {
      roundedRefined.insert(poseKey, pose);
    }

    // Alternative rounding: project rotations using rotation-only SVD basis.
    constexpr int d = 3;
    Values rotSVDValues =
        ProjectRotationsViaRotSVD(result.values, result.layout, d);
    // Per-block ClosestTo on the rotation entries.
    auto rotsOnly = ExtractPoses(rotSVDValues);
    // The translations from rotSVDValues are still rank-p; we discard them
    // and re-solve via LS.
    auto refinedRotSVD = RefineTranslations(rotsOnly, graph);
    for (auto& [poseKey, pose] : refinedRotSVD) {
      roundedRotSVD.insert(poseKey, pose);
    }
  }

  std::cout << "\n========== Result ==========\n"
            << "Certified:         " << (result.certified ? "yes" : "no") << "\n"
            << "Final rank:        " << result.finalRank << "\n"
            << "Cold-start BM ×2:  " << (2.0 * (result.costPerLevel.empty()
                                              ? 0.0
                                              : result.costPerLevel.back()))
            << "\n";
  if (rounded.empty()) {
    std::cout << "Explicit F (rounded): n/a\n";
  } else {
    const ExplicitCost cost = EvaluateExplicitCost(graph, rounded);
    std::cout << "Explicit F (rounded, SVD t): rotation=" << cost.rotation
              << ", translation=" << cost.translation
              << ", total=" << cost.total() << "\n";
    const ExplicitCost refinedCost =
        EvaluateExplicitCost(graph, roundedRefined);
    std::cout << "Explicit F (rounded, LS t):  rotation=" << refinedCost.rotation
              << ", translation=" << refinedCost.translation
              << ", total=" << refinedCost.total() << "\n";
    const ExplicitCost rotSVDCost =
        EvaluateExplicitCost(graph, roundedRotSVD);
    std::cout << "Explicit F (R-SVD + LS t):   rotation=" << rotSVDCost.rotation
              << ", translation=" << rotSVDCost.translation
              << ", total=" << rotSVDCost.total() << "\n";

    // Local refinement: polish the LS-refined rounded poses with a few GN
    // steps on the BetweenFactor<Pose3> cost (logmap residual, not F itself).
    // Add a tight prior on the first pose so the gauge is fixed.
    NonlinearFactorGraph polishGraph = graph;
    const Key firstPoseKey = *poseKeys.begin();
    polishGraph.emplace_shared<PriorFactor<Pose3>>(
        firstPoseKey, roundedRefined.at<Pose3>(firstPoseKey),
        noiseModel::Isotropic::Sigma(6, 1e-6));
    GaussNewtonParams gnParams;
    gnParams.maxIterations = 20;
    gnParams.relativeErrorTol = 1e-10;
    GaussNewtonOptimizer gn(polishGraph, roundedRefined, gnParams);
    const Values polished = gn.optimize();
    const ExplicitCost polishedCost = EvaluateExplicitCost(graph, polished);
    std::cout << "Explicit F (LS t + GN):      rotation="
              << polishedCost.rotation
              << ", translation=" << polishedCost.translation
              << ", total=" << polishedCost.total() << "\n";

    // TIGHTNESS CHECK: rebuild a rank-3 initial Values from the rounded
    // (LS-refined) Pose3s and re-run the staircase. If this certifies at
    // rank 3 with cost ≈ BM-at-rank-4, the joint SDP IS tight at rank 3 and
    // our original BM merely got stuck at a bad rank-3 local minimum.
    {
      Values warmInit;
      for (const auto& [poseKey, pose] : roundedRefined.extract<Pose3>()) {
        InsertQcqpValue<Rot3, 3>(RotKeyOf(poseKey), pose.rotation(), &warmInit);
        Matrix trow = Matrix::Zero(1, 3);
        trow.row(0) = pose.translation().transpose();
        warmInit.insert(TransKeyOf(poseKey), trow);
      }
      RiemannianStaircaseParams warmParams = params;
      warmParams.pMin = 3;
      warmParams.verbose = false;
      RiemannianStaircaseOptimizer warmRso(graph, warmInit, warmParams);
      const auto warmRes = warmRso.optimize();
      std::cout << "[Tightness check] Warm-start rank-3 staircase: "
                << "certified=" << warmRes.certified
                << " finalRank=" << warmRes.finalRank
                << " BM cost ×2="
                << (2.0 * (warmRes.costPerLevel.empty()
                              ? 0.0
                              : warmRes.costPerLevel.back()))
                << "\n";
      // Round the warm-start solution and evaluate F on real Pose3s.
      if (warmRes.rounded && warmRes.finalRank == 3) {
        Values warmAtD = warmRes.layout.unstack(warmRes.rounded->Yd);
        auto warmPoses = ExtractPoses(warmAtD);
        auto warmRefined = RefineTranslations(warmPoses, graph);
        Values warmValues;
        for (auto& [k, p] : warmRefined) warmValues.insert(k, p);
        const ExplicitCost warmCost =
            EvaluateExplicitCost(graph, warmValues);
        std::cout << "Explicit F (warm-start rank-3 + LS t):  rotation="
                  << warmCost.rotation << ", translation="
                  << warmCost.translation
                  << ", total=" << warmCost.total() << "\n";
      }
    }
  }
  std::cout << "Staircase BM cost (×2 = explicit F): "
            << (2.0 * (result.costPerLevel.empty()
                          ? 0.0
                          : result.costPerLevel.back())) << "\n"
            << "Wall time:         " << wallSeconds << " s\n";

  return result.certified ? 0 : 1;
}
