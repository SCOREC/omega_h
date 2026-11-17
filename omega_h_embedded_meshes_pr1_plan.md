# Omega_h: Embedded 2D-in-3D Meshes — Implementation Plan (PR 1)

Agent handoff, 2026-11-16. Target: SCOREC/omega_h `master` (version 11.4.0 at time of writing).

## Scope

PR 1 lets an Omega_h mesh carry a spatial dimension larger than its topological dimension (triangles and edges in 3D), preserves existing non-embedded computations, and makes every unported code path fail loudly on an embedded mesh. All new osh outputs use version 12; older Omega_h readers cannot read these new files. Older osh inputs remain supported.

Goals:

- `Mesh::space_dim()`, defaulting to `dim()`, stored, copied, migrated and written to osh files.
- A coordinate view, `CoordsView<N>`, with mdspan-style indexing and a host-side check of `N` against `space_dim()`.
- `Mesh::coords()` and `Mesh::set_coords(Reals)` deprecated for downstream codes and forbidden inside Omega_h.
- Coarse gates so adaptation, metrics, quality, recovery, conservation and AMR refuse embedded meshes with a clear error.
- A small ported set (measures, balancing, osh, VTK output) so embedded meshes are usable and testable.

Non-goals: adapting, measuring quality of, or computing metrics on embedded meshes; readers that create embedded meshes (gmsh, VTK, exodus); mixed meshes (`MixedMesh::coords_mix()`); the full audit of `dim()` uses and the bare-`dim` ban; curved meshes.

## Settled decisions

These were agreed in design review and are not open for the implementer to change.

| Topic | Decision |
| --- | --- |
| Names | `dim()` stays topological and public; new `tdim()` alias; new `space_dim()`. Invariant `dim() <= space_dim() <= 3`. |
| Default | `space_dim()` returns `dim()` unless set, so existing meshes, files and call sites behave exactly as today. |
| Raw access | `coords()` and `set_coords(Reals)` stay, deprecated with `[[deprecated]]`, for downstream codes, and explicitly reject embedded meshes. Downstream can silence warnings with `OMEGA_H_SUPPRESS_DEPRECATION`; suppression does not disable gates. |
| File output | Always write osh version 12, including ordinary meshes (confirmed by reviewer). |
| Internal access | Omega_h builds with `-Werror=deprecated-declarations`; internal code never calls the deprecated API. |
| View type | `CoordsView<N>`: rank-2, `coords(v, d)`, static `N` or `dynamic_extent`. Static `N` checked against `space_dim()` once at construction, on the host, in release builds. Both forms are permanent. |
| View backing | An owning wrapper (holds the `Reals`) around an `Omega_h::mdspan`. |
| mdspan source | Kokkos builds require Kokkos 4.4.00 or newer and use its bundled mdspan. Non-Kokkos builds stay supported and use the vendored `kokkos/mdspan` single header with its namespace set to `Kokkos`. Code only spells `Omega_h::mdspan`. |
| Gating | Coarse: explicit gates at public entry points, a backstop in `MetricElementQualities`, and a gated accessor for every internal coordinate read not yet ported. |
| Gate failure | `Omega_h_fail` with a message naming the function and both dimensions. Throws when built with `Omega_h_THROW=ON`. |

## Step 1: space dimension on Mesh

The mesh stores an optional spatial dimension, and every place that creates, copies or serializes a mesh carries it.

**`Omega_h_mesh.hpp` / `Omega_h_mesh.cpp`**

```cpp
Int tdim() const { return dim(); }
Int space_dim() const { return space_dim_ < 0 ? dim() : space_dim_; }
void set_space_dim(Int sdim);   // new
// private:
Int space_dim_ = -1;            // -1 = unset, follow dim()
```

- `set_space_dim` requires `dim_` already set and `dim() <= sdim <= 3`. If a `"coordinates"` tag exists, its component count must equal `sdim`; otherwise fail with a message.
- `copy_meta()` copies `space_dim_`. This is what keeps migration, ghosting and adaptation copies consistent.
- `bcast_mesh()` broadcasts `space_dim()` after `dim()` and initializes it on new ranks before adding tags. Test communicator expansion, including reading a one-part embedded osh on 2 and 4 ranks.
- `add_coords(Reals)` uses `space_dim()` as the component count; `set_coords` checks `size == nverts() * space_dim()`.
- `Omega_h_compare.cpp`: meshes with different `space_dim()` compare unequal, with a message.

**Construction (`Omega_h_build.hpp` / `.cpp`)**

- Keep `build_from_elems_and_coords(mesh, family, edim, ev2v, coords)` unchanged (space dimension = `edim`).
- Add an overload with a trailing `Int sdim`: explicitly validate `1 <= edim <= sdim <= 3` and coordinate-length divisibility before dividing, using release-active failures; then compute `nverts = coords.size() / sdim`, call `build_from_elems2verts`, `set_space_dim(sdim)`, and `add_coords`. Distributed construction may use the lower-level comm-taking builder followed by `set_space_dim` and `add_coords` on all ranks.

**osh files (`Omega_h_file.cpp` / `.hpp`)**

- Bump `latest_version` from 11 to 12.
- `write_meta`: write `I8(space_dim())` immediately after the dimension.
- `read_meta`: when `version >= 12`, read it and call `set_space_dim` after `set_dim`; older files leave it unset.
- After tags are read, check the `"coordinates"` tag's component count equals `space_dim()` and fail with a message otherwise.

Acceptance: the existing test suite passes unchanged; a pre-12 osh file from the existing test data (`meshes/plate_6elem.osh`, version 9) loads with `space_dim() == dim()`; an embedded mesh round-trips through osh with `space_dim() == 3` and identical coordinates.

## Step 2: mdspan source

One header, `Omega_h_mdspan.hpp`, decides where mdspan comes from and exports it as `Omega_h::mdspan`; nothing else in Omega_h includes an mdspan header directly.

Kokkos builds now require Kokkos 4.4.00 or newer, the first release that ships mdspan by default. Non-Kokkos builds remain supported, because we cannot confirm nobody uses them (CI still tests them), and they use the vendored copy.

```cpp
#if defined(OMEGA_H_USE_KOKKOS)
#include <Kokkos_Core.hpp>
#include <mdspan/mdspan.hpp>   // Kokkos's bundled copy; Kokkos >= 4.4 enforced at configure time
#else
#define MDSPAN_IMPL_STANDARD_NAMESPACE Kokkos
#define MDSPAN_IMPL_PROPOSED_NAMESPACE Experimental
#include <mdspan/mdspan.hpp>   // vendored single header from tpl/mdspan
#endif
namespace Omega_h {
using Kokkos::mdspan; using Kokkos::extents; using Kokkos::dextents;
using Kokkos::layout_right; using Kokkos::dynamic_extent;
}
```

- In `CMakeLists.txt`, raise `Kokkos_REQUIRED_VERSION` from 4.0 to 4.4. In `.github/workflows/cmake.yml`, change the job labeled as the oldest compatible Kokkos from `kokkos_version: 4.3.00` to `4.4.00`, or it will fail the new configure check. The effective minimum is already 4.3: that CI job pins it, and `Omega_h_USE_Kokkos_Sort` (default ON) requires it.
- When `Omega_h_USE_Kokkos` is ON, fail at configure time with a clear message if Kokkos was built without mdspan. In Kokkos 4.x that means `Kokkos_ENABLE_IMPL_MDSPAN` is OFF (the macro `KOKKOS_ENABLE_IMPL_MDSPAN` is then absent from Kokkos's config header); in 5.x mdspan is always present.
- The existing checks for Kokkos below 4.1 (CUDA lambda) and below 4.3 (`sort_by_key`) become unreachable. Leave them in PR 1 and list them for later cleanup.
- Vendor the single header from the `single-header` branch of `kokkos/mdspan`, pinned to one commit, at `tpl/mdspan/mdspan/mdspan.hpp`, following the repo's `tpl/pss` and `tpl/random123` convention. Record the commit and license in `tpl/mdspan/README.md` and `tpl/mdspan/LICENSE`.
- Add `tpl/mdspan` to the include path and install it only when `Omega_h_USE_Kokkos` is OFF, so the two copies never meet in one build.
- Release notes: Kokkos builds of Omega_h now need Kokkos 4.4.00 or newer. Note too that in non-Kokkos builds, a downstream translation unit that includes a second mdspan copy will hit the shared `MDSPAN_HPP_` include guard.
- Leave a reserved `OMEGA_H_USE_STD_MDSPAN` branch out of PR 1; it lands when C++23 and device support allow dropping the vendored copy.

Acceptance: Omega_h builds without Kokkos and with Kokkos 4.4 or newer, and `Omega_h::mdspan` resolves in both; configuring against Kokkos below 4.4, or 4.x with mdspan disabled, fails with the message.

## Step 3: CoordsView and Mesh accessors

A new public header, `Omega_h_coords_view.hpp`, defines a rank-2 owning view over vertex coordinates; `Mesh` gains typed accessors built on it.

```cpp
template <class T, std::size_t N>   // T = const Real (read) or Real (write)
class BasicCoordsView {
 public:
  using extents_type = extents<LO, dynamic_extent, N>;
  using mdspan_type  = mdspan<T, extents_type, layout_right>;
  BasicCoordsView(ArrayOf<T> data, Int sdim);           // host only, checked
  OMEGA_H_INLINE T& operator()(LO v, Int d) const;      // debug-checks d < extent(1)
  OMEGA_H_INLINE LO extent(Int r) const;                // 0: nverts, 1: sdim
  OMEGA_H_INLINE mdspan_type span() const;              // for kernels wanting mdspan
  ArrayOf<T> const& raw() const;                        // flat array; lint-restricted
 private:
  ArrayOf<T> data_;   // Reals or Write<Real>: keeps memory alive
  mdspan_type span_;
};
template <std::size_t N = dynamic_extent> using CoordsView      = BasicCoordsView<const Real, N>;
template <std::size_t N = dynamic_extent> using CoordsWriteView = BasicCoordsView<Real, N>;
```

- The constructor checks `1 <= sdim <= 3`, `size % sdim == 0`, and, for static `N`, `N == sdim`. It calls `Omega_h_fail` directly, because `OMEGA_H_CHECK` compiles away under `NDEBUG` in Kokkos builds.
- Static-to-dynamic conversion is implicit; dynamic-to-static is `view.as<N>()` and runs the same check.
- Add `gather_vectors<n>(CoordsView<N> const&, Few<LO, n>)` returning `Matrix<N, n>`, matching the existing `Reals` overload's layout; `static_assert` that `N` is static.
- `ArrayOf<T>` is a private alias: `Reals` for `const Real`, `Write<Real>` for `Real`.

**Mesh additions**

```cpp
template <std::size_t N = dynamic_extent>
CoordsView<N> coords_view() const;           // reads the "coordinates" tag; N checked
void set_coords(CoordsView<> const& coords); // extent(0) == nverts(), extent(1) == space_dim()
```

The `N` check in `coords_view<N>()` is deliberate: a kernel that requests `coords_view<dim>()` with the topological dimension fails at once on an embedded mesh.

Acceptance: the view unit tests in Step 7 pass in serial and Kokkos builds.

## Step 4: deprecation and internal call sites

Every internal use of the raw coordinate API is replaced in this PR, because the library builds with `-Werror=deprecated-declarations`. On current `master` that is 58 calls to `coords()` in 29 library files, 35 in 19 test and tool files, 3 internal calls to `set_coords(Reals)` (`Omega_h_approach.cpp` twice, `Omega_h_overlay.cpp` once), and the Python binding in `PyOmega_h_mesh.cpp`. Public headers `Omega_h_shape.hpp` (`RealEdgeLengths`, `MetricEdgeLengths`) and `Omega_h_quality.hpp` (`MetricElementQualities`) contain inline calls and must be converted too, or downstream codes including them would see the warning.

**Deprecation mechanics**

- Add `OMEGA_H_DEPRECATED(msg)` to `Omega_h_macros.h`: expands to `[[deprecated(msg)]]` unless `OMEGA_H_SUPPRESS_DEPRECATION` is defined.
- Apply it to `Mesh::coords()` ("raw stride is space_dim(); use coords_view()") and `Mesh::set_coords(Reals const&)` ("use set_coords(CoordsView)").
- Gate both deprecated methods on embedded meshes. Typed accessors read/update the tag directly, without calling deprecated methods.
- Add `-Werror=deprecated-declarations` to the library, tests and tools for GCC, Clang and IntelLLVM, and pass it through to the host compiler in CUDA and HIP builds.
- Add `OMEGA_H_ALLOW_DEPRECATED_BEGIN/END` (diagnostic push and pop) for the one test that exercises the deprecated API.

**Gated accessor for unported code** (in `Omega_h_coords_view.hpp`, namespace `Omega_h::detail`, installed because public headers use it):

```cpp
Reals coords_tdim_only(Mesh const& mesh, char const* caller);   // fails if space_dim() != dim()
void  set_coords_tdim_only(Mesh& mesh, Reals coords, char const* caller);
#define OMEGA_H_COORDS_TDIM_ONLY(mesh) ::Omega_h::detail::coords_tdim_only(mesh, __func__)
```

The failure message names the caller and both dimensions. This accessor is itself a gate: any unported path that reads coordinates refuses an embedded mesh.

**Rule for each call site**

| Where | Replace with |
| --- | --- |
| Code ported in Step 6 | `coords_view<N>()` inside dispatch on `space_dim()`, or `coords_view()` with `space_dim()` as the stride |
| All other library code, including `Omega_h_shape.hpp` and `Omega_h_quality.hpp` | `OMEGA_H_COORDS_TDIM_ONLY(*mesh)` or `set_coords_tdim_only` |
| Tools (`osh_*` and other executables) | `OMEGA_H_COORDS_TDIM_ONLY` |
| Tests | `coords_view()` or `coords_view().raw()`; tests may use `raw()` freely |
| Python binding | Keep the Python `coords()` method, implemented with `coords_view().raw()`; add `space_dim()` |

Convert every site in every file, including files only built with optional dependencies (EGADS, Exodus, ADIOS2, dolfin, meshb, XGC). The Step 8 lint, not the local build, is the check that none were missed.

Acceptance: no call to `coords()` or `set_coords(Reals)` remains outside the definitions and the one deprecated-API test; the library builds with the warning as an error; the existing suite passes unchanged.

## Step 5: gating

An embedded mesh is refused at three levels, so a missed entry point is still caught before any wrong-stride arithmetic runs.

1. **Entry-point gates.** `OMEGA_H_GATE_EMBEDDED(mesh)` expands to `detail::gate_embedded(mesh, __func__)`, which fails when `space_dim() != dim()`. Place it as the first statement of every public function declared in the headers below that takes a `Mesh`.
2. **Backstop.** The constructors of `MetricElementQualities` and `MetricEdgeLengths` read coordinates through `OMEGA_H_COORDS_TDIM_ONLY`, so every metric-quality or metric-length path refuses embedded meshes even if it skipped an entry gate.
3. **Gated accessor.** Every unported coordinate read (Step 4) refuses embedded meshes on its own.

The message format, which the tests match on, is `"<function>: meshes with space_dim (3) != dim (2) are not supported"`.

| Header | Gate every public function taking a Mesh |
| --- | --- |
| `Omega_h_adapt.hpp` | `adapt`, `grade_fix_adapt`, `print_adapt_status`, `print_adapt_histograms` |
| `Omega_h_refine.hpp`, `Omega_h_coarsen.hpp`, `Omega_h_swap.hpp` | `refine`, `refine_by_size`, `coarsen_by_size`, `coarsen_slivers`, `swap_edges` and the rest |
| Fix, approach and warp headers | `fix`, `fix_for_given_metric`, `approach_metric`, `warp_to_limit` |
| `Omega_h_metric.hpp` | every `generate_*`, `add_*_tag`, `get_*_metrics`, `get_implied_*`, `get_complexity*`, `get_expected_nelems*`, `limit_metric_gradation`, `smooth_metric_once`, `project_metrics` |
| `Omega_h_recover.hpp`, `Omega_h_fit.hpp` | `recover_gradients`, `recover_hessians*`, `derive_element_*`, `project_by_fit`, `project_by_average` |
| `Omega_h_conserve.hpp` | `setup_conservation_tags`, `correct_integral_errors`, all `transfer_*` |
| `Omega_h_quality.hpp` | `measure_qualities` |
| `Omega_h_amr.hpp` | every AMR entry point |
| `Omega_h_surface.hpp`, `Omega_h_confined.hpp` | every function taking a Mesh |
| `Mesh` members | `ask_qualities()`, and `balance(true)` (predictive balancing uses metric complexity) |

Pure array functions that take no `Mesh` (for example `intersect_metrics`, `metrics_from_isos`) are not gated. The header list names files as they exist on `master`; where a name differs, gate the file that declares the function.

Acceptance: every function in the table refuses an embedded mesh with the message format above (Step 7 test T5).

## Step 6: components ported in PR 1

Only the components below work on embedded meshes after PR 1; everything else is refused by Step 5.

| Component | Change |
| --- | --- |
| Dispatch helper | New `dispatch_sdim_edim(sdim, edim, f)` in an internal header. Lists exactly the pairs (1,1), (2,1), (2,2), (3,1), (3,2), (3,3); any other pair fails with a message. |
| Element and edge measures | `measure_ents_real`, `measure_elements_real`, `measure_edges_real`: dispatch on `(space_dim(), ent_dim)` and read through `coords_view<sdim>()`. The `<3, 2>` and `<3, 1>` kernels already exist. |
| `Mesh::ask_sizes()` | Works through the ported measures. |
| `Mesh::ask_lengths()` | Without a `"metric"` tag: `RealEdgeLengths` dispatched on `space_dim()`. With one: stays gated through `MetricEdgeLengths`. |
| Balancing | `Mesh::balance(false)` and `Mesh::balance(Reals)`: average with width `space_dim()`, and pad to 3 only when `space_dim() < 3`. `balance(true)` stays gated. |
| osh I/O | Done in Step 1. |
| VTK output | Writers pad with `resize_vectors(coords, mesh->space_dim(), 3)`. Readers unchanged. |
| Migration and ghosting | No code change expected: the coordinates tag moves by its component count and `copy_meta` carries `space_dim_`. Verified by test T6. |

**Classification.** An embedded mesh built in tests has no classification. Use only topological classification (`classify_sides_by_exposure`, `classify_elements`, then `finalize_classification`), and confirm none of these reads coordinates. `classify_by_angles` stays gated.

**Measure sign.** A triangle's area is signed in 2D and unsigned in 3D, so embedded element sizes are always positive. Every caller that treats a non-positive size as an inverted element is in gated code; do not port any such caller in this PR.

Acceptance: tests T3, T4 and T6 pass.

## Step 7: tests

Two new test executables cover the PR: `unit_embedded.cpp` (serial) and `embedded_mpi_test.cpp` (2 and 4 ranks), both registered with `test_func` in `src/CMakeLists.txt`.

**Fixture: the tilted plane.** Build a 2D triangle box with `build_box`, take its element-to-vertex array, and map each vertex (x, y) to R·(x, y, 0) + t. Use a fixed rotation R (30° about x, then 45° about y, then 60° about z) and t = (0.3, −1.2, 2.5). Build the embedded mesh with the new `build_from_elems_and_coords(..., sdim = 3)` overload. Do the same for a 1D box to get edges in 3D. Never use z = 0 alone: a wrong-stride read of a zero component passes by accident.

| ID | Test | Catches | Pass criterion |
| --- | --- | --- | --- |
| T1 | Existing suite | Default-path regressions | All existing CTest tests pass; gold-file comparisons unchanged |
| T2 | Mesh metadata | Wrong default or validation | `space_dim() == dim()` for 1D/2D/3D boxes; `set_space_dim` fails for sdim < dim, sdim > 3 and coordinate component mismatch; `copy_meta` keeps it; meshes differing only in `space_dim` compare unequal |
| T3 | Coordinate view | View indexing and checks | Static and dynamic views index identically; static `N` mismatch fails at construction in a release build; `d >= extent(1)` fails in a debug build; `gather_vectors` matches manual reads; `set_coords(CoordsView)` rejects wrong extents |
| T4 | Invariance under embedding | Wrong strides, dropped components | Element areas and edge lengths of the tilted mesh match the 2D mesh to 1e-12 relative; total area matches; same for edge lengths in the 1D-in-3D and 1D-in-2D cases; a folded triangulation has the expected unsigned areas |
| T5 | Gates | Missing or misfiring gates | Every function in the Step 5 table throws `Omega_h::exception` whose message contains `space_dim (3) != dim (2)`; built and registered only when `Omega_h_THROW=ON` |
| T6 | Parallel | Wrong migration widths, lost metadata | On 2 and 4 ranks: build on rank 0, then `balance(false)`, ghost one layer, return to element-based; `space_dim() == 3` on every rank; each vertex's coordinates, matched by global id, equal the originals exactly; the global area sum matches the serial total to 1e-12 relative |
| T7 | I/O | Lost space dimension in files | Embedded mesh round-trips through osh (serial and 2 ranks) with identical coordinates and `space_dim() == 3`; `meshes/plate_6elem.osh` (version 9) loads with `space_dim() == dim()`; an uncompressed binary VTU written for the embedded mesh has 3-component points equal to the mesh coordinates, checked by decoding the Points array |
| T8 | Deprecated API | Breaking downstream behavior | Inside `OMEGA_H_ALLOW_DEPRECATED_BEGIN/END`: `coords()` equals `coords_view().raw()` and `set_coords(Reals)` still works on a 2D mesh |

For T6, build with the comm-taking `build_from_elems2verts` overload: rank 0 passes the full arrays, the other ranks pass empty arrays. Then call `set_space_dim(3)` and `add_coords` on every rank.

## Step 8: lint and CI

One grep-based script, `tools/lint/check_coords.sh`, enforces what the compiler cannot; it runs in CI and is also registered as a CTest test so it runs locally.

| Rule | Check | Allowlist |
| --- | --- | --- |
| L1 | No call to `coords()` (not `coords_mix`, not `coords_view`) and no `set_coords` taking `Reals` | Their definitions in `Omega_h_mesh.*`; the T8 test |
| L2 | The string literal `"coordinates"` | Files that use it on current `master` (mesh, file, VTK, transfer, compare, mixed mesh, and the dolfin, exodus and meshb readers), frozen into `tools/lint/coordinates_allowlist.txt` |
| L3 | `.raw()` on a coordinate view | `Omega_h_coords_view.hpp`, the ported balancing and VTK code, the Python binding, all test files |
| L4 | `OMEGA_H_ALLOW_DEPRECATED_BEGIN` | The T8 test only |
| L5 | Count of `OMEGA_H_COORDS_TDIM_ONLY` uses | Fails if the count rises above `tools/lint/tdim_only_baseline.txt`; the count is the audit backlog for later PRs |

| Config | MPI | Kokkos | Build | THROW | Runs |
| --- | --- | --- | --- | --- | --- |
| C1 | on | off | Debug | on | T1–T8, lint; vendored mdspan |
| C2 | off | off | Release | off | T1–T4, T7 (serial), T8; vendored mdspan |
| C3 | on | ≥ 4.4, Serial or OpenMP | Release | on | T1–T8; Kokkos's mdspan |
| C4 | on | ≥ 4.4, CUDA or HIP | Release | on | Build, T3, T4, T6; device capture of views |
| C5 | — | below 4.4 | — | — | Configure only: must fail with the version message |

Extend the existing `.github/workflows/cmake.yml` matrix rather than adding a workflow: it already builds with Kokkos on and off, with and without MPI, which covers C1 to C3. In C2, where failures abort, failure-expecting checks (the static-extent mismatch in T3) go in a separate small executable registered with the CTest `WILL_FAIL` property. C4 runs only if a GPU runner exists; if none does, record that in the PR description.

## Instructions for the implementing agent

Implement Steps 1 to 5, then write and exercise Step 7 tests before the Step 6 ports, then finish Steps 7 and 8. Use one commit per completed step on a branch from SCOREC/omega_h `master`; stop and report rather than improvise when the plan and the code disagree.

1. Before Step 1, build `master` in configs C1 and C2 and record the full CTest result. T1 compares against this baseline, so pre-existing failures must be known, not introduced.
2. After each step, rebuild and run the full suite in C1 and C2. A step is done only when its acceptance line holds.
3. Do not change behavior outside this plan: no refactors, no renames beyond those listed, no fixes to unrelated bugs. Note anything suspicious in the PR description instead.
4. In Step 4, convert call sites mechanically by the rule table. Do not port a site to `space_dim()` unless Step 6 lists it; when unsure, use the gated accessor.
5. Write the Step 7 tests before the Step 6 ports, and confirm T4 and T6 fail without the ports. This proves the tests catch the bugs they target.
6. Report the final L5 count (remaining gated coordinate reads) and the list of files that contain them.

**Stop and ask if:**

- a gated function is called on a path that existing tests use for 2D or 3D meshes and the gate fires there (that is a bug in the gate condition);
- a mesh-construction, I/O or migration path reads coordinates through the gated accessor, which would make embedded meshes unusable;
- `finalize_classification` or any other topological routine turns out to read coordinates;
- a Kokkos 4.4 or newer install passes the configure check but `mdspan/mdspan.hpp` is not found.

**Definition of done**

- [ ] Steps 1 to 8 merged as separate commits; each passes its acceptance line
- [ ] T1 matches the recorded baseline in C1 and C2
- [ ] T2 to T8 pass in every config listed for them, or the PR states why a config could not run
- [ ] Lint rules L1 to L5 pass
- [ ] PR description lists the L5 backlog, any skipped configs, the now-unreachable Kokkos version checks, and anything suspicious found

## Out of scope

Later PRs, not this one: the audit of the gated coordinate reads (the L5 backlog) and the bare-`dim` ban; opt-in readers that create embedded meshes (gmsh, VTK) and boundary extraction into an embedded mesh; porting bounding boxes, Hilbert reordering, `classify_by_angles` and centroids; `MixedMesh`; downstream migrations (PUMI-Tally, and meshFields' `SetCoordFieldFunctor`, which strides coordinates by the mesh dimension). Adaptation on embedded meshes is expected to come from a separate adaptation library, not from Omega_h.

## Defaults pending review

Use the defaults below, including the file-version and test-coverage decisions confirmed in review.

| Question | Default to implement |
| --- | --- |
| osh version: always write 12, or only when `space_dim() != dim()`? | Always write version 12 |
| Tools: gate all `osh_*` tools, or port trivial ones such as `osh_scale`? | Gate all tools |
| `set_space_dim` visibility: public, or restricted to build overloads? | Public |
| 1D in 2D: add an edge-in-2D case to T4? | Yes; also test a folded, nonplanar triangulation with known areas |
| Python: expose a typed `coords_view()` too? | Expose `space_dim()` only |
| GPU CI: is a CUDA or HIP runner available for C4? | Skip C4 if none; say so in the PR |
