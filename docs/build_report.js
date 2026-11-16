// Builds docs/Parallel-Vision-Report.docx from results/summary.json, results/*.csv and the PNG figures.
//   cd docs && npm install && node build_report.js
// Every number and graph in the report comes from the scripts in ../scripts (run_all.ps1, demo.ps1).
const fs = require("fs");
const os = require("os");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  HeadingLevel, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber, PageBreak, Math: DMath, MathRun,
  MathFraction, MathSubScript, VerticalAlign,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const RES = path.join(ROOT, "results");
const S = JSON.parse(fs.readFileSync(path.join(RES, "summary.json"), "utf8"));
const FONT = "Calibri";
const W = 9638; // content width in DXA (A4, 2 cm margins)
const f1 = (x) => x.toFixed(1), f2 = (x) => x.toFixed(2), f0 = (x) => Math.round(x).toString();
const num = (n) => Math.round(n).toLocaleString("en-US");
const px = (n) => (n >= 2147483647 ? "never" : n <= 0 ? "always" : num(n));

// ---------------------------------------------------------------- helpers
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: 22, ...o });
const P = (t, o = {}) => new Paragraph({
  spacing: { after: 120, line: 276 }, alignment: o.align,
  children: Array.isArray(t) ? t : [run(t, o.run)], ...(o.p || {}),
});
const rich = (...parts) => P(parts.map((p) => (typeof p === "string" ? run(p) : run(p.t, p))));
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })] });
const bullet = (parts) => new Paragraph({
  numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 276 },
  children: (Array.isArray(parts) ? parts : [parts]).map((p) => (typeof p === "string" ? run(p) : run(p.t, p))),
});
const code = (lines) => lines.map((l, i) => new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: "F3F4F6" },
  spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 160 : 0 },
  indent: { left: 120, right: 120 },
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 18 })],
}));
const caption = (t) => P(t, { run: { italics: true, size: 19, color: "4B5563" }, p: { spacing: { before: 60, after: 180 } } });
function pngSize(file) { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function figure(rel, cap, widthPx = 620) {
  const full = path.join(RES, rel);
  if (!fs.existsSync(full)) { console.warn("missing figure", rel); return []; }
  const [w, h] = pngSize(full);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(full), transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
        altText: { title: cap, description: cap, name: path.basename(rel) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [run(cap, { italics: true, size: 19, color: "4B5563" })] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, width, o = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({ alignment: o.align || AlignmentType.LEFT, children: [run(String(text), { size: o.size || 20, bold: o.bold, color: o.color })] })],
  });
}
function table(headers, rows, widths, opts = {}) {
  const hdr = new TableRow({ tableHeader: true, cantSplit: true,
    children: headers.map((h, i) => cell(h, widths[i], { bold: true, fill: "1F3A5F", color: "FFFFFF", size: opts.size, align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) });
  const body = rows.map((r, ri) => new TableRow({ cantSplit: true,
    children: r.map((c, i) => {
      const hl = opts.highlight && opts.highlight(ri, r);
      return cell(c, widths[i], { size: opts.size, align: i ? AlignmentType.CENTER : AlignmentType.LEFT, fill: hl ? "DCFCE7" : ri % 2 ? "F9FAFB" : undefined, bold: hl });
    }) }));
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths, rows: [hdr, ...body] });
}
const spacer = () => new Paragraph({ spacing: { after: 160 }, children: [] });
const mr = (t) => new MathRun(t);
const eq = (...children) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 }, children: [new DMath({ children })] });
const sub = (b, s) => new MathSubScript({ children: [mr(b)], subScript: [mr(s)] });

// ---------------------------------------------------------------- data
const sc = S.scaling_4096 || [];
const omp4096 = sc.find((r) => r.config.startsWith("OpenMP"));
const bestHyb = sc.filter((r) => r.config.startsWith("Hybrid")).sort((a, b) => a.ms - b.ms)[0];
const disp = Object.fromEntries((S.dispatch || []).map((d) => [d.strategy, d]));
const aSeq = disp["Always sequential"], aOmp = disp["Always OpenMP"], hz = disp["pv heuristic"];
const cal = disp["pv calibrated (robust)"], old = disp["pv calibrated (old rule)"];
const ops = S.ops || [];
const topOps = [...ops].sort((a, b) => b.speedup - a.speedup);
const nOps = ops.length || 11;
const cpu = os.cpus()[0].model.replace(/\s+/g, " ").trim();
const ocv = S.opencv_4096;
const sw = S.switch_points || {};
const spread = S.calibration_spread;
const model = S.mpi_model || [];
const nChecks = (fs.readFileSync(path.join(ROOT, "tests", "test_main.cpp"), "utf8").match(/CHECK\(/g) || []).length;
const avgModelErr = (() => { const m = model.filter((r) => r.local === "sequential"); return m.length ? m.reduce((a, r) => a + r.heldout_error_pct, 0) / m.length : NaN; })();
const last = (a) => a[a.length - 1];
const mpiNever = model.length > 0 && model.every((r) => r.crossover_px < 0);

// ---------------------------------------------------------------- content
const kids = [];

// ---- title page
kids.push(
  new Paragraph({ spacing: { before: 1800, after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Parallel-Vision", font: FONT, size: 72, bold: true, color: "1F3A5F" })] }),
  new Paragraph({ spacing: { after: 360 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "A C++ image-processing library that chooses between sequential and parallel execution", font: FONT, size: 30, color: "374151" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3A5F", space: 8 } }, children: [] }),
  new Paragraph({ spacing: { before: 300, after: 60 }, alignment: AlignmentType.CENTER,
    children: [run("Parallel Programming Laboratory — Semester 5 Mini-Project", { size: 26, bold: true })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [run("Final Report", { size: 24, bold: true, color: "2563EB" })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [run("Technologies: C++17 · OpenMP · MPI (MS-MPI) · CUDA · CMake · Python (matplotlib)", { size: 22, color: "4B5563" })] }),
  new Table({
    width: { size: 7000, type: WidthType.DXA }, columnWidths: [4600, 2400], alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ children: [cell("Author", 4600, { bold: true, fill: "1F3A5F", color: "FFFFFF" }), cell("Roll No.", 2400, { bold: true, fill: "1F3A5F", color: "FFFFFF", align: AlignmentType.CENTER })] }),
      new TableRow({ children: [cell("Mohammed Mehdi Hedayati", 4600), cell("241080023", 2400, { align: AlignmentType.CENTER })] }),
      new TableRow({ children: [cell("Nathan Julian Dsouza", 4600, { fill: "F9FAFB" }), cell("241080016", 2400, { align: AlignmentType.CENTER, fill: "F9FAFB" })] }),
    ],
  }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---- 1 Introduction
kids.push(H1("1. What is Parallel-Vision?"),
  P(`Parallel-Vision (pv) is a C++17 shared library for image-processing algorithms. It provides ${nOps} operations — grayscale conversion, Gaussian blur, median filter, Sobel, Prewitt and Laplacian edge detectors, the full Canny edge detector, thresholding (including Otsu’s method), histogram equalisation, and erosion and dilation. Every operation exists in a sequential version and in parallel versions: OpenMP on one machine, MPI across processes, a hybrid of the two, and a CUDA version of Canny for NVIDIA GPUs. The distinguishing feature is that the caller does not have to decide which one to use: by default pv looks at the operation and the image size and picks the backend expected to be fastest.`),
  P("The motivation is a common mistake in parallel programming: parallelism is not free. Starting threads, splitting work and exchanging data cost time, so on a small image a parallel program can be many times slower than a simple loop, while on a large image a sequential program leaves most of the machine idle. A library that always runs in parallel, or never does, loses either way. pv measures where the break-even point lies, for each operation and on the machine it runs on, and uses it."),
  rich({ t: "Goals. ", bold: true }, "(1) one API with several interchangeable backends; (2) identical results from every backend; (3) automatic, measurable backend selection; (4) every claim backed by results and graphs that a single script regenerates."));

// ---- 2 Architecture
kids.push(H1("2. How the library is organised"),
  ...figure("fig_architecture.png", "Figure 1. Architecture: one public API, a dispatcher, and interchangeable backends sharing the same row kernels."),
  H2("2.1 Operations and backends"),
  table(["Operation", "Function", "Sequential", "OpenMP", "MPI", "CUDA"],
    [["Grayscale (BGR → gray)", "to_gray", "yes", "yes", "in Canny", "in Canny"],
     ["Gaussian blur (separable)", "gaussian_blur", "yes", "yes", "in Canny", "in Canny"],
     ["Median filter (3×3, 5×5, 7×7)", "median_filter", "yes", "yes", "–", "–"],
     ["Sobel gradient", "sobel", "yes", "yes", "in Canny", "in Canny"],
     ["Prewitt gradient", "prewitt", "yes", "yes", "–", "–"],
     ["Laplacian", "laplacian", "yes", "yes", "–", "–"],
     ["Canny edge detector", "canny", "yes", "yes", "yes", "yes*"],
     ["Threshold, Otsu’s method", "threshold, otsu_threshold", "yes", "yes", "–", "–"],
     ["Histogram equalisation", "equalize_hist", "yes", "yes", "–", "–"],
     ["Erosion / dilation (square element)", "erode, dilate", "yes", "yes", "–", "–"]],
    [2800, 2100, 1100, 1000, 1250, 1388], { size: 19 }),
  caption("Table 1. Operations and the backends that implement them. *CUDA covers the whole Canny pipeline except hysteresis (Sections 4.3 and 6.9)."),
  H2("2.2 Using the library"),
  P("The library has no dependency on OpenCV; an image is a small struct holding a byte buffer. The command-line tool uses OpenCV only to read and write image files."),
  ...code([
    '#include "pv/pv.hpp"',
    '',
    'pv::Image edges = pv::canny(img);                        // Backend::Auto: pv decides',
    'pv::Image a = pv::canny(img, 50, 150, 1.4, pv::Backend::Sequential);',
    'pv::Image b = pv::median_filter(gray, 1, pv::Backend::OpenMP);',
    '',
    'pv::calibrate();                                          // optional: measure THIS machine',
    'pv::save_calibration("pv_calibration.txt");               // reload next time with load_calibration()',
  ]));

// ---- 3 Sequential and parallel in one library
kids.push(H1("3. Sequential and parallel code in the same library"),
  P("Each algorithm is written once, as a function that processes a range of image rows. A small helper decides how that function is driven:"),
  ...code([
    'template <class F>',
    'void for_rows(Backend b, int y0, int y1, F&& f) {',
    '    if (b == Backend::OpenMP) {',
    '        #pragma omp parallel for schedule(static)',
    '        for (int y = y0; y < y1; ++y) f(y);',
    '    } else {',
    '        for (int y = y0; y < y1; ++y) f(y);          // sequential',
    '    }',
    '}',
  ]),
  P(`Because the arithmetic is shared, the sequential and parallel versions cannot drift apart. This is checked, not assumed: the test-suite (${nChecks} automated checks) requires the OpenMP result to equal the sequential result bit for bit for every operation, and the MPI result to equal it for 1, 2, 3, 4 and 8 processes on six different image shapes (Section 6.2). The MPI backend simply runs the same row kernels on each process’s strip of rows.`));

// ---- 4 Backends
kids.push(H1("4. The parallel backends"),
  H2("4.1 OpenMP"),
  P("Rows are distributed over threads with a static schedule. The separable filters (Gaussian blur, erosion, dilation) run a horizontal and a vertical pass with a barrier between them, arranged so the innermost loop is contiguous and vectorises. Median 3×3 uses a fixed 19-step sorting network, which has no data-dependent branches. Histogram equalisation is a parallel reduction: each thread counts into a private 256-bin array and the arrays are summed, so the result does not depend on the thread count. In Canny, hysteresis thresholding needs connectivity across the whole image, so only its seed scan is parallel; the flood fill is serial but touches only edge pixels. Scratch buffers are left uninitialised and filled by the threads that use them, which avoids a serial zero-fill of hundreds of megabytes."),
  H2("4.2 MPI with halo exchange"),
  P("The image is cut into horizontal strips, one per process. Rank 0 scatters the strips; neighbouring ranks then exchange halo rows so that every owned row is computed from exactly the same neighbourhood as in the sequential run. The halo depth is the Gaussian radius plus one row each for the gradient and for non-maximum suppression:"),
  eq(mr("d = ⌈ 3σ ⌉ + 2")),
  P("For the default σ = 1.4 this is 7 rows. Each rank runs blur, gradient and suppression on its extended strip and keeps only its own rows; compact edge labels (1 byte per pixel: none, weak, strong) are gathered on rank 0, which performs the global hysteresis step. Ranks may also use OpenMP inside their strip, giving a hybrid MPI + OpenMP mode."),
  ...code([
    'pv::mpi_init(&argc, &argv);',
    'pv::Image out = pv::mpi_canny(in, 50, 150, 1.4, pv::Backend::OpenMP);  // collective; result on rank 0',
    'pv::mpi_finalize();',
  ]),
  H2("4.3 CUDA"),
  P("The GPU backend runs the Canny pipeline as five kernels (gray, horizontal blur, vertical blur, gradient, non-maximum suppression); hysteresis stays on the CPU because it needs global connectivity. The kernels mirror the CPU code operation for operation — the same tap order, the same direction rules, and compilation with fused multiply-add disabled — so the output is designed to be bit-identical to the sequential result. A test (tests/test_cuda.cpp) that compares the GPU result with the sequential one and prints timings is provided for any CUDA machine (Linux, Windows with MSVC, or Google Colab). The expected performance of this backend is estimated in Section 6.9."),
  H2("4.4 Measuring parallel performance"),
  P("With T_seq the sequential time and T_p the time using p workers, speedup and efficiency are"),
  eq(sub("S", "p"), mr(" = "), new MathFraction({ numerator: [sub("T", "seq")], denominator: [sub("T", "p")] }),
     mr(",      "), sub("E", "p"), mr(" = "), new MathFraction({ numerator: [sub("S", "p")], denominator: [mr("p")] })),
  P("If a fraction f of the work can be parallelised, Amdahl’s law bounds the speedup:"),
  eq(sub("S", "p"), mr(" ≤ "), new MathFraction({ numerator: [mr("1")], denominator: [mr("(1 − f) + "), new MathFraction({ numerator: [mr("f")], denominator: [mr("p")] })] })));

// ---- 5 Dispatcher
kids.push(H1("5. Choosing the backend automatically"),
  P("When the caller passes Backend::Auto (the default), the dispatcher compares the image size N = width × height with a per-operation switch point N*(op):"),
  eq(mr("backend = OpenMP  if  N ≥ N*(op)  and more than one thread is available;  otherwise  Sequential")),
  H2("5.1 Heuristic and calibrated switch points"),
  bullet([{ t: "Heuristic: ", bold: true }, "a fixed default of 100,000 pixels for every operation."]),
  bullet([{ t: "Calibrated: ", bold: true }, "pv::calibrate() times sequential against OpenMP on synthetic images from 16² to 1024² on the current machine, separately for each operation. The robust rule smooths the measured log(time ratio) with a 3-point median, so a single noisy measurement cannot move the answer, finds the size from which OpenMP stays ahead, and interpolates the crossing in log-size. The original rule, kept for comparison, took the first grid size from which OpenMP won by at least 10% at every larger size. Switch points can be saved and reloaded (pv --calibrate, pv --use-calibration)."]),
  H2("5.2 MPI cost model"),
  P("MPI is a collective operation (every process must call it), so Backend::Auto never selects it. Instead the library can fit a cost model that says when it would pay off. The time of each backend is modelled as a straight line in the number of pixels, T(N) = a + bN, fitted by least squares from measurements on 1024² to 3072² images (the intercept is constrained to be non-negative, since fixed overhead cannot be negative):"),
  eq(sub("T", "backend"), mr("(N) = "), sub("a", "backend"), mr(" + "), sub("b", "backend"), mr(" · N")),
  P("MPI pays off from the size at which its line stays below both the sequential and the OpenMP lines (pv::fit_cost, pv::fit_mpi_cost, pv::mpi_crossover_pixels). The model is then checked on larger images (4096² and 5120²) that were not used for fitting. In hybrid mode the single-process OpenMP baseline is measured with all hardware threads, so MPI is always compared with the best a single process can do."),
  H2("5.3 CUDA"),
  P("When a GPU is present, calibrate() also times the GPU against the best CPU backend and Auto uses CUDA for Canny above the measured size; without a GPU this path is never taken."));

// ================================================================ 6 Results
kids.push(new Paragraph({ children: [new PageBreak()] }), H1("6. Results"),
  H2("6.1 Test setup"),
  table(["Item", "Value"],
    [["Machine", `${cpu}, ${os.cpus().length} logical threads, ${Math.round(os.totalmem() / 2 ** 30)} GB RAM`],
     ["System", "Windows 11, MinGW-w64 g++ 16.1 (MSYS2 UCRT64), -O3, OpenMP 5.2, MS-MPI 2.0, OpenCV 4.13 (reference only)"],
     ["Workloads", `All ${nOps} operations on synthetic images from 128² to 4096²; Canny from 32² to 4096²; a 3840×2400 photograph`],
     ["Method", "Best of 3–7 runs after a warm-up; MPI timings include scatter, halo exchange and gather (data starts and ends on rank 0)"],
     ["Reproduce", "scripts\\run_all.ps1 regenerates every number and figure in this section; scripts\\demo.ps1 is the one-minute live version"]],
    [1700, 7938]),
  spacer());

// key results box
const keyPts = [];
keyPts.push([{ t: "Same answer everywhere. ", bold: true }, `OpenMP, MPI (1, 2, 3, 4, 8 ranks) and hybrid output is bit-identical to the sequential output for the tested operations; Canny edges also agree with OpenCV’s (overlap IoU ${f2(S.iou || 0)}).`]);
if (topOps.length) keyPts.push([{ t: `Up to ${f2(topOps[0].speedup)}× faster than sequential `, bold: true }, `(${topOps[0].op}, ${S.ops_size}² image, ${f1(topOps[0].seq_ms)} ms → ${f1(topOps[0].omp_ms)} ms); ${topOps.filter((o) => o.speedup >= 2.5).length} of ${nOps} operations are at least 2.5× faster on ${omp4096 ? omp4096.workers : 8} threads.`]);
if (aOmp && cal) keyPts.push([{ t: `Automatic choice costs only about ${f1((cal.geomean - 1) * 100)}% on average `, bold: true }, `versus the best possible backend, while always using OpenMP is ${f0((aOmp.geomean - 1) * 100)}% slower on average and always using sequential is ${f0((aSeq.geomean - 1) * 100)}% slower (${S.dispatch_cases} cases, ${S.dispatch_ops} operations).`]);
if (aOmp && cal) keyPts.push([{ t: "Avoids disasters. ", bold: true }, `Worst single case: ${f1(cal.worst)}× slower than optimal with pv, against ${f0(aOmp.worst)}× for naive always-parallel on tiny images.`]);
if (hz && cal) keyPts.push([{ t: "Right choice almost every time. ", bold: true }, `pv picked a backend within 10% of the best in ${f0(hz.within10)}% of cases with the fixed heuristic and ${f0(cal.within10)}% after calibration.`]);
const cuda = S.cuda_projection ? S.cuda_projection.rows[S.cuda_projection.rows.length - 1] : null;
if (cuda) keyPts.push([{ t: `GPU backend expected to reach about ${f1(cuda.speedup_vs_seq)}× over sequential `, bold: true }, `for Canny on a ${cuda.size}² image (${f0(cuda.seq_ms)} ms → ${f0(cuda.cuda_ms_typical)} ms, ${f1(cuda.speedup_vs_omp)}× over 8-thread OpenMP). (Expected speedups from a bandwidth model; not verified on GPU hardware.)`]);
if (model.length) keyPts.push([{ t: "Predicts when MPI would pay off. ", bold: true }, `The fitted cost model predicts running times on larger, unseen images to within about ${f0(avgModelErr)}% on average, and ${mpiNever ? "correctly reports that MPI never beats OpenMP on this single machine" : "reports the crossover size per configuration"}.`]);
kids.push(new Table({
  width: { size: W, type: WidthType.DXA }, columnWidths: [W],
  rows: [new TableRow({ children: [new TableCell({
    width: { size: W, type: WidthType.DXA }, margins: { top: 140, bottom: 100, left: 220, right: 220 },
    shading: { type: ShadingType.CLEAR, fill: "ECFDF5", color: "auto" },
    borders: { top: { style: BorderStyle.SINGLE, size: 12, color: "059669" }, bottom: border, left: border, right: border },
    children: [new Paragraph({ spacing: { after: 100 }, children: [run("Key results", { bold: true, size: 26, color: "065F46" })] }), ...keyPts.map(bullet)],
  })] })],
}), spacer());

// 6.2 correctness
kids.push(H2("6.2 Correctness"),
  table(["Check", "Result"],
    [[`Unit tests: OpenMP vs sequential for all ${nOps} operations, median vs brute-force reference, morphology and histogram properties, dispatcher rules (${nChecks} checks)`, "all pass"],
     ["MPI vs sequential Canny (1, 2, 3, 4, 8 ranks × 6 image shapes incl. 100×9 and 16×16)", "bit-identical (30 / 30 runs)"],
     ["Hybrid MPI + OpenMP vs sequential (4 ranks × 2 threads × 6 image shapes)", "bit-identical (6 / 6 runs)"],
     ["Calibration save / load round-trip, fallback rules for unavailable backends", "pass"],
     [`Canny edge agreement with OpenCV on a 3840×2400 photo (${num(S.edge_px_pv || 0)} vs ${num(S.edge_px_cv || 0)} edge pixels)`, `IoU = ${f2(S.iou || 0)}`]],
    [6800, 2838]),
  spacer(),
  ...figure("demo/fig_demo_montage.png", `Figure 2. Every algorithm applied to the same photograph by the pv command-line tool (output of scripts\\demo.ps1).`, 640),
  ...figure("fig_edges.png", "Figure 3. Canny edges produced by pv (automatic backend) on a 3840×2400 photograph."));

// 6.3 all operations
kids.push(H2("6.3 Speedup of every operation"),
  ...figure("fig_ops.png", `Figure 4. Left: speedup of OpenMP over sequential for each operation. Right: speedup versus image size.`, 640),
  table(["Operation", "Sequential (ms)", "OpenMP (ms)", "Speedup", "Efficiency"],
    [...ops].sort((a, b) => b.speedup - a.speedup).map((o) => [o.op, f2(o.seq_ms), f2(o.omp_ms), f2(o.speedup) + "×", f0(o.efficiency * 100) + "%"]),
    [2600, 1900, 1900, 1600, 1638], { highlight: (i) => i < 3 }),
  caption(`Table 2. ${S.ops_size}×${S.ops_size} image, ${omp4096 ? omp4096.workers : 8} threads. Efficiency = speedup / threads. The three fastest-scaling operations are highlighted.`),
  P(`The best-scaling operations are ${topOps.slice(0, 3).map((o) => o.op + " (" + f2(o.speedup) + "×)").join(", ")}; all ${nOps} operations are faster in parallel at this size. Canny scales least (${f2((ops.find((o) => o.op === "canny") || { speedup: 0 }).speedup)}×): it makes many full passes over the image, so it is limited by memory bandwidth, and its hysteresis step is partly serial (Amdahl’s law). Efficiency is below 50% for every operation because eight threads share one memory system.`));

// 6.4 canny scaling
kids.push(H2("6.4 Canny across all backends"),
  ...figure("fig_scaling.png", "Figure 5. Left: Canny time versus image size. Right: speedup at 4096×4096 for each parallel configuration."),
  table(["Configuration", "Workers", "Time (ms)", "Speedup", "Efficiency"],
    sc.map((r) => [r.config, r.workers, f1(r.ms), f2(r.speedup) + "×", f0(r.efficiency * 100) + "%"]),
    [3300, 1400, 1600, 1600, 1738], { highlight: (i, r) => r[0] === (sc.reduce((a, b) => (b.ms < a.ms ? b : a), sc[0] || { ms: 1e30 }).config) }),
  caption("Table 3. 4096×4096 Canny. Efficiency = speedup / workers."),
  ...(S.stages ? [
    ...figure("fig_stages.png", `Figure 5b. Left: time per Canny stage for sequential and OpenMP. Right: share of the serial hysteresis stage when running in parallel.`, 640),
    P(`The stage timing explains the ceiling. Gray conversion and the filters (blur, gradient, suppression) speed up well with OpenMP, but hysteresis takes ${f0(S.stages.sequential.hysteresis_ms)} ms sequentially and ${f0(S.stages.openmp.hysteresis_ms)} ms with eight threads: it barely speeds up, and so it grows to ${f0(S.hysteresis_share_openmp_pct)}% of Canny’s parallel run time at ${S.stages_size}². This is Amdahl’s law in practice.`)] : []),
  P(`Parallel execution pays off from about 256² pixels upward; below that, thread or process start-up cost dominates. At 4096² every parallel configuration lands between ${f2(Math.min(...sc.filter((r) => r.workers > 1).map((r) => r.speedup)))}× and ${f2(Math.max(...sc.map((r) => r.speedup)))}×: the same row kernels scale across OpenMP, MPI and the hybrid, and all are bounded by the same memory bandwidth and serial hysteresis.`));

// 6.5 dispatcher
kids.push(H2("6.5 The dispatcher at work"),
  P(`To test the automatic selection, ${S.dispatch_cases} cases (${S.dispatch_ops} operations × 14 image sizes, including non-square shapes that were not used for calibration) were timed with both backends. The “oracle” is the faster of the two in each case. The strategies are compared against it:`),
  ...figure("fig_dispatch.png", "Figure 6. Left: average slowdown relative to the best possible choice (geometric mean over all cases). Right: the single worst case.", 640),
  table(["Strategy", "Total time vs. best", "Average slowdown", "Worst case", "Within 10% of best"],
    (S.dispatch || []).map((d) => [d.strategy, f0(d.total_vs_oracle_pct) + "%", f2(d.geomean) + "×", f1(d.worst) + "×", f0(d.within10) + "%"]),
    [2900, 1700, 1700, 1400, 1938], { highlight: (i, r) => r[0].startsWith("pv") }),
  caption("Table 4. Backend-selection strategies over all test cases (lower is better; highlighted rows are pv)."),
  P(`A fixed choice is wrong somewhere: always-OpenMP is catastrophic on tiny images (up to ${f0(aOmp.worst)}× slower), and always-sequential wastes the machine on large ones (${f2(S.large_always_seq_slowdown)}× slower on images of one megapixel and above). The dispatcher avoids both failure modes.`),
  H2("6.6 Where the crossover is"),
  ...figure("fig_crossover.png", "Figure 7. Sequential versus OpenMP time for each operation. The curves cross at different sizes, which is why the switch point is per operation.", 640),
  table(["Operation", "Heuristic", "Calibrated (robust)", "Calibrated (old rule)", "Approx. square"],
    Object.keys(sw).map((k) => ["" + k, "100,000", px(sw[k].robust), px(sw[k].strict), sw[k].robust > 0 && sw[k].robust < 2147483647 ? Math.round(Math.sqrt(sw[k].robust)) + "²" : "–"]),
    [2000, 1700, 2100, 2100, 1738]),
  caption("Table 5. Switch points (pixels) measured on the test machine. Heavy per-pixel operations such as median pay off at much smaller sizes than cheap ones such as threshold."));

// 6.7 calibration stability
if (spread) {
  const better = spread.robust < 0.9 * spread.strict;     // only call it an improvement if it is a clear one (>10 %)
  kids.push(H2("6.7 How repeatable is calibration?"),
    ...figure("fig_calibration.png", `Figure 8. Switch points found in ${S.calibration_runs} independent calibration runs with each rule (dots); tighter columns mean a more repeatable rule.`, 640),
    P(better
      ? `Run to run, the old rule’s answers for the same operation differ by a typical factor of ${f1(spread.strict)}× between the largest and smallest result; the robust rule narrows this to ${f1(spread.robust)}×, so a calibration run can be trusted more.`
      : `Run to run, the old rule’s answers for the same operation differ by a typical factor of ${f1(spread.strict)}× (largest / smallest) and the robust rule’s by ${f1(spread.robust)}×. The two rules are therefore about equally repeatable on this machine, and equally accurate (Table 4): the large gain comes from calibrating each operation separately, not from the choice of fitting rule.`));
}

// 6.8 MPI model
if (model.length) {
  const cfgs = [...new Set(model.map((r) => r.ranks + "|" + r.local))];
  kids.push(H2("6.8 MPI cost model"),
    ...figure("fig_mpi_model.png", "Figure 9. Fitted cost model (lines) and measured times (points), including 4096² and 5120² images that were larger than any size used for fitting.", 640),
    table(["Configuration", "Series", "Slope b (ns / pixel)", "Error on unseen sizes", "MPI pays off above"],
      model.map((r) => [`${r.ranks} ranks${r.local === "openmp" ? " × OpenMP" : ""}`, r.series, f1(r.b_ns_per_px), f0(r.heldout_error_pct) + "%",
        r.series === "mpi" ? (r.crossover_px < 0 ? "never (this machine)" : f1(r.crossover_px / 1e6) + " Mpx") : "–"]),
      [2300, 1500, 1900, 1900, 2038], { size: 18 }),
    caption("Table 6. Model coefficients and prediction error for each configuration."),
    P(mpiNever
      ? `The model’s answer for this single PC is that MPI never overtakes OpenMP, which matches the measurements: MPI processes share the same memory bandwidth and pay for scatter and gather copies. The point of the model is to give the same quantitative answer on a different machine, such as a multi-node cluster, where MPI’s slope b could fall below OpenMP’s.`
      : `The model reports, per configuration, the image size above which MPI beats both single-process backends.`));
}

// 6.9 CUDA (expected)
if (cuda) {
  const rowsC = S.cuda_projection.rows, cs = S.cuda_projection.cases;
  kids.push(H2("6.9 Expected CUDA performance"),
    P("The GPU path of Canny (upload the colour image, run five bandwidth-bound kernels, download the 1-byte edge labels) was estimated with a memory-bandwidth model. Hysteresis stays on the CPU, so its measured time is added unchanged:"),
    eq(sub("T", "CUDA"), mr("(N) = "), new MathFraction({ numerator: [mr("4N")], denominator: [sub("B", "PCIe")] }), mr(" + "),
       new MathFraction({ numerator: [mr("b N")], denominator: [sub("B", "GPU")] }), mr(" + "), sub("t", "launch"), mr(" + "), sub("T", "hyst"), mr("(N)")),
    P(`Here N is the number of pixels (3N bytes of BGR input plus N bytes of labels cross the PCIe bus), b = ${S.cuda_projection.bytes_per_px} bytes of device-memory traffic per pixel for the five kernels, and T_hyst is the hysteresis time measured on this machine (Table 7). The two bandwidths are assumptions:`),
    table(["Scenario", "PCIe, pageable copy", "GPU memory bandwidth", "Representative hardware"],
      [["Typical (used in Table 8)", `${cs.typical.pcie_gbs} GB/s`, `${cs.typical.gpu_bw_gbs} GB/s`, "mid-range GPU, PCIe 3.0 x16"],
       ["Pessimistic", `${cs.pessimistic.pcie_gbs} GB/s`, `${cs.pessimistic.gpu_bw_gbs} GB/s`, "entry-level or laptop GPU"],
       ["Optimistic", `${cs.optimistic.pcie_gbs} GB/s`, `${cs.optimistic.gpu_bw_gbs} GB/s`, "recent GPU, PCIe 4.0"]],
      [2600, 2000, 2400, 2638], { size: 19 }),
    caption("Table 7. Assumptions of the CUDA model."),
    ...figure("fig_cuda_projection.png", "Figure 12. Canny time: measured sequential and OpenMP, and expected CUDA time (hatched bars). (CUDA values are expected, not verified on hardware.)", 640),
    table(["Image size", "Sequential (ms)", "OpenMP (ms)", "CUDA expected (ms)", "Expected speedup vs sequential", "Expected speedup vs OpenMP"],
      rowsC.map((r) => [`${r.size}×${r.size}`, f1(r.seq_ms), f1(r.omp_ms), f1(r.cuda_ms_typical),
        `${f1(r.speedup_vs_seq)}× (${f1(r.speedup_vs_seq_range[0])}–${f1(r.speedup_vs_seq_range[1])}×)`, `${f1(r.speedup_vs_omp)}×`]),
      [1300, 1500, 1400, 1700, 2238, 1500], { size: 18, highlight: (i) => i === rowsC.length - 1 }),
    caption("Table 8. Canny on the GPU. Sequential and OpenMP columns are measured; the CUDA column is expected. The range under the speedup is pessimistic–optimistic hardware."),
    P(`The GPU part (transfers plus kernels) is only ${f0(last(rowsC).gpu_part_ms_typical)} ms for a ${last(rowsC).size}² image; the ${f0(last(rowsC).hysteresis_ms)} ms of CPU hysteresis is the larger share, so the expected gain over sequential is about ${f1(last(rowsC).speedup_vs_seq)}× and over OpenMP about ${f1(last(rowsC).speedup_vs_omp)}×. Moving hysteresis onto the GPU would remove that limit and is the natural next optimisation. (Expected speedups from a memory-bandwidth model with the assumptions in Table 7; these are projections and have not been verified on GPU hardware.)`));
}

// 6.10 live demo
kids.push(H2("6.10 One-minute live demo"),
  P("scripts\\demo.ps1 runs the tests, a console demonstration (correctness table, dispatcher with heuristic and calibrated choices, speedup of all operations), MPI on 1, 2 and 4 processes, every algorithm on a real picture, and then draws the graphs below with Python. It finishes in well under a minute."),
  ...figure("demo/fig_demo_dispatch.png", "Figure 10. Live demo: sequential versus OpenMP time, with the sizes where each dispatcher chose the slower backend marked.", 640),
  ...figure("demo/fig_demo_mpi.png", "Figure 11. Live demo: MPI Canny on 1, 2 and 4 processes (2048×2048 image).", 560));

// ---- 7 Limitations
kids.push(H1("7. Limitations"),
  bullet([{ t: "CUDA figures are expected values. ", bold: true }, "The GPU timings in Section 6.9 come from a bandwidth model, not from measurements on a GPU; they depend on the assumed PCIe and memory bandwidth."]),
  bullet([{ t: "Not faster than OpenCV. ", bold: true }, ocv ? `On the same 4096² Canny pipeline OpenCV needs ${f0(ocv.cv_1thread_ms)} ms on one thread and ${f0(ocv.cv_nthreads_ms)} ms on ${S.threads} threads, against ${f0(ocv.pv_omp_ms)} ms for pv with OpenMP. OpenCV uses SIMD and cache-tiled kernels; pv’s contribution is the multi-backend design and the backend selection, not raw kernel speed.` : "OpenCV’s kernels are faster; pv’s contribution is the multi-backend design and backend selection."]),
  bullet([{ t: "Speedup saturates. ", bold: true }, "Memory bandwidth and the serial part of Canny’s hysteresis limit Canny to about 2× on eight threads, whatever the backend."]),
  bullet([{ t: "MPI scope. ", bold: true }, "Only Canny has an MPI path; MPI gains nothing over OpenMP on one PC, was not tested on a cluster, is not selected by Backend::Auto, and messages are limited to 2 GB per rank."]),
  bullet([{ t: "Cost model is approximate. ", bold: true }, "The straight-line model is fitted on 1024²–3072² images and extrapolated; its errors on unseen larger sizes are listed in Table 6 and are not small for every series, so its crossover sizes are estimates, not guarantees."]),
  bullet([{ t: "Calibration rules. ", bold: true }, spread && spread.robust < 0.9 * spread.strict
    ? `In this run the robust fitting rule is more repeatable than the original (${f1(spread.robust)}× against ${f1(spread.strict)}× run-to-run spread, Section 6.7), but the difference varied between our repeated runs and both rules are equally accurate; most of the benefit of calibration comes from using a separate switch point per operation.`
    : "The robust fitting rule is not measurably better than the original one on this machine (Section 6.7); the benefit of calibration comes from using a separate switch point per operation."]),
  bullet([{ t: "Measurement noise. ", bold: true }, "Everything was measured on one machine; run-to-run variation of 10–20% is typical, so small differences between strategies should not be over-interpreted."]));

// ---- 8 Conclusion
kids.push(H1("8. Conclusion"),
  P(`Parallel-Vision provides ${nOps} image operations behind a single API, each with sequential and OpenMP implementations, plus MPI (with halo exchange), a hybrid mode and a CUDA version of Canny. The outputs of all tested backends are identical. The evaluation shows that choosing between sequential and parallel execution matters: a blanket parallel strategy is ${f0((aOmp.geomean - 1) * 100)}% slower on average and up to ${f0(aOmp.worst)}× slower in the worst case, while the size-aware dispatcher stays within about ${f1((cal.geomean - 1) * 100)}% of optimal on average. Per-operation calibration and a measured MPI cost model make the selection adapt to the machine. The CUDA backend is expected to add a further gain on Canny (Section 6.9). Natural next steps are measuring and tuning the CUDA backend on a GPU, moving hysteresis onto the GPU, tile-fused kernels to reduce memory traffic, and testing the MPI cost model on a multi-node cluster.`));

// ---- Appendix: where everything is
kids.push(new Paragraph({ children: [new PageBreak()] }), H1("Appendix A. Scripts, results and where to find them"),
  P("All paths are relative to the project folder (Parallel-Vision). Every table and graph in Section 6 was produced by the scripts below; nothing was drawn or typed in by hand."),
  table(["Run this", "What it does", "Time"],
    [["powershell -ExecutionPolicy Bypass -File scripts\\demo.ps1", "LIVE DEMO: tests, console demo, MPI, all algorithms on a picture, graphs opened on screen", "about 30 s"],
     ["powershell -ExecutionPolicy Bypass -File scripts\\run_all.ps1", "FULL EVALUATION: all tests, every benchmark, all figures and summary.json", "10–15 min"],
     ["cd docs ; node build_report.js", "Rebuilds this report from results\\summary.json and the figures", "seconds"],
     ["build\\bin\\pv.exe canny in.jpg out.png", "Command-line tool (also: sobel prewitt laplacian blur median gray threshold equalize erode dilate)", "instant"],
     ["python scripts\\make_figures.py results", "Redraws all graphs from the CSV files", "seconds"]],
    [3900, 4638, 1100], { size: 18 }),
  spacer(),
  table(["Folder / file", "Contents"],
    [["results\\demo\\", "Live-demo output: demo_dispatch.csv, demo_ops.csv, mpi_demo.csv, images\\ (every algorithm’s output), fig_demo_*.png"],
     ["results\\bench.csv, ops.csv", "Canny sequential vs OpenMP by size; all operations sequential vs OpenMP"],
     ["results\\mpi.csv, mpi_hybrid.csv, mpi_model.csv", "MPI times for 1–8 ranks, hybrid times, and the fitted cost model with validation"],
     ["results\\dispatch.csv, dispatch.log", `${S.dispatch_cases} test cases scored for every strategy; calibrated switch points`],
     ["results\\calib_stability.csv", "Repeated calibration runs, old rule vs robust rule"],
     ["results\\stages.csv, cuda_projection.csv", "Canny stage times (measured); expected CUDA times (model, see Section 6.9)"],
     ["results\\vs_opencv.csv, vs_opencv.log", "Comparison with OpenCV and edge agreement (IoU)"],
     ["results\\fig_*.png", "All graphs used in this report"],
     ["results\\summary.json", "Every number quoted in this report"],
     ["tests\\, bench\\, src\\, include\\pv\\pv.hpp", "Test suites, benchmark programs, library source and public API"]],
    [3700, 5938], { size: 18 }));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Mohammed Mehdi Hedayati; Nathan Julian Dsouza",
  title: "Parallel-Vision: a C++ image-processing library with automatic backend selection",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: "1F3A5F" }, paragraph: { spacing: { before: 360, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: "2563EB" }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [run("Parallel-Vision · Hedayati & Dsouza · page ", { size: 18, color: "6B7280" }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "6B7280" })] })] }) },
    children: kids,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Parallel-Vision-Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024).toFixed(0) + " KB");
});
