// Builds docs/Parallel-Vision-Report.docx from results/*.csv, results/summary.json and the figures.
//   cd docs && npm install && node build_report.js
const fs = require("fs");
const os = require("os");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  HeadingLevel, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber, PageBreak, Math: DMath, MathRun,
  MathFraction, MathSubScript, VerticalAlign,
} = require("docx");

const RES = path.join(__dirname, "..", "results");
const S = JSON.parse(fs.readFileSync(path.join(RES, "summary.json"), "utf8"));
const FONT = "Calibri";
const W = 9638; // content width in DXA (A4, 2 cm margins)
const f1 = (x) => x.toFixed(1), f2 = (x) => x.toFixed(2), f0 = (x) => Math.round(x).toString();
const num = (n) => n.toLocaleString("en-US");

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
function pngSize(file) { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function figure(file, caption, widthPx = 620) {
  const full = path.join(RES, file);
  const [w, h] = pngSize(full);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(full), transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [run(caption, { italics: true, size: 19, color: "4B5563" })] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, width, o = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({ alignment: o.align || AlignmentType.LEFT, children: [run(String(text), { size: 20, bold: o.bold, color: o.color })] })],
  });
}
function table(headers, rows, widths, opts = {}) {
  const hdr = new TableRow({ tableHeader: true, cantSplit: true,
    children: headers.map((h, i) => cell(h, widths[i], { bold: true, fill: "1F3A5F", color: "FFFFFF", align: i ? AlignmentType.CENTER : AlignmentType.LEFT })) });
  const body = rows.map((r, ri) => new TableRow({ cantSplit: true,
    children: r.map((c, i) => {
      const hl = opts.highlight && opts.highlight(ri, r);
      return cell(c, widths[i], { align: i ? AlignmentType.CENTER : AlignmentType.LEFT, fill: hl ? "DCFCE7" : ri % 2 ? "F9FAFB" : undefined, bold: hl });
    }) }));
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths, rows: [hdr, ...body] });
}
const spacer = () => new Paragraph({ spacing: { after: 160 }, children: [] });
const mr = (t) => new MathRun(t);
const eq = (...children) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 }, children: [new DMath({ children })] });
const sub = (b, s) => new MathSubScript({ children: [mr(b)], subScript: [mr(s)] });

// ---------------------------------------------------------------- data
const sc = S.scaling_4096;
const seq = sc[0], omp = sc[1];
const bestHyb = sc.filter((r) => r.config.startsWith("Hybrid")).sort((a, b) => a.ms - b.ms)[0];
const disp = Object.fromEntries(S.dispatch.map((d) => [d.strategy, d]));
const aSeq = disp["Always sequential"], aOmp = disp["Always OpenMP"], hz = disp["pv heuristic"], cal = disp["pv calibrated"];
const bestSizes = S.best_by_size.filter((b) => b.speedup > 1);
const peak = S.best_by_size.reduce((a, b) => (b.speedup > a.speedup ? b : a));
const cpu = os.cpus()[0].model.replace(/\s+/g, " ").trim();
const ocv = S.opencv_4096;
const sw = S.switch_points;
const side = (px) => Math.round(Math.sqrt(px));

// ---------------------------------------------------------------- content
const kids = [];

// title page
kids.push(
  new Paragraph({ spacing: { before: 1800, after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Parallel-Vision", font: FONT, size: 72, bold: true, color: "1F3A5F" })] }),
  new Paragraph({ spacing: { after: 360 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "A C++ image-processing library that chooses between sequential and parallel execution", font: FONT, size: 30, color: "374151" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3A5F", space: 8 } }, children: [] }),
  new Paragraph({ spacing: { before: 300, after: 60 }, alignment: AlignmentType.CENTER,
    children: [run("Parallel Programming Laboratory — Semester 5 Mini-Project", { size: 26, bold: true })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [run("Technologies: C++17 · OpenMP · MPI (MS-MPI) · CMake", { size: 22, color: "4B5563" })] }),
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

// 1 Introduction
kids.push(H1("1. What is Parallel-Vision?"),
  P("Parallel-Vision (pv) is a C++17 shared library for image-processing algorithms such as grayscale conversion, Gaussian blur, Sobel edge detection, thresholding and the full Canny edge detector. Every operation exists in a sequential version and in parallel versions (OpenMP on one machine, MPI across processes, and a hybrid of the two). The library's distinguishing feature is that the caller does not have to decide which one to use: by default pv looks at the operation and the image size and picks the backend that is expected to be fastest."),
  P("The motivation is a common mistake in parallel programming: parallelism is not free. Starting threads, splitting work and exchanging data cost time, so on small inputs a parallel program can be many times slower than a simple loop, while on large inputs a sequential program leaves most of the machine idle. A library that always runs in parallel, or never does, loses either way. pv measures where the break-even point lies and uses it."),
  rich({ t: "Goals. ", bold: true }, "(1) one API with several interchangeable backends; (2) identical results from every backend; (3) automatic, explainable backend selection; (4) measurable evidence, reproducible with a single script."));

// 2 Architecture
kids.push(H1("2. How the library is organised"),
  ...figure("fig_architecture.png", "Figure 1. Architecture: one public API, a dispatcher, and interchangeable backends sharing the same row kernels."),
  H2("2.1 Operations provided"),
  table(["Operation", "Function", "Sequential", "OpenMP", "MPI"],
    [["Grayscale (BGR → gray)", "to_gray", "yes", "yes", "inside Canny"],
     ["Gaussian blur (separable)", "gaussian_blur", "yes", "yes", "inside Canny"],
     ["Sobel gradient magnitude", "sobel", "yes", "yes", "inside Canny"],
     ["Threshold, Otsu’s method", "threshold, otsu_threshold", "yes", "yes", "–"],
     ["Canny edge detector", "canny", "yes", "yes", "yes (mpi_canny)"]],
    [2900, 2400, 1300, 1300, 1738]),
  spacer(),
  H2("2.2 Using the library"),
  P("The library has no dependency on OpenCV; images are a small struct with a byte buffer. The command-line tool uses OpenCV only to read and write image files."),
  ...code([
    '#include "pv/pv.hpp"',
    '',
    'pv::Image edges = pv::canny(img);                       // Backend::Auto: pv decides',
    'pv::Image a = pv::canny(img, 50, 150, 1.4, pv::Backend::Sequential);',
    'pv::Image b = pv::canny(img, 50, 150, 1.4, pv::Backend::OpenMP);',
    '',
    'pv::calibrate();                                         // optional: measure this machine',
    'pv::save_calibration("pv_calibration.txt");              // reuse next run with load_calibration()',
  ]));

// 3 Sequential and parallel in one library
kids.push(H1("3. Sequential and parallel code in the same library"),
  P("Each algorithm is written once, as a function that processes a range of image rows. A tiny helper decides how that function is driven:"),
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
  P("Because the arithmetic is shared, the sequential and parallel versions cannot drift apart. This is checked, not assumed: the test-suite requires the OpenMP result to equal the sequential result bit for bit on every operation, and the MPI result to equal it for 1, 2, 3, 4 and 8 processes on six different image shapes (Section 6.1)."),
  P("The same row kernels serve the MPI backend: each process simply runs them on its own strip of rows."));

// 4 Parallel design
kids.push(H1("4. The parallel backends"),
  H2("4.1 OpenMP"),
  P("Rows are distributed over threads with a static schedule. Blur is split into a horizontal and a vertical pass with a barrier between them; the gradient and non-maximum-suppression passes are independent per row. Scratch buffers are left uninitialised and filled by the threads that use them, which avoids a serial zero-fill of hundreds of megabytes. Hysteresis thresholding in Canny needs connectivity across the whole image, so only its seed scan is parallel; the flood fill is serial but touches only edge pixels."),
  H2("4.2 MPI with halo exchange"),
  P("The image is cut into horizontal strips, one per process. Rank 0 scatters the strips; neighbouring ranks then exchange halo rows so that every owned row is computed from exactly the same neighbourhood as in the sequential run. The halo depth is the Gaussian radius plus one row each for the gradient and for non-maximum suppression:"),
  eq(mr("d = ⌈ 3σ ⌉ + 2")),
  P("For the default σ = 1.4 this is 7 rows. Each rank runs blur, gradient and suppression on its extended strip and keeps only its own rows; the compact edge labels (1 byte per pixel: none, weak, strong) are gathered on rank 0, which performs the global hysteresis step. Ranks may also use OpenMP inside their strip, giving a hybrid MPI + OpenMP mode."),
  ...code([
    'pv::mpi_init(&argc, &argv);',
    'pv::Image out = pv::mpi_canny(in, 50, 150, 1.4, pv::Backend::OpenMP);  // collective; result on rank 0',
    'pv::mpi_finalize();',
  ]),
  H2("4.3 Measuring parallel performance"),
  P("With T_seq the sequential time and T_p the time using p workers, speedup and efficiency are"),
  eq(sub("S", "p"), mr(" = "), new MathFraction({ numerator: [sub("T", "seq")], denominator: [sub("T", "p")] }),
     mr(",      "), sub("E", "p"), mr(" = "), new MathFraction({ numerator: [sub("S", "p")], denominator: [mr("p")] })),
  P("If a fraction f of the work can be parallelised, Amdahl’s law bounds the speedup:"),
  eq(sub("S", "p"), mr(" ≤ "), new MathFraction({ numerator: [mr("1")], denominator: [mr("(1 − f) + "), new MathFraction({ numerator: [mr("f")], denominator: [mr("p")] })] })));

// 5 Dispatcher
kids.push(H1("5. Choosing the backend automatically"),
  P("When the caller passes Backend::Auto (the default), the dispatcher compares the image size N = width × height with a per-operation switch point N*:"),
  eq(mr("backend = "), mr("OpenMP  if  N ≥ N*(op)  and more than one thread is available;  otherwise  Sequential")),
  P("Two ways of setting the switch points are provided and compared in Section 6:"),
  bullet([{ t: "Heuristic: ", bold: true }, "a fixed default of 100,000 pixels for every operation."]),
  bullet([{ t: "Calibrated: ", bold: true }, "pv::calibrate() times sequential against OpenMP on synthetic images from 16² to 1024² on the current machine, and sets each operation’s switch point to the smallest size from which OpenMP wins clearly at every larger size. Results can be saved and reloaded (pv --calibrate, pv --use-calibration)."]),
  P("MPI is deliberately not selected automatically: it is a collective operation that every process must call, so the caller opts in with mpi_canny or Backend::MPI."));

// 6 Results
kids.push(new Paragraph({ children: [new PageBreak()] }), H1("6. Results"),
  H2("Test setup"),
  table(["Item", "Value"],
    [["Machine", `${cpu}, ${os.cpus().length} logical threads, ${Math.round(os.totalmem() / 2 ** 30)} GB RAM`],
     ["System", "Windows 11, MinGW-w64 g++ 16.1 (MSYS2 UCRT64), -O3, OpenMP 5.2, MS-MPI 2.0"],
     ["Workload", "Canny pipeline (gray → Gaussian σ = 1.4 → Sobel → suppression → hysteresis 50/150) on synthetic images from 32² to 4096² and a 3840×2400 photograph"],
     ["Method", "Best of 5 runs after one warm-up; MPI timings include scatter, halo exchange and gather (data starts and ends on rank 0)"],
     ["Reproduce", "scripts\\run_all.ps1 regenerates every number and figure in this section"]],
    [1700, 7938]),
  spacer());

// key results box
const keyPts = [
  [{ t: "Same answer everywhere. ", bold: true }, "OpenMP, MPI (1, 2, 3, 4, 8 ranks) and hybrid output is bit-identical to the sequential output; edges also agree with OpenCV’s Canny (overlap IoU " + f2(S.iou) + ")."],
  [{ t: `Up to ${f2(peak.speedup)}× faster than sequential `, bold: true }, `(${peak.size}² image, ${f1(peak.seq_ms)} ms → ${f1(peak.best_ms)} ms); ${f2(omp.speedup)}× on a 4096² image with ${omp.workers} threads.`],
  [{ t: `Automatic choice costs only about ${f0((Math.max(hz.geomean, cal.geomean) - 1) * 100)}% on average `, bold: true }, `versus the best possible backend, while always using OpenMP is ${f0((aOmp.geomean - 1) * 100)}% slower on average and always using sequential is ${f0((aSeq.geomean - 1) * 100)}% slower.`],
  [{ t: "Avoids disasters. ", bold: true }, `Worst single case: ${f1(hz.worst)}× slower than optimal with pv, against ${f0(aOmp.worst)}× for naive always-parallel on tiny images.`],
  [{ t: "Right choice almost every time. ", bold: true }, `pv picked a backend within 10% of the best in ${f0(hz.within10)}% (heuristic) and ${f0(cal.within10)}% (calibrated) of ${S.dispatch_cases} test cases.`],
];
kids.push(new Table({
  width: { size: W, type: WidthType.DXA }, columnWidths: [W],
  rows: [new TableRow({ children: [new TableCell({
    width: { size: W, type: WidthType.DXA }, margins: { top: 140, bottom: 100, left: 220, right: 220 },
    shading: { type: ShadingType.CLEAR, fill: "ECFDF5", color: "auto" },
    borders: { top: { style: BorderStyle.SINGLE, size: 12, color: "059669" }, bottom: border, left: border, right: border },
    children: [new Paragraph({ spacing: { after: 100 }, children: [run("Key results", { bold: true, size: 26, color: "065F46" })] }), ...keyPts.map(bullet)],
  })] })],
}), spacer());

// 6.1 correctness
kids.push(H2("6.1 Correctness"),
  table(["Check", "Result"],
    [["OpenMP vs sequential (all operations, random 257×131 image)", "bit-identical"],
     ["MPI vs sequential (1, 2, 3, 4, 8 ranks × 6 image shapes incl. 100×9 and 16×16)", "bit-identical (30 / 30 runs)"],
     ["Hybrid MPI + OpenMP vs sequential (4 ranks × 2 threads × 6 image shapes)", "bit-identical (6 / 6 runs)"],
     ["Calibration save / load round-trip, dispatcher fallback rules", "pass"],
     [`Edge agreement with OpenCV Canny on a 3840×2400 photo (${num(S.edge_px_pv)} vs ${num(S.edge_px_cv)} edge pixels)`, `IoU = ${f2(S.iou)}`]],
    [6800, 2838]),
  spacer(),
  ...figure("fig_edges.png", "Figure 2. Canny edges produced by pv (automatic backend) on a 3840×2400 photograph."));

// 6.2 speedup
kids.push(H2("6.2 Speedup over sequential"),
  ...figure("fig_scaling.png", "Figure 3. Left: Canny time versus image size. Right: speedup at 4096×4096 for each parallel configuration."),
  table(["Configuration", "Workers", "Time (ms)", "Speedup", "Efficiency"],
    sc.map((r) => [r.config, r.workers, f1(r.ms), f2(r.speedup) + "×", f0(r.efficiency * 100) + "%"]),
    [3300, 1400, 1600, 1600, 1738], { highlight: (i, r) => r === sc.reduce((a, b) => (b.ms < a.ms ? b : a)) || false }),
  P("Table 1. 4096×4096 Canny. Efficiency = speedup / workers.", { run: { italics: true, size: 19, color: "4B5563" }, p: { spacing: { before: 60, after: 160 } } }),
  P(`Parallel execution pays off from about 256² pixels upward. Every parallel backend shows the same behaviour: below the break-even size the thread or process start-up cost dominates; above it, runtime drops to roughly ${f0(100 / omp.speedup)}% of the sequential time. MPI with ${sc.find((r) => r.config === "MPI 8 ranks").workers} ranks reaches ${f2(sc.find((r) => r.config === "MPI 8 ranks").speedup)}× and the hybrid configurations match OpenMP (${f2(bestHyb.speedup)}×), confirming that the same row kernels scale across both programming models.`));

// 6.3 dispatcher
kids.push(H2("6.3 The dispatcher at work"),
  P(`To test the automatic selection, ${S.dispatch_cases} cases (5 operations × 14 image sizes, including non-square shapes that were not used for calibration) were timed with both backends. The “oracle” is the faster of the two in each case. Four strategies are compared against it:`),
  ...figure("fig_dispatch.png", "Figure 4. Left: average slowdown relative to the best possible choice (geometric mean over all cases). Right: the single worst case."),
  table(["Strategy", "Total time vs. best", "Average slowdown", "Worst case", "Within 10% of best"],
    S.dispatch.map((d) => [d.strategy, f0(d.total_vs_oracle_pct) + "%", f2(d.geomean) + "×", f1(d.worst) + "×", f0(d.within10) + "%"]),
    [2700, 1750, 1750, 1500, 1938], { highlight: (i) => i >= 2 }),
  P("Table 2. Backend-selection strategies over all test cases (lower is better; highlighted rows are pv).", { run: { italics: true, size: 19, color: "4B5563" }, p: { spacing: { before: 60, after: 160 } } }),
  P(`A fixed choice is wrong somewhere: always-OpenMP is catastrophic on tiny images (up to ${f0(aOmp.worst)}× slower), and always-sequential wastes the machine on large ones (${f2(S.large_always_seq_slowdown)}× slower on images of one megapixel and above). The dispatcher avoids both failure modes.`),
  H2("6.4 Where the crossover is"),
  ...figure("fig_crossover.png", "Figure 5. Sequential versus OpenMP time per operation. The curves cross at different sizes, which is why the switch point is per operation.", 640),
  table(["Operation", "Heuristic switch point", "Calibrated switch point", "≈ square image"],
    [["Grayscale", "100,000 px", num(sw.grayscale) + " px", side(sw.grayscale) + "²"],
     ["Gaussian blur", "100,000 px", num(sw.blur) + " px", side(sw.blur) + "²"],
     ["Sobel", "100,000 px", num(sw.sobel) + " px", side(sw.sobel) + "²"],
     ["Threshold", "100,000 px", num(sw.threshold) + " px", side(sw.threshold) + "²"],
     ["Canny", "100,000 px", num(sw.canny) + " px", side(sw.canny) + "²"]],
    [2300, 2400, 2700, 2238]),
  P("Table 3. Switch points measured on the test machine. Cheap per-pixel operations need larger images before threads pay off than heavy ones.", { run: { italics: true, size: 19, color: "4B5563" }, p: { spacing: { before: 60, after: 160 } } }));

// 7 Limitations
kids.push(H1("7. Limitations"),
  bullet([{ t: "Not faster than OpenCV. ", bold: true }, `On the same 4096² Canny pipeline OpenCV needs ${f0(ocv.cv_1thread_ms)} ms on one thread and ${f0(ocv.cv_nthreads_ms)} ms on ${S.threads} threads, against ${f0(ocv.pv_omp_ms)} ms for pv with OpenMP. OpenCV uses SIMD and cache-tiled kernels; pv’s contribution is the backend selection and the multi-backend design, not raw kernel speed.`]),
  bullet([{ t: "Speedup saturates near 2×. ", bold: true }, "The passes are memory-bandwidth bound and hysteresis is partly serial, so (Amdahl) eight workers give about 1.9× rather than 8×."]),
  bullet([{ t: "MPI does not beat OpenMP on one PC. ", bold: true }, "Processes on one machine share the same memory bandwidth and pay for scatter/gather copies, so MPI reaches at most the OpenMP speedup here. It was not tested on a cluster, and Backend::Auto never selects it."]),
  bullet([{ t: "Calibration is not clearly better than the heuristic. ", bold: true }, `Both give about ${f2(hz.geomean)}× average slowdown here; calibration should matter more on other hardware, which was not tested.`]),
  bullet([{ t: "Scope. ", bold: true }, "CUDA is not implemented (the backend interface leaves room for it). Only Canny has an MPI path. Results come from one machine and vary by 10–20% between runs."]));

// 8 Conclusion
kids.push(H1("8. Conclusion"),
  P(`Parallel-Vision shows, with reproducible measurements, that choosing between sequential and parallel execution matters: a blanket parallel strategy is ${f0((aOmp.geomean - 1) * 100)}% slower on average and up to ${f0(aOmp.worst)}× slower in the worst case, while a size-aware dispatcher stays within about ${f0((Math.max(hz.geomean, cal.geomean) - 1) * 100)}% of optimal on average. The library offers sequential, OpenMP, MPI and hybrid backends behind a single API with identical results, and the whole evaluation can be regenerated with one script. Natural next steps are a CUDA backend, tile-fused kernels to reduce memory traffic, and a communication-cost model that lets the dispatcher decide when MPI is worthwhile on a multi-node cluster.`),
  H2("Reproducing these results"),
  ...code([
    'powershell -ExecutionPolicy Bypass -File scripts\\run_all.ps1',
    'cd docs && npm install && node build_report.js        # regenerates this document',
  ]),
  P("Full build and usage instructions are in README.md."));

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
