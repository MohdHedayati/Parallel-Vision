# References

Recent papers used as background for Parallel-Vision. Each was downloaded from the source listed and its title checked
against the file. Each supports one design decision; none of them validates our measurements.

| File | Citation | Used for |
|---|---|---|
| `Souza2026_cpu-gpu-image-filtering.pdf` | V. A. Souza, "Comparative Performance Analysis of GPU and CPU Processing for Digital Filtering Operations on Images Captured by the NAO Humanoid Robot," *25th Workshop on Performance of Computer and Communication Systems (WPerformance 2026)*, CSBC 2026, Gramado, Brazil. | The best processor depends on the operation, batch size and image resolution, which motivates our per-operation switch points. |
| `Abram2026_cross-architecture-autotuning.pdf` | H. Abram, N. Papadopoulou, J. Domke, M. Pericas, "Cross-Architecture Autotuning for Single-Source Heterogeneous Programming Models," *Proc. 40th ACM International Conference on Supercomputing (ICS '26)*, Belfast, 2026, pp. 856-867. DOI 10.1145/3797905.3800519 | Good settings differ between CPUs and GPUs, which motivates calibrating on the machine we run on. |
| `Collom2025_persistent-partitioned-mpi-stencil.pdf` | G. Collom, J. Burmark, O. Pearce, A. Bienz, "Persistent and Partitioned MPI for Stencil Communication," arXiv:2508.13370 [cs.DC], 18 Aug 2025. | Halo exchange for stencils (our MPI strips), and persistent/partitioned MPI as future work. |

Sources:
- Souza: https://sol.sbc.org.br/index.php/wperformance/article/download/43184/42950/
- Abram et al.: https://eprints.gla.ac.uk/384364/ (open access, CC BY 4.0 as stated on that page)
- Collom et al.: https://arxiv.org/abs/2508.13370

The PDFs are copyrighted by their authors/publishers. They are kept here as study copies for the lab submission; if this
repository is made public, check each license first or replace the PDFs with the links above.
