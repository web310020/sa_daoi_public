# Load-D reference results

`load_D_blocks.csv` contains one row per method and deployment block for the main
comparison. It has seven methods and five blocks per method (35 rows total).

The independent statistical unit is the deployment block (`n=5`). Each block contains
30 sequential episodes and 30 distinct environment seeds. These rows intentionally omit
local wall-clock timing because the paper's runtime carrier was measured separately under
a one-logical-CPU protocol.

The reference file was derived from the frozen camera-ready battery by selecting the
floor-preserving SA-DAoI implementation and the six paper baselines. Its SHA-256 is listed
in `SHA256SUMS`.

