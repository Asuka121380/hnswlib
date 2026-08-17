# Official JHQ reference boundary

The implementation is pinned to public JHQ commit
`1636e197a36871db4a79d66d22f9f9dd0fa51e31` under Apache-2.0.

The behavioral mapping is:

| Replay code | Pinned official source |
|---|---|
| Gaussian QR rotation | `jhq/jhqlib/impl/IndexJHQTrain.cpp::generate_qr_rotation_matrix` |
| Per-subspace statistics and centroid generation | `jhq/jhqlib/impl/IndexJHQSearch.cpp::analytical_gaussian_init` |
| Primary nearest-centroid encoding | `jhq/jhqlib/impl/IndexJHQTrain.cpp::encode_single_vector_with_scratch` |
| One-byte primary code per subspace | `jhq/jhqlib/IndexJHQ.cpp::compute_code_size` |
| Demo single-level configuration | `jhq/examples/demo_jhq_test.cpp`, `level_bits={8}` |

The replay intentionally does not claim bitwise parity for random artifacts.
The official implementation uses Faiss `float_randn`, platform LAPACK QR, and
libstdc++ `normal_distribution`; the replay uses NumPy MT19937 and NumPy QR.
Every experiment therefore persists and hashes its actual rotation and
centroids. Encode/decode behavior is exact relative to those persisted tables,
and the original-space error is conservatively bounded even when the stored
float32 rotation is not perfectly orthogonal.

This limitation is preferable to silently presenting a different platform RNG
sequence as the official one. A future official binary fixture can be added as
an additional gate without changing the replay or artifact formats.
