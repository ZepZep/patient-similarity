
## Config
* Record filter - 11
    * `Fall, Fr01, Fr02, ..., Fr10`
    * `{patient: [str]}`
    * input `parts.feather`
    * `Fall.feather`
    * `RecordFilterComputer.compute(method="rnn")`
* Vectorization - 3-9
    * `Vtfi, Vd2v`
    * `{patient: Matrix[n_parts, e_dim]}`
    * `Vtfi050-Fall.feather`
    * `PartVectorizerComputer.compute(method="tfi", dim=50, patterns=["F*.feather"])`
* Matrix similarity - 4
    * `Mrv2, MdCo, Mdna,`
    * Full: `{patient: Matrix[n_pac, n_pac]}` 3000x3000
    * Relevant: `{patient: Matrix[n_rel, n_rel]}` 50x50
    * `Rrv2-Vtfr050-Fall.npy`
    * `MatSimComputer.compute(method="rv2", patient_subset=set(), patterns=["V*.npy"])`

* Embedding - 4
    * ???


