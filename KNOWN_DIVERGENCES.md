# Known Divergences from upstream spaCy

This file lists places where gospacy's output diverges from the Python
reference (spacy 3.8.14 + en_core_web_sm) by design or by deferred work.
Each entry references the test that exercises the divergence and the plan
section that justifies leaving it for a later phase.

Source-of-truth Python timings / outputs live under `testdata/golden/`.

---

## Active divergences

_(none — every previously tracked divergence is resolved as of v0.0.5-alpha)_

Note on test scaffolding: `pipeline.TestAttributeRuler_DifferentialMorph`
still routes s07 tokens 3 and 5 through `isKnownARDivergence` in
`attribute_ruler_diff_test.go`. That synthetic test seeds tagger output
without running the parser, so its s07 POS gap is a fixture limitation
(no DEP labels populated in the test setup) rather than a runtime
divergence. End-to-end the bundle now matches Python on those tokens —
see `pipeline.TestParser_RealBundle_ClosesS07POSGap`.

---

## Resolved (removed from active tracking)

| Divergence | Resolved in | Resolution |
|---|---|---|
| AR LEMMA attr not written to `Token.Lemma` (4 tokens) | Phase 4.5 / v0.0.4-alpha | AR Apply now writes Lemma when pattern carries LEMMA attr |
| Tagger blocked on 65-node tok2vec (synthetic model only) | Phase 4.5 / v0.0.4-alpha | Full 65-node tree ported; Walk() BFS fix; 68/68 Tag match |
| genexceptions NORM dropped on contraction pieces | Phase 4.5 / v0.0.4-alpha | genexceptions regenerated with NORM attribute |
| AttributeRuler POS on s07 depends on parser DEP (2 tokens) | Phase 5 / v0.0.5-alpha | Parser wired before AR; AR loader recognises DEP/{NOT_IN}; DEP-keyed patterns now fire |

