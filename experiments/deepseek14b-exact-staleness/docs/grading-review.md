# Training grader review: September 26, 2026

The teammate's snag list describes a separate v3.2 sweep evaluator. Its implementation and frozen question-policy files were not supplied; their paths point to the teammate's machine. They are optional for this run, but necessary to claim exact reproduction of that evaluator. We independently examined the pinned training grader and added targeted regression cases. This is not an estimate of the grader's overall error rate.

## Confirmed bugs and corrections

The old training grader accepted `(1997,0)` versus `{1997,0}` and `A` versus `a`. The current `strict-final-box-v2-structure-case` policy rejects both. It checks tuple/set distinctions, tuple component structure and cardinality before calling Math-Verify. An additional case-preserving parse checks otherwise accepted answers containing uppercase symbols, because the pinned parser and comparator can fold symbol case. Case-sensitive parsing uses the bounded worker infrastructure; unsupported predictions have explicit reason codes.

The grader remains our package's adapter around pinned Math-Verify and latex2sympy. Official PrimeRL and Verifiers files are unchanged. It does not import the teammate's evaluator.

Grading errors matter for the staleness study: they change binary rewards, group-centered advantages and the fraction of tokens with zero direct GRPO signal. Unsupported output formats could occur at different rates as the policy changes. Keep one frozen reward policy across compared lag values and report unsupported/extraction outcomes alongside rewards; their counts alone do not establish accuracy.

## Snag-list findings

| Items | Evidence in this code | Remaining limit |
| --- | --- | --- |
| 1: large integers | Tested nearby integers are rejected, including 2050313 versus 2050312 | Not a new universal exact-numeric policy for every expression |
| 2–3: multi-part structure | Added tuple/set, nested-tuple and cardinality guards; tests reject missing set members and swapped labels | Does not reproduce registered labels or prove complete matching for every nested object |
| 4: intervals and matrices | Tested changed interval boundaries, matrix shapes and determinant/matrix distinctions are rejected | Explicit regression cases, not a general proof |
| 5: partial parsing | Tested trailing unknown LaTeX and malformed answers are rejected | No general full-expression consumption checker was added |
| 6–7: formatting | Tested coordinate formatting, text wrappers, fractions and scalar thousands separators work | Other variants can still produce false negatives |
| 8: symbols | Added case guards, including cancellation; unsupported `E_k` reference is explicitly rejected | General subscript support is incomplete; question-dependent imaginary-unit rules are absent |
| 9: `np.arcsin` | No special normalization added | Need matching training examples before claiming support or rewriting references |
| 10–11: units and equivalence | Tested equation rearrangement, factorial and binomial equivalence work | No question-specific unit/domain rules; the 66 formatting hints are not integrated |
| 12–14, 18–20: benchmark fixes | No mapping to this frozen DeepScaleR question set was supplied | Do not apply Minerva/Olympiad/AIME index patches to unrelated IDs or assume reference correctness |
| 15: extraction | Last balanced box after `</think>` remains required | Unboxed correct answers receive zero; changing this changes the reward policy |
| 16–17: conservative grading | Unsupported outcomes have reason codes; simple factorial/binomial probes pass | The actual seven teammate examples were not supplied. Unsupported still earns zero, not proof of mathematical incorrectness |
| 21: different graders | Confirmed; the two reported training false positives are addressed | No claim of v3.2 parity |
| 22: overall error rate | Unknown | Regression tests and no crashes do not establish zero errors |

## Dataset and run identity

Preparation was rerun on all 37,713 rows. The corrected manifest accepts 37,703 and excludes the same 10 question IDs; retained IDs and inclusion flags exactly match the prior manifest. Its canonical manifest identity is `2815cfdbe90623acfbdc2c581b5de9f473f168ee1351d152a6b3b903dec93176`; the JSON file SHA-256 is `ffd6ecf40730cd78a6eb00125e1269afe76a3bce32a6df57d8459522e4c00c8c`. This confirms selection stability, not correctness of every reference.

The cluster uses new `assets/train-manifest-v2.json` and a fresh output directory. The stopped run and old manifest remain intact; its rewards, partial optimizer work and queued rollouts are not reused. Reward identity is frozen before launch and included in the source/data contract.

Further reward-policy changes require a new identified run. An independent stratified audit of saved correct/incorrect/unsupported outcomes would be needed to estimate remaining errors, including whether they vary with response length or staleness. That audit is not required for this relaunch and has not been performed here.
