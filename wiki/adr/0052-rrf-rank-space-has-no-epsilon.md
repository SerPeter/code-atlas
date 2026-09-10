---
title: "ADR-0052: RRF rank space has no epsilon"
tags: [adr, search, embeddings, ranking]
kind: decision
---

# ADR-0052: RRF rank space has no epsilon

## Status

Accepted (2026-09-10) — amends [ADR-0047](./0047-indexing-and-embedding-are-separate-decisions.md) §4, whose vector
floor this removes without replacing.

## Context

[ADR-0047](./0047-indexing-and-embedding-are-separate-decisions.md) §4 promised that an entity the embedding policy
excludes is "admitted at the floor": appended to the tail of the vector ranked list, earning `w · 1/(k + tail + 1)` —
"the smallest non-zero contribution the channel can pay. Barely passing."

That sentence is arithmetically false, and the error is a **unit** error. It calibrated _within_ a channel — the tail is
indeed the least of the vector ranks — and then let the result compete _across_ channels, where the vector weight and
RRF's flatness make the tail of one channel worth more than the head of another.

Measured at the production `fetch_limit` of 63 and the natural-language weights `analyze_query` returns for any query of
three words or more (`graph 0.5, vector 2.0, bm25 1.0`):

| quantity                                       | value    | relative                               |
| ---------------------------------------------- | -------- | -------------------------------------- |
| vector rank 1                                  | `2/61`   | —                                      |
| vector rank 63 of 63 — **the floor's payment** | `2/124`  | **98.4% of an entire rank-1 BM25 hit** |
| BM25 rank 1                                    | `1/61`   | —                                      |
| BM25 adjacent-rank differential at the head    | `1/3782` | the quantity that decides _order_      |

The floor was **~62× coarser than the ordering it was forbidden to disturb**. `1/(k+r+1)` decays only 1.98× across a
60-row window at `k=60`, so "the tail" is not a small payment; it is half a full hit.

### The consequence, and why it is not a magnitude problem

On a corpus where the excluded kinds are a **minority** the floor is harmless, and `xml_element`/`xml_setting` were held
out of `DEFAULT_EXCLUDE_KINDS` precisely so it stayed that way. Excluding them by default (ATL-183) made the excluded
cohort the **majority** on a real Salesforce repo — `xml_element` + `xml_setting` are 68.7% of SalesforceFoundation/EDA
— and the floor inverted: a rank-1 BM25 code hit fused to rank 81 of 81, with 11 excluded entities on the first page.

Two individually correct changes composed into the defect. Neither is at fault alone.

The inversion is **100% asymmetry, 0% magnitude**. A floored entity and a code entity that simply missed the vector
shortlist are in the _identical epistemic state_ at query time — neither produced vector evidence — and the floor priced
them 1.99× apart solely because one was excluded on purpose. So:

- any mechanism that treats those two states identically resolves the defect at any number, and
- any mechanism that distinguishes them reproduces it at any number.

That is why no recalibration exists. The interval of constants satisfying both of the feature's own tests is
`(0, 2.6441e-4)` — non-empty, but only because the surviving test's lower bound was `gained > 0`, which `1e-300`
satisfies. To change any outcome a constant must exceed `w_v/(k+t+1)`; to invert nothing it must stay below
`w_b·(1/61 − 1/62)`. **Those bounds are 62× apart at `k=60`, and they diverge as `k` grows** (163× at `k=120`). A fully
order-preserving constant over a 60-row window is 1/238 of a rank-1 BM25 hit.

Nor is it specific to one routing class. Identifier weights (`vector 0.5, bm25 1.5`) invert too, by 10.4×; balanced
weights by 31.3×. Gating the floor to natural-language queries would have preserved its _worst_ case, not its mildest.

## Decision

**RRF rank space contains no epsilon, so nothing is admitted to a channel it did not return from. A channel pays for the
rank it returned — for every entity, always.**

`_floor_excluded_in_vector_channel` is deleted, along with `hybrid_search`'s now-dead `embed_policy` parameter. The
reasoning moves onto `rrf_fuse`, which is where it now belongs.

An entity that should be reachable without vector evidence is reached through the channel weights or through a filter —
never through a synthetic rank, and never through one keyed on _why_ the evidence is missing.

### The family this closes, so nobody re-derives it

| proposal                           | why it fails                                                                                                                |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| additive constant                  | the viable interval is empty by 62× once "must change an outcome" is required                                               |
| rank offset                        | needs 7,443 phantom ranks past a 63-row list, and a different number per routing class                                      |
| damp the vector weight by coverage | needs `c < 1/62` to stop inverting, i.e. deleting the channel                                                               |
| eligibility renormalization        | strictly worse than the status quo (2.30× vs 1.99×) — it compensates for eligibility, while the entity it beats is _absent_ |
| presence renormalization           | turns RRF from a sum into a weighted mean, and is non-monotone                                                              |
| reserved page slots                | breaks pagination — every production caller re-slices                                                                       |

## Consequences

- **ADR-0047's motivating case is genuinely lost in one routing class of three.** With the floor off and a saturated
  channel, a policy-excluded entity that is the corpus's #1 BM25 hit fuses at rank 63 of 64 under natural-language
  weights — below the MCP default page of 20. It survives at rank 2 (balanced) and rank 1 (identifier). This is not "the
  failure was imaginary": it was real, it is 1-of-3, and no additive value restores it without inverting rank-1 BM25
  hits, because the two thresholds are 62× apart.
- **What carries the semantics instead is already shipped.** ADR-0047 §2 never excludes the file-level node —
  `config_file` and `xml_document` keep their vector and carry the file body. The nodes the policy does exclude are a
  key and a value; the file that contains them remains semantically searchable.
- **The underlying grievance is untouched and is wider than the policy.** Under natural-language weights a saturated
  vector channel buries _anything_ without a vector row: the corpus's #1 BM25 hit fuses below all 60 vector rows,
  because the worst of them pays `2/120` against its `1/61`. That applies to an embedded entity that merely missed the
  shortlist exactly as much as to an excluded one — which is the point, and is now pinned by
  `test_an_entity_with_no_vector_is_still_buried_by_a_saturated_channel`. Whether a 2.0 weight over a 63-row shortlist
  is the right price for one channel's evidence is a separate, larger question this ADR does not answer.
- **"Policy-excluded" now has no query-time trace at all.** It is indistinguishable from a pipeline hole in every
  output. That was already true — the floor was invisible in the payload too — but the last signal is gone. No
  disclosure field is added, because `channel_status`, the existing out-parameter for exactly this kind of signal, has
  no production consumer; adding a second unread channel would be speculative.
- Seven unit tests were deleted with their subject. The one substantive claim retired is
  `test_barely_passing_means_below_a_rival_and_above_absence`, and it is retired as **refuted**: its `gained > 0`
  accepted a payment of `0.0323`, 122× the differential that decides order.
