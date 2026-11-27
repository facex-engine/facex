# Fit two identification thresholds with separate error budgets

For a fixed Go identity gallery, the score of a person is the largest cosine similarity over their enrollment photos. The margin is the best person's score minus the best other person's score. ScoreProbes calculates both exactly. It also handles ties, empty galleries, single-person galleries and zero queries according to Identify.

```go
snapshot, revision := live.Snapshot()
probes, err := snapshot.ScoreProbes(calibrationQueries, expectedIDs)
if err != nil {
    return err
}
fit, err := facex.FitThresholds(probes, maxUnknownAccepted, maxKnownWrong)
if err != nil {
    return err
}
ids, err := snapshot.IdentifyBatch(ctx, heldOutQueries, fit.MinScore, fit.MinMargin, workers)
```

Use a fixed snapshot and checkpoint for scoring, fitting and independent evaluation. Record the revision with the thresholds. An ID absent from enrollment is counted as unknown. The two budgets count accepted unknown queries and accepted known queries assigned to another person. Use a representative mix of both kinds, keep visits independent across calibration and evaluation, and report both error rates and denominators on held-out visits before deploying. A zero calibration count alone cannot establish a deployment error rate.

## Why the sweep is exact

An accepted query must have a unique candidate, score at least the score cutoff, and margin at least the margin cutoff. As the cutoffs vary, acceptance changes only at observed scores and margins. The maximum legal cutoffs (1 and 2) cover the empty-selection case when attainable. A false match at both maxima can make an error budget infeasible.

Sweep distinct scores in descending order, inserting all tied scores together. Coordinate-compress the distinct margins in descending order and maintain three prefix counts with a Fenwick tree. At a fixed score, relaxing the margin extends one prefix; each count is monotone. Binary lifting finds the longest prefix satisfying both budgets and hence the maximum attainable number of correct matches. Among prefixes with that many correct matches, the shortest incurs the fewest errors and has the highest margin cutoff. Compare these choices across scores by correct matches, total errors, then stricter score and margin. This enumerates all distinct outcomes needed for the optimum including inclusive threshold boundaries.

For n labeled probes, fitting takes O(n log n) time and O(n) extra space. Generating the exact top-two scores from raw embeddings is a separate offline operation. Reuse the scored probes to explore different budgets without repeating gallery lookup.

## Reproduction

From go/facex run `go test -race ./...` for exhaustive small-set checks and deployment parity, and `FACEX_GALLERY_SCALE=1 go test -v -run '^TestCalibrationMillion$' -count=1` for the seeded million-score benchmark. The benchmark times fitting on generated scores, replays both error budgets and compares with the best score-only cutoff on the same calibration records.
