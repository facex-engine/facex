package facex

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"slices"
	"sort"
)

// ScoredProbe describes one labeled query against a fixed enrollment snapshot.
// ExpectedID is -1 for an unenrolled person. BestID is -1 for a tied best
// identity, an empty gallery or a query with no direction. For a single
// enrolled identity Margin is zero, matching IdentityGallery.Identify.
type ScoredProbe struct {
	ExpectedID, BestID int64
	Score, Margin      float64
}

// ScoreProbes computes exact per-person scores once for offline calibration.
// Labels for people absent from this gallery are recorded as unknown.
func (g *IdentityGallery) ScoreProbes(queries [][]float32, labels []int64) ([]ScoredProbe, error) {
	if err := g.validate(0, 0); err != nil {
		return nil, err
	}
	if len(queries) != len(labels) {
		return nil, errors.New("facex: expected a label per query")
	}
	enrolled := make(map[int64]bool, len(g.people))
	for _, p := range g.people {
		enrolled[p.id] = true
	}
	out := make([]ScoredProbe, len(queries))
	for i, q := range queries {
		if labels[i] < -1 {
			return nil, fmt.Errorf("facex: invalid label at query %d", i)
		}
		norm, err := vectorNorm(q)
		if err != nil {
			return nil, fmt.Errorf("query %d: %w", i, err)
		}
		v := ScoredProbe{ExpectedID: labels[i], BestID: -1}
		if !enrolled[labels[i]] {
			v.ExpectedID = -1
		}
		best, second := math.Inf(-1), math.Inf(-1)
		if norm != 0 {
			for j := range g.people {
				p := &g.people[j]
				value := math.Inf(-1)
				for row := range p.norms {
					value = max(value, p.cosine(row, q, norm))
				}
				if value > best {
					best, second, v.BestID = value, best, p.id
				} else {
					second = max(second, value)
				}
			}
		}
		if best == second {
			v.BestID = -1
		} else {
			v.Score = best
			if len(g.people) > 1 {
				v.Margin = best - second
			}
		}
		out[i] = v
	}
	return out, nil
}

// CalibrationCounts counts accepted probes, distinguishing both error types.
type CalibrationCounts struct {
	Correct, UnknownAccepted, KnownWrong int
}

func (a CalibrationCounts) plus(b CalibrationCounts) CalibrationCounts {
	return CalibrationCounts{a.Correct + b.Correct, a.UnknownAccepted + b.UnknownAccepted, a.KnownWrong + b.KnownWrong}
}

func (v ScoredProbe) count() CalibrationCounts {
	switch {
	case v.ExpectedID < 0:
		return CalibrationCounts{UnknownAccepted: 1}
	case v.ExpectedID != v.BestID:
		return CalibrationCounts{KnownWrong: 1}
	default:
		return CalibrationCounts{Correct: 1}
	}
}

// CalibrationResult contains inclusive thresholds and their calibration counts.
type CalibrationResult struct {
	MinScore, MinMargin float64
	Counts              CalibrationCounts
}

// A Fenwick tree over descending margin values. A prefix represents all rows
// above an inclusive margin cutoff among the currently active scores.
type marginCounts []CalibrationCounts

func (t marginCounts) add(i int, v CalibrationCounts) {
	for ; i < len(t); i += i & -i {
		t[i] = t[i].plus(v)
	}
}

func (t marginCounts) prefix(i int) (sum CalibrationCounts) {
	for ; i > 0; i -= i & -i {
		sum = sum.plus(t[i])
	}
	return sum
}

// Longest prefix satisfying three monotone upper limits, in logarithmic time.
func (t marginCounts) longest(limit CalibrationCounts) (int, CalibrationCounts) {
	var sum CalibrationCounts
	i := 0
	for step := 1 << (bits.Len(uint(len(t)-1)) - 1); step > 0; step >>= 1 {
		next := i + step
		if next >= len(t) {
			continue
		}
		v := sum.plus(t[next])
		if v.Correct <= limit.Correct && v.UnknownAccepted <= limit.UnknownAccepted && v.KnownWrong <= limit.KnownWrong {
			i, sum = next, v
		}
	}
	return i, sum
}

// ErrNoFeasibleThresholds means even score 1 and margin 2 exceed a budget.
var ErrNoFeasibleThresholds = errors.New("facex: no feasible threshold pair")

// FitThresholds maximizes correct identifications under two inclusive error
// counts on the supplied labeled probes. Ties minimize total errors, then
// prefer a higher score cutoff followed by a higher margin cutoff.
// Search takes O(n log n) time and O(n) space after scoring.
func FitThresholds(probes []ScoredProbe, maxUnknown, maxWrong int) (CalibrationResult, error) {
	if len(probes) == 0 || maxUnknown < 0 || maxWrong < 0 {
		return CalibrationResult{}, errors.New("facex: expected labeled probes and nonnegative budgets")
	}
	active := make([]ScoredProbe, 0, len(probes))
	margins := []float64{2}
	for i, v := range probes {
		if v.ExpectedID < -1 || v.BestID < -1 || !(v.Score >= -1 && v.Score <= 1 && v.Margin >= 0 && v.Margin <= 2) {
			return CalibrationResult{}, fmt.Errorf("facex: invalid scored probe %d", i)
		}
		if v.BestID >= 0 {
			active = append(active, v)
			margins = append(margins, v.Margin)
		}
	}
	sort.Slice(active, func(i, j int) bool { return active[i].Score > active[j].Score })
	sort.Slice(margins, func(i, j int) bool { return margins[i] > margins[j] })
	margins = slices.Compact(margins)
	tree := make(marginCounts, len(margins)+1)
	best := CalibrationResult{}
	found := false
	score := 1.
	for i := 0; ; {
		for i < len(active) && active[i].Score == score {
			v := active[i]
			rank := sort.Search(len(margins), func(j int) bool { return margins[j] <= v.Margin })
			tree.add(rank+1, v.count())
			i++
		}
		last, counts := tree.longest(CalibrationCounts{len(probes), maxUnknown, maxWrong})
		if last > 0 {
			// The shortest prefix retaining all achievable correct matches
			// also minimizes errors for this fixed score cutoff.
			first := 1
			if counts.Correct > 0 {
				pos, _ := tree.longest(CalibrationCounts{counts.Correct - 1, maxUnknown, maxWrong})
				first = pos + 1
			}
			counts = tree.prefix(first)
			if !found || counts.Correct > best.Counts.Correct ||
				(counts.Correct == best.Counts.Correct && counts.UnknownAccepted+counts.KnownWrong < best.Counts.UnknownAccepted+best.Counts.KnownWrong) {
				best = CalibrationResult{score, margins[first-1], counts}
				found = true
			}
		}
		if i == len(active) {
			break
		}
		score = active[i].Score
	}
	if !found {
		return CalibrationResult{}, ErrNoFeasibleThresholds
	}
	return best, nil
}
