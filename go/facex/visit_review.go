package facex

import (
	"errors"
	"math"
	"sort"
)

// ReviewVisit is one accepted ID decision for a labeled visit. Higher priority
// is reviewed first. Wrong includes a false grant or an incorrect identity.
type ReviewVisit struct {
	Priority float64
	Wrong    bool
}
type ReviewPoint struct {
	Threshold                                               float64
	Reviewed, WrongRemoved, CorrectWithheld, RemainingWrong int
}
type ReviewAnalysis struct {
	Curve                                                     []ReviewPoint
	BestWithinCorrectBudget, LeastForErrorTarget, MinimumCost ReviewPoint
	TargetFeasible                                            bool
	RiskCoverageArea                                          float64
}

// AnalyzeReview sweeps all inclusive priority thresholds with equal scores kept
// together. RiskCoverageArea averages the expected retained error fraction at
// every nonzero integer coverage, with random order within equal priorities.
func AnalyzeReview(visits []ReviewVisit, maxCorrectWithheld, maxResidualErrors int, reviewCost, wrongCost float64) (ReviewAnalysis, error) {
	if len(visits) == 0 || maxCorrectWithheld < 0 || maxResidualErrors < 0 || !(reviewCost >= 0 && wrongCost >= 0) || math.IsInf(reviewCost, 0) || math.IsInf(wrongCost, 0) {
		return ReviewAnalysis{}, errors.New("facex: invalid review settings")
	}
	rows := append([]ReviewVisit(nil), visits...)
	total := 0
	for _, v := range rows {
		if math.IsNaN(v.Priority) || math.IsInf(v.Priority, 0) {
			return ReviewAnalysis{}, errors.New("facex: nonfinite priority")
		}
		if v.Wrong {
			total++
		}
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Priority > rows[j].Priority })
	p := ReviewPoint{Threshold: math.Inf(1), RemainingWrong: total}
	out := ReviewAnalysis{Curve: []ReviewPoint{p}, BestWithinCorrectBudget: p, LeastForErrorTarget: p, MinimumCost: p, TargetFeasible: total <= maxResidualErrors}
	// Normalize the costs to keep finite counts from overflowing an extreme price.
	scale := max(reviewCost, wrongCost)
	if scale > 0 {
		reviewCost /= scale
		wrongCost /= scale
	}
	bestCost := float64(total) * wrongCost
	for i := 0; i < len(rows); {
		start := i
		wrong := 0
		level := rows[i].Priority
		for i < len(rows) && rows[i].Priority == level {
			if rows[i].Wrong {
				wrong++
			}
			i++
		}
		group := i - start
		p.Threshold = level
		p.Reviewed = i
		p.WrongRemoved += wrong
		p.RemainingWrong -= wrong
		p.CorrectWithheld = i - p.WrongRemoved
		out.Curve = append(out.Curve, p)
		if p.CorrectWithheld <= maxCorrectWithheld && p.WrongRemoved > out.BestWithinCorrectBudget.WrongRemoved {
			out.BestWithinCorrectBudget = p
		}
		if !out.TargetFeasible && p.RemainingWrong <= maxResidualErrors {
			out.TargetFeasible = true
			out.LeastForErrorTarget = p
		}
		cost := float64(i)*reviewCost + float64(p.RemainingWrong)*wrongCost
		if cost < bestCost {
			bestCost = cost
			out.MinimumCost = p
		}
		kept := len(rows) - i
		for j := 1; j <= group; j++ {
			out.RiskCoverageArea += (float64(p.RemainingWrong) + float64(j)*float64(wrong)/float64(group)) / float64(kept+j)
		}
	}
	out.RiskCoverageArea /= float64(len(rows))
	return out, nil
}
