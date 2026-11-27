package facex

import (
	"errors"
	"math"
	"sort"
)

// VerificationTrial is a scored pair of face embeddings with a genuine/impostor label.
type VerificationTrial struct {
	Score   float64
	Genuine bool
}
type OperatingPoint struct {
	Threshold               float64
	TrueAccept, FalseAccept int
}
type VerificationCurve struct {
	AUC, PartialAUC, EER   float64
	MinCost                float64
	CostPoint, BudgetPoint OperatingPoint
}

// EvaluateVerification integrates the tie-grouped ROC with linear interpolation.
// PartialAUC is the raw area over [0, maxFAR]. EER uses randomized interpolation;
// both returned operating points are deterministic inclusive thresholds.
// costMiss and costFalse are prior-weighted costs per class error rate.
func EvaluateVerification(trials []VerificationTrial, maxFAR, costMiss, costFalse float64, falseBudget int) (VerificationCurve, error) {
	if !(maxFAR > 0 && maxFAR <= 1 && costMiss >= 0 && costFalse >= 0) || math.IsInf(costMiss, 0) || math.IsInf(costFalse, 0) || falseBudget < 0 {
		return VerificationCurve{}, errors.New("facex: invalid verification limits")
	}
	rows := append([]VerificationTrial(nil), trials...)
	p := 0
	for _, r := range rows {
		if math.IsNaN(r.Score) || math.IsInf(r.Score, 0) {
			return VerificationCurve{}, errors.New("facex: nonfinite verification score")
		}
		if r.Genuine {
			p++
		}
	}
	n := len(rows) - p
	if p == 0 || n == 0 {
		return VerificationCurve{}, errors.New("facex: both verification classes required")
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].Score > rows[j].Score })
	point := OperatingPoint{Threshold: math.Inf(1)}
	out := VerificationCurve{CostPoint: point, BudgetPoint: point, MinCost: costMiss}
	for i := 0; i < len(rows); {
		prev := point
		point.Threshold = rows[i].Score
		for i < len(rows) && rows[i].Score == point.Threshold {
			if rows[i].Genuine {
				point.TrueAccept++
			} else {
				point.FalseAccept++
			}
			i++
		}
		x0, y0 := float64(prev.FalseAccept)/float64(n), float64(prev.TrueAccept)/float64(p)
		x1, y1 := float64(point.FalseAccept)/float64(n), float64(point.TrueAccept)/float64(p)
		out.AUC += (x1 - x0) * (y1 + y0) / 2
		if x1 > x0 && x0 < maxFAR {
			dx := min(x1, maxFAR) - x0
			out.PartialAUC += dx * (y0 + .5*(y1-y0)*dx/(x1-x0))
		}
		before, after := 1-y0-x0, 1-y1-x1
		if before > 0 && after <= 0 {
			out.EER = x0 + (x1-x0)*before/(before-after)
		}
		cost := costMiss*(1-y1) + costFalse*x1
		if cost < out.MinCost {
			out.MinCost = cost
			out.CostPoint = point
		}
		if point.FalseAccept <= falseBudget && point.TrueAccept > out.BudgetPoint.TrueAccept {
			out.BudgetPoint = point
		}
	}
	return out, nil
}
