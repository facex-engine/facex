package facex

import (
	"errors"
	"math"
)

type ProbabilityInterval struct{ Low, High float64 }

// Binomial CDF via the smaller-side recurrence around the binomial mode.
func binomialBelow(n, k int, p float64) float64 {
	if k < 0 {
		return 0
	}
	if k >= n || p <= 0 {
		return 1
	}
	if p >= 1 {
		return 0
	}
	if float64(k) >= float64(n)*p {
		return 1 - binomialLeft(n, n-k-1, 1-p)
	}
	return binomialLeft(n, k, p)
}
func binomialLeft(n, k int, p float64) float64 {
	if p == 0 {
		return 1
	}
	if p == 1 {
		return 0
	}
	a, _ := math.Lgamma(float64(n + 1))
	b, _ := math.Lgamma(float64(k + 1))
	c, _ := math.Lgamma(float64(n - k + 1))
	term := math.Exp(a - b - c + float64(k)*math.Log(p) + float64(n-k)*math.Log1p(-p))
	sum := term
	for j := k; j > 0; j-- {
		term *= float64(j) / float64(n-j+1) * ((1 - p) / p)
		old := sum
		sum += term
		if sum == old {
			break
		}
	}
	return min(1., sum)
}
func upperBinomial(n, k int, tail float64) float64 {
	if k == n {
		return 1
	}
	if k == 0 {
		return -math.Expm1(math.Log(tail) / float64(n))
	}
	lo, hi := 0., 1.
	for i := 0; i < 64; i++ {
		mid := (lo + hi) / 2
		if binomialBelow(n, k, mid) > tail {
			lo = mid
		} else {
			hi = mid
		}
	}
	return hi
}
func centralInterval(n, k int, alpha float64) ProbabilityInterval {
	return ProbabilityInterval{1 - upperBinomial(n, n-k, alpha/2), upperBinomial(n, k, alpha/2)}
}

// VisitErrorInterval gives the central Clopper-Pearson interval for a fixed
// binary error definition and independent identically distributed visits.
func VisitErrorInterval(n, k int, alpha float64) (ProbabilityInterval, error) {
	if n < 1 || n > 1000000 || k < 0 || k > n || !(alpha >= 1e-12 && alpha < 1) {
		return ProbabilityInterval{}, errors.New("facex: invalid binomial audit")
	}
	return centralInterval(n, k, alpha), nil
}

// ZeroErrorVisits gives the least number of independent error-free visits for
// a one-sided upper bound at most target, at confidence 1-alpha.
func ZeroErrorVisits(target, alpha float64) (int, error) {
	if !(target > 0 && target < 1 && alpha >= 1e-12 && alpha < 1) {
		return 0, errors.New("facex: invalid zero-error target")
	}
	v := math.Ceil(math.Log(alpha) / math.Log1p(-target))
	if v > 1e9 {
		return 0, errors.New("facex: required sample exceeds one billion")
	}
	return int(v), nil
}

// Fixed counts visits where only the old policy erred; introduced counts
// visits where only the new policy erred. Both policies see the same visits.
type PairedAudit struct {
	McNemarP                       float64
	ErrorReduction, DiscordantOdds ProbabilityInterval
}

func AuditPairedVisits(n, fixed, introduced int, alpha float64) (PairedAudit, error) {
	if n < 1 || n > 1000000 || fixed < 0 || introduced < 0 || fixed > n || introduced > n-fixed || !(alpha >= 1e-12 && alpha < 1) {
		return PairedAudit{}, errors.New("facex: invalid paired audit")
	}
	d := fixed + introduced
	out := PairedAudit{McNemarP: 1, DiscordantOdds: ProbabilityInterval{0, math.Inf(1)}}
	if d > 0 {
		out.McNemarP = min(1., 2*binomialBelow(d, min(fixed, introduced), .5))
		x := centralInterval(d, fixed, alpha)
		out.DiscordantOdds = ProbabilityInterval{x.Low / (1 - x.Low), x.High / (1 - x.High)}
	}
	// Simultaneous marginal intervals via a union bound, then subtract.
	a, b := centralInterval(n, fixed, alpha/2), centralInterval(n, introduced, alpha/2)
	out.ErrorReduction = ProbabilityInterval{a.Low - b.High, a.High - b.Low}
	return out, nil
}
