package facex

import (
	"errors"
	"math"
	"math/bits"
)

// AssignmentPosterior describes the Gibbs distribution on partial matchings.
// Each row may abstain with log weight zero. The last marginal column is abstention.
type AssignmentPosterior struct {
	LogNormalizer, Entropy float64
	Marginals              [][]float64
	CountProbability       []float64
	MinimumHamming         []int
}

func logPlus(a, b float64) float64 {
	if a < b {
		a, b = b, a
	}
	if math.IsInf(b, -1) {
		return a
	}
	return a + math.Log1p(math.Exp(b-a))
}

// InferAssignments accepts log weights relative to abstention, in [-100,100],
// or negative infinity for forbidden edges. Scores need external calibration
// before the model probabilities can be interpreted as observed frequencies.
func InferAssignments(w [][]float64) (AssignmentPosterior, error) {
	n := len(w)
	m := 0
	if n > 0 {
		m = len(w[0])
	}
	if m > 16 || n+1 > (1<<20)/(1<<m) {
		return AssignmentPosterior{}, errors.New("facex: posterior state limit exceeded")
	}
	size := 1 << m
	for _, r := range w {
		if len(r) != m {
			return AssignmentPosterior{}, errors.New("facex: jagged weights")
		}
		for _, v := range r {
			if !math.IsInf(v, -1) && !(v >= -100 && v <= 100) {
				return AssignmentPosterior{}, errors.New("facex: invalid log weight")
			}
		}
	}
	f := make([]float64, (n+1)*size)
	for i := 1; i < len(f); i++ {
		f[i] = math.Inf(-1)
	}
	for i, row := range w {
		for mask, v := range f[i*size : (i+1)*size] {
			if math.IsInf(v, -1) {
				continue
			}
			next := f[(i+1)*size : (i+2)*size]
			next[mask] = logPlus(next[mask], v)
			for j, edge := range row {
				bit := 1 << j
				if mask&bit == 0 {
					next[mask|bit] = logPlus(next[mask|bit], v+edge)
				}
			}
		}
	}
	last := f[n*size:]
	z := math.Inf(-1)
	for _, v := range last {
		z = logPlus(z, v)
	}
	out := AssignmentPosterior{LogNormalizer: z, Entropy: z, Marginals: make([][]float64, n), CountProbability: make([]float64, min(n, m)+1), MinimumHamming: make([]int, n)}
	for mask, v := range last {
		if !math.IsInf(v, -1) {
			out.CountProbability[bits.OnesCount(uint(mask))] += math.Exp(v - z)
		}
	}
	back, temp := make([]float64, size), make([]float64, size)
	for i := n - 1; i >= 0; i-- {
		p := make([]float64, m+1)
		out.Marginals[i] = p
		for mask, b := range back {
			prefix := f[i*size+mask] - z
			p[m] += math.Exp(prefix + b)
			s := b
			for j, edge := range w[i] {
				bit := 1 << j
				if mask&bit != 0 || math.IsInf(edge, -1) {
					continue
				}
				v := edge + back[mask|bit]
				p[j] += math.Exp(prefix + v)
				s = logPlus(s, v)
			}
			temp[mask] = s
		}
		for j, edge := range w[i] {
			if !math.IsInf(edge, -1) {
				out.Entropy -= p[j] * edge
			}
		}
		back, temp = temp, back
	}
	out.Entropy = max(0., out.Entropy)
	// Maximize expected correct row decisions under the same exclusion rules.
	clear(back)
	policy := make([]int8, n*size)
	for i := n - 1; i >= 0; i-- {
		for mask := range back {
			best := out.Marginals[i][m] + back[mask]
			choice := int8(-1)
			for j, p := range out.Marginals[i][:m] {
				bit := 1 << j
				if mask&bit == 0 && !math.IsInf(w[i][j], -1) && p+back[mask|bit] > best {
					best, choice = p+back[mask|bit], int8(j)
				}
			}
			temp[mask] = best
			policy[i*size+mask] = choice
		}
		back, temp = temp, back
	}
	mask := 0
	for i := range out.MinimumHamming {
		j := int(policy[i*size+mask])
		out.MinimumHamming[i] = j
		if j >= 0 {
			mask |= 1 << j
		}
	}
	return out, nil
}
