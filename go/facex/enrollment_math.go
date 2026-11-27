package facex

import (
	"errors"
	"math"
	"slices"
	"sort"
)

func unitViews(rows [][]float32) ([][]float64, error) {
	if len(rows) == 0 {
		return nil, errors.New("facex: empty views")
	}
	u := make([][]float64, len(rows))
	for i, row := range rows {
		n, e := vectorNorm(row)
		if e != nil {
			return nil, e
		}
		if n == 0 {
			return nil, errors.New("facex: zero enrollment view")
		}
		u[i] = make([]float64, OutputSize)
		for j, x := range row {
			u[i][j] = float64(x) / n
		}
	}
	return u, nil
}

func directionDot(a, b []float64) float64 {
	var sum float64
	for j, x := range a {
		sum += x * b[j]
	}
	return sum
}

// DirectionFit summarizes a weighted chordal least-squares template and the
// enrolled view minimizing weighted summed cosine distance. A nil template
// denotes cancellation within the numerical tolerance. Angles are radians.
type DirectionFit struct {
	Template              []float64
	Medoid                int
	WorstLeaveOneOutAngle float64
	LeaveOneOutDefined    bool
}

// FitDirections uses strictly positive weights (nil means equal). The template,
// medoid and all leave-one-out comparisons take O(n*d) time.
func FitDirections(rows [][]float32, weights []float64) (DirectionFit, error) {
	u, e := unitViews(rows)
	if e != nil {
		return DirectionFit{}, e
	}
	if weights != nil && len(weights) != len(u) {
		return DirectionFit{}, errors.New("facex: expected one weight per view")
	}
	maxW := 1.
	if weights != nil {
		maxW = 0
		for _, w := range weights {
			if !(w > 0) || math.IsInf(w, 0) {
				return DirectionFit{}, errors.New("facex: invalid view weight")
			}
			maxW = max(maxW, w)
		}
	}
	w := make([]float64, len(u))
	suffix := make([][OutputSize]float64, len(u)+1)
	mass := make([]float64, len(u)+1)
	for i := len(u) - 1; i >= 0; i-- {
		w[i] = 1
		if weights != nil {
			w[i] = weights[i] / maxW
		}
		mass[i] = mass[i+1] + w[i]
		for j, x := range u[i] {
			suffix[i][j] = suffix[i+1][j] + w[i]*x
		}
	}
	sum := suffix[0][:]
	out := DirectionFit{}
	best := math.Inf(-1)
	for i, v := range u {
		s := directionDot(v, sum)
		if s > best {
			best = s
			out.Medoid = i
		}
	}
	length := math.Sqrt(directionDot(sum, sum))
	if length <= 1e-14*mass[0] {
		return out, nil
	}
	out.Template = make([]float64, OutputSize)
	for j, x := range sum {
		out.Template[j] = x / length
	}
	out.LeaveOneOutDefined = true
	var prefix, remaining [OutputSize]float64
	var prefixMass float64
	for i, v := range u {
		for j, x := range v {
			remaining[j] = prefix[j] + suffix[i+1][j]
			prefix[j] += w[i] * x
		}
		n := math.Sqrt(directionDot(remaining[:], remaining[:]))
		if n <= 1e-14*(prefixMass+mass[i+1]) {
			out.LeaveOneOutDefined = false
		} else {
			// atan2 of transverse and axial components remains useful at tiny angles.
			proj := directionDot(remaining[:], out.Template)
			var transverse float64
			for j, x := range remaining {
				delta := x - proj*out.Template[j]
				transverse += delta * delta
			}
			out.WorstLeaveOneOutAngle = max(out.WorstLeaveOneOutAngle, math.Atan2(math.Sqrt(transverse), proj))
		}
		prefixMass += w[i]
	}
	return out, nil
}

// ViewCover maximizes the worst cosine to the nearest selected enrollment view.
type ViewCover struct {
	Indices   []int
	MinCosine float64
}

// CoverViews selects one or two distinct exemplars. Ties prefer earlier
// indices. The two-center solver stores O(n*n) scores for offline enrollment.
func CoverViews(rows [][]float32, k int) (ViewCover, error) {
	if (k != 1 && k != 2) || len(rows) < k {
		return ViewCover{}, errors.New("facex: expected one or two exemplars")
	}
	u, e := unitViews(rows)
	if e != nil {
		return ViewCover{}, e
	}
	n := len(u)
	if k == 1 {
		worst := make([]float64, n)
		for i := range worst {
			worst[i] = 1
			for j := 0; j < i; j++ {
				v := max(-1., min(1., directionDot(u[i], u[j])))
				worst[i], worst[j] = min(worst[i], v), min(worst[j], v)
			}
		}
		best := slices.Max(worst)
		return ViewCover{[]int{slices.Index(worst, best)}, best}, nil
	}
	sim := make([][]float64, n)
	levels := []float64{1}
	for i := range u {
		sim[i] = make([]float64, n)
		sim[i][i] = 1
		for j := 0; j < i; j++ {
			v := max(-1., min(1., directionDot(u[i], u[j])))
			sim[i][j], sim[j][i] = v, v
			levels = append(levels, v)
		}
	}
	slices.Sort(levels)
	levels = slices.Compact(levels)
	words := (n + 63) / 64
	covers := make([]uint64, n*words)
	pair := func(level float64) (int, int) {
		clear(covers)
		for i, row := range sim {
			for j, v := range row {
				if v >= level {
					covers[i*words+j/64] |= uint64(1) << uint(j%64)
				}
			}
		}
		last := ^uint64(0)
		if n%64 != 0 {
			last = (uint64(1) << uint(n%64)) - 1
		}
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				ok := true
				for t := 0; t < words; t++ {
					mask := ^uint64(0)
					if t == words-1 {
						mask = last
					}
					if covers[i*words+t]|covers[j*words+t] != mask {
						ok = false
						break
					}
				}
				if ok {
					return i, j
				}
			}
		}
		return -1, -1
	}
	pos := sort.Search(len(levels), func(i int) bool { a, _ := pair(levels[i]); return a < 0 }) - 1
	a, b := pair(levels[pos])
	return ViewCover{[]int{a, b}, levels[pos]}, nil
}
