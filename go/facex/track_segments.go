package facex

import (
	"errors"
	"math"
)

// SegmentFit summarizes a partition of a track into K contiguous nonempty segments.
// Ends are exclusive. SecondLoss is the cost of the best distinct partition
// with this K, including an equal-cost alternative. Losses sum squared chords
// to unit representatives. WorstLoss is minimized separately over partitions.
type SegmentFit struct {
	Ends                        []int
	Loss, SecondLoss, WorstLoss float64
}
type SegmentFrontier struct {
	ByCount                       []SegmentFit
	PenalizedCount, CountForBound int
}

// SegmentDirections fits from one through maxSegments segments (up to 256 frames).
// CountForBound is zero when no partition respects maxSegmentLoss for every segment.
func SegmentDirections(rows [][]float32, maxSegments int, penalty, maxSegmentLoss float64) (SegmentFrontier, error) {
	n := len(rows)
	if n > 256 || maxSegments < 1 || maxSegments > n || !(penalty >= 0 && maxSegmentLoss >= 0) || math.IsInf(penalty, 0) {
		return SegmentFrontier{}, errors.New("facex: invalid segmentation settings")
	}
	u, e := unitViews(rows)
	if e != nil {
		return SegmentFrontier{}, e
	}
	width := n + 1
	cost := make([]float64, width*width)
	for start := 0; start < n; start++ {
		var sum [OutputSize]float64
		for end := start + 1; end <= n; end++ {
			for j, v := range u[end-1] {
				sum[j] += v
			}
			if end-start > 1 {
				cost[start*width+end] = 2 * max(0., float64(end-start)-math.Sqrt(directionDot(sum[:], sum[:])))
			}
		}
	}
	type cell struct {
		best, second, worst float64
		start               int
	}
	table := make([]cell, (maxSegments+1)*width)
	for i := range table {
		table[i] = cell{math.Inf(1), math.Inf(1), math.Inf(1), -1}
	}
	table[0] = cell{0, math.Inf(1), 0, -1}
	out := SegmentFrontier{}
	opt := math.Inf(1)
	for k := 1; k <= maxSegments; k++ {
		for end := k; end <= n; end++ {
			c := &table[k*width+end]
			for start := k - 1; start < end; start++ {
				p := table[(k-1)*width+start]
				x := cost[start*width+end]
				cand := p.best + x
				if cand < c.best {
					c.second, c.best, c.start = c.best, cand, start
				} else {
					c.second = min(c.second, cand)
				}
				c.second = min(c.second, p.second+x)
				c.worst = min(c.worst, max(p.worst, x))
			}
		}
		c := table[k*width+n]
		ends := make([]int, k)
		end := n
		for j := k; j > 0; j-- {
			ends[j-1] = end
			end = table[j*width+end].start
		}
		out.ByCount = append(out.ByCount, SegmentFit{ends, c.best, c.second, c.worst})
		if val := c.best + float64(k-1)*penalty; val < opt {
			opt = val
			out.PenalizedCount = k
		}
		if out.CountForBound == 0 && c.worst <= maxSegmentLoss {
			out.CountForBound = k
		}
	}
	return out, nil
}
