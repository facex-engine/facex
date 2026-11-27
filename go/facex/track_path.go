package facex

import (
	"errors"
	"math"
)

// TrackPath treats each column as a state (which may include unknown).
// Scores and switchPenalty are integer utilities in the same units.
type TrackPath struct {
	States                             []int
	Utility                            int64
	AlternativeLoss, BestBySwitchCount []int64
	ConstantPenalty                    float64
	PenaltyIntervals                   []PenaltyInterval
}

// PenaltyInterval is the closed nonnegative penalty interval on which a switch count is optimal.
type PenaltyInterval struct {
	Low, High float64
	Possible  bool
}

const pathAbsent int64 = math.MinInt64 / 4

// SolveTrackPath solves the offline sequence for up to 256 frames and 64 states.
// Every state at every frame has an explicit finite utility in [-1e6,1e6].
func SolveTrackPath(scores [][]int, switchPenalty int) (TrackPath, error) {
	t := len(scores)
	if t == 0 || t > 256 || len(scores[0]) == 0 || len(scores[0]) > 64 || switchPenalty < 0 || switchPenalty > 1000000000 {
		return TrackPath{}, errors.New("facex: invalid track shape or penalty")
	}
	k := len(scores[0])
	for _, row := range scores {
		if len(row) != k {
			return TrackPath{}, errors.New("facex: jagged track")
		}
		for _, x := range row {
			if x < -1000000 || x > 1000000 {
				return TrackPath{}, errors.New("facex: invalid track utility")
			}
		}
	}
	penalty := int64(switchPenalty)
	forward := make([]int64, t*k)
	pred := make([]int, t*k)
	for j, v := range scores[0] {
		forward[j] = int64(v)
	}
	for i := 1; i < t; i++ {
		for j, v := range scores[i] {
			best := forward[(i-1)*k+j]
			origin := j
			for prev := 0; prev < k; prev++ {
				x := forward[(i-1)*k+prev] - penalty
				if x > best {
					best, origin = x, prev
				}
			}
			forward[i*k+j] = best + int64(v)
			pred[i*k+j] = origin
		}
	}
	out := TrackPath{States: make([]int, t), AlternativeLoss: make([]int64, t)}
	end := 0
	for j := 1; j < k; j++ {
		if forward[(t-1)*k+j] > forward[(t-1)*k+end] {
			end = j
		}
	}
	out.Utility = forward[(t-1)*k+end]
	for i := t - 1; i >= 0; i-- {
		out.States[i] = end
		end = pred[i*k+end]
	}
	tail, next := make([]int64, k), make([]int64, k)
	for i := t - 1; i >= 0; i-- {
		alt := pathAbsent
		for j := 0; j < k; j++ {
			if j != out.States[i] {
				alt = max(alt, forward[i*k+j]+tail[j])
			}
		}
		out.AlternativeLoss[i] = -1
		if alt != pathAbsent {
			out.AlternativeLoss[i] = out.Utility - alt
		}
		for j := 0; j < k; j++ {
			best := int64(scores[i][j]) + tail[j]
			for h := 0; h < k; h++ {
				best = max(best, int64(scores[i][h])+tail[h]-penalty)
			}
			next[j] = best
		}
		tail, next = next, tail
	}
	// For exactly c changes, keep the two best predecessor states so the
	// best one different from j is available in O(1).
	prev, curr := make([]int64, t*k), make([]int64, t*k)
	for a := range prev {
		prev[a] = pathAbsent
	}
	copy(prev, forward[:k])
	for i := 1; i < t; i++ {
		for c := 0; c <= i; c++ {
			first, second, idx := pathAbsent, pathAbsent, -1
			if c > 0 {
				for h, x := range prev[(c-1)*k : c*k] {
					if x > first {
						first, second, idx = x, first, h
					} else {
						second = max(second, x)
					}
				}
			}
			for j, v := range scores[i] {
				change := first
				if idx == j {
					change = second
				}
				stay := pathAbsent
				if c < i {
					stay = prev[c*k+j]
				}
				best := max(change, stay)
				curr[c*k+j] = best
				if best != pathAbsent {
					curr[c*k+j] += int64(v)
				}
			}
		}
		prev, curr = curr, prev
	}
	counts := t
	if k == 1 {
		counts = 1
	}
	out.BestBySwitchCount = make([]int64, counts)
	for c := 0; c < counts; c++ {
		best := pathAbsent
		for _, v := range prev[c*k : (c+1)*k] {
			best = max(best, v)
		}
		out.BestBySwitchCount[c] = best
	}
	for c, v := range out.BestBySwitchCount {
		lo, hi := 0., math.Inf(1)
		for h, w := range out.BestBySwitchCount {
			if h > c {
				lo = max(lo, float64(w-v)/float64(h-c))
			} else if h < c {
				hi = min(hi, float64(v-w)/float64(c-h))
			}
		}
		out.PenaltyIntervals = append(out.PenaltyIntervals, PenaltyInterval{lo, hi, lo <= hi})
	}
	out.ConstantPenalty = out.PenaltyIntervals[0].Low
	return out, nil
}
