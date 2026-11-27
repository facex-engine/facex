package facex

import (
	"errors"
	"math"
	"sort"
)

// Association utilities use integer millionths in [-1000000,1000000].
// ForbiddenAssociation removes an edge. An unmatched face earns zero.
const ForbiddenAssociation = -1000001
const associationInfinity int64 = math.MaxInt64 / 4

// Association maps each input face to a different column or -1.
type Association struct {
	Columns []int
	Utility int64
}

// AssociationAnalysis reports the best assignment for every attainable match
// count, the best optional assignment, and the maximum-cardinality optimum.
type AssociationAnalysis struct {
	ByCount           []Association
	Best, MostMatches Association
}

func associationShape(scores [][]int) (int, error) {
	m := 0
	if len(scores) > 0 {
		m = len(scores[0])
	}
	for _, row := range scores {
		if len(row) != m {
			return 0, errors.New("facex: jagged association matrix")
		}
		for _, v := range row {
			if v < ForbiddenAssociation || v > 1000000 {
				return 0, errors.New("facex: invalid association score")
			}
		}
	}
	return m, nil
}

func unmatched(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = -1
	}
	return out
}

// AnalyzeAssociations augments a minimum-cost flow by one unit at a time.
// The residual graph can reassign earlier matches. Each augmentation gives
// the optimum at its new cardinality, including when the next gain is negative.
func AnalyzeAssociations(scores [][]int) (AssociationAnalysis, error) {
	m, e := associationShape(scores)
	if e != nil {
		return AssociationAnalysis{}, e
	}
	n := len(scores)
	cols, owners := unmatched(n), unmatched(m)
	zero := Association{append([]int(nil), cols...), 0}
	out := AssociationAnalysis{ByCount: []Association{zero}, Best: zero, MostMatches: zero}
	for k := 0; k < min(n, m); k++ {
		dRow, dCol := make([]int64, n), make([]int64, m)
		pred := unmatched(m)
		for i, c := range cols {
			if c >= 0 {
				dRow[i] = associationInfinity
			}
		}
		for j := range dCol {
			dCol[j] = associationInfinity
		}
		for pass := 0; pass <= k; pass++ {
			changed := false
			for i, row := range scores {
				if dRow[i] == associationInfinity {
					continue
				}
				for j, v := range row {
					if v == ForbiddenAssociation || cols[i] == j {
						continue
					}
					next := dRow[i] - int64(v)
					if next < dCol[j] {
						dCol[j], pred[j] = next, i
						changed = true
					}
				}
			}
			for j, i := range owners {
				if i >= 0 && dCol[j] != associationInfinity {
					dRow[i] = min(dRow[i], dCol[j]+int64(scores[i][j]))
				}
			}
			if !changed {
				break
			}
		}
		end := -1
		for j, i := range owners {
			if i < 0 && pred[j] >= 0 && (end < 0 || dCol[j] < dCol[end]) {
				end = j
			}
		}
		if end < 0 {
			break
		}
		total := out.MostMatches.Utility - dCol[end]
		for end >= 0 {
			i := pred[end]
			prev := cols[i]
			cols[i], owners[end] = end, i
			end = prev
		}
		match := Association{append([]int(nil), cols...), total}
		out.ByCount = append(out.ByCount, match)
		out.MostMatches = match
		if total > out.Best.Utility {
			out.Best = match
		}
	}
	return out, nil
}

// AssociationRegrets returns the minimum total utility lost when each face is
// forced to receive a different decision from the optional global optimum.
// A zero loss means there is an equally good alternative. A -1 result means
// no alternative exists. The returned optimum fixes the reference tie choice.
func AssociationRegrets(scores [][]int) (Association, []int64, error) {
	a, e := AnalyzeAssociations(scores)
	if e != nil {
		return Association{}, nil, e
	}
	best := a.Best
	n := len(scores)
	if n == 0 {
		return best, nil, nil
	}
	m := len(scores[0])
	hub := m + n
	d := make([][]int64, hub+1)
	used := make([]bool, hub)
	for i := range d {
		d[i] = make([]int64, hub+1)
		for j := range d[i] {
			if i != j {
				d[i][j] = associationInfinity
			}
		}
	}
	old := make([]int, n)
	value := make([]int64, n)
	for i, j := range best.Columns {
		if j < 0 {
			j = m + i
		} else {
			value[i] = int64(scores[i][j])
		}
		old[i] = j
		used[j] = true
		for c, v := range scores[i] {
			if v != ForbiddenAssociation && c != j {
				d[j][c] = value[i] - int64(v)
			}
		}
		if j != m+i {
			d[j][m+i] = value[i]
		}
	}
	// Closing an alternating path through the hub accounts for swaps that
	// both release an occupied column and consume a previously unused one.
	for j, occupied := range used {
		if occupied {
			d[hub][j] = 0
		} else {
			d[j][hub] = 0
		}
	}
	for k := range d {
		for i := range d {
			if d[i][k] == associationInfinity {
				continue
			}
			for j, right := range d[k] {
				if right != associationInfinity {
					d[i][j] = min(d[i][j], d[i][k]+right)
				}
			}
		}
	}
	loss := make([]int64, n)
	for i, j := range old {
		v := associationInfinity
		for c, s := range scores[i] {
			if s != ForbiddenAssociation && c != j && d[c][j] != associationInfinity {
				v = min(v, value[i]-int64(s)+d[c][j])
			}
		}
		if c := m + i; c != j && d[c][j] != associationInfinity {
			v = min(v, value[i]+d[c][j])
		}
		if v == associationInfinity {
			v = -1
		}
		loss[i] = v
	}
	return best, loss, nil
}

// AssociationBottleneck finds the highest weakest edge among all matchings of
// the requested positive cardinality. ok=false means the count is infeasible.
func AssociationBottleneck(scores [][]int, count int) (level int, ok bool, err error) {
	m, e := associationShape(scores)
	if e != nil {
		return 0, false, e
	}
	if count <= 0 {
		return 0, false, errors.New("facex: expected positive match count")
	}
	if count > min(len(scores), m) {
		return 0, false, nil
	}
	feasible := func(level int) bool {
		owner := unmatched(m)
		seen := make([]int, m)
		var augment func(int, int) bool
		augment = func(i, stamp int) bool {
			for j, s := range scores[i] {
				if s < level || seen[j] == stamp {
					continue
				}
				seen[j] = stamp
				if owner[j] < 0 || augment(owner[j], stamp) {
					owner[j] = i
					return true
				}
			}
			return false
		}
		hits := 0
		for i := range scores {
			if augment(i, i+1) {
				hits++
				if hits == count {
					return true
				}
			}
		}
		return false
	}
	// Search the fixed-point utility range. Index zero is the lowest legal score.
	pos := sort.Search(2000001, func(i int) bool { return !feasible(i - 1000000) })
	if pos == 0 {
		return 0, false, nil
	}
	return pos - 1 - 1000000, true, nil
}
