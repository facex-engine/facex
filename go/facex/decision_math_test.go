package facex

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func near25(t *testing.T, a, b float64) {
	t.Helper()
	if a == b {
		return
	}
	if math.IsNaN(a) || math.IsNaN(b) || math.Abs(a-b) > 2e-9 {
		t.Fatalf("got %.14g want %.14g", a, b)
	}
}
func TestDecisionPosteriorOracle(t *testing.T) {
	r := rand.New(rand.NewSource(125))
	for trial := 0; trial < 180; trial++ {
		n, m := 1+r.Intn(5), r.Intn(5)
		w := make([][]float64, n)
		for i := range w {
			w[i] = make([]float64, m)
			for j := range w[i] {
				x := r.Intn(7) - 3
				w[i][j] = float64(x)
				if x == 3 {
					w[i][j] = math.Inf(-1)
				}
			}
		}
		got, e := InferAssignments(w)
		if e != nil {
			t.Fatal(e)
		}
		var choices [][]int
		var weight []float64
		var visit func([]int, int, float64)
		visit = func(a []int, used int, l float64) {
			if len(a) == n {
				choices = append(choices, slices.Clone(a))
				weight = append(weight, math.Exp(l))
				return
			}
			i := len(a)
			visit(append(a, -1), used, l)
			for j, x := range w[i] {
				if used&(1<<j) == 0 && !math.IsInf(x, -1) {
					visit(append(a, j), used|(1<<j), l+x)
				}
			}
		}
		visit(nil, 0, 0)
		z := 0.
		for _, v := range weight {
			z += v
		}
		near25(t, got.LogNormalizer, math.Log(z))
		marg := make([][]float64, n)
		for i := range marg {
			marg[i] = make([]float64, m+1)
		}
		count := make([]float64, min(n, m)+1)
		entropy := 0.
		for l, a := range choices {
			p := weight[l] / z
			entropy -= p * math.Log(p)
			k := 0
			for i, j := range a {
				if j < 0 {
					j = m
				} else {
					k++
				}
				marg[i][j] += p
			}
			count[k] += p
		}
		near25(t, got.Entropy, entropy)
		for k, x := range count {
			near25(t, got.CountProbability[k], x)
		}
		for i, v := range marg {
			for j, p := range v {
				near25(t, got.Marginals[i][j], p)
			}
		}
		objective := func(a []int) float64 {
			s := 0.
			for i, j := range a {
				if j < 0 {
					j = m
				}
				s += marg[i][j]
			}
			return s
		}
		best := 0.
		valid := false
		for _, a := range choices {
			best = max(best, objective(a))
			valid = valid || slices.Equal(a, got.MinimumHamming)
		}
		if !valid {
			t.Fatal("infeasible Hamming decision")
		}
		near25(t, best, objective(got.MinimumHamming))
	}
}

func TestDecisionTrackOracle(t *testing.T) {
	r := rand.New(rand.NewSource(225))
	for trial := 0; trial < 220; trial++ {
		n, k := 1+r.Intn(6), 1+r.Intn(4)
		s := make([][]int, n)
		for i := range s {
			s[i] = make([]int, k)
			for j := range s[i] {
				s[i][j] = r.Intn(9) - 4
			}
		}
		lambda := r.Intn(9)
		got, e := SolveTrackPath(s, lambda)
		if e != nil {
			t.Fatal(e)
		}
		by := make([]int64, len(got.BestBySwitchCount))
		for i := range by {
			by[i] = pathAbsent
		}
		alt := make([]int64, n)
		for i := range alt {
			alt[i] = pathAbsent
		}
		best := pathAbsent
		path := make([]int, n)
		var visit func(int, int, int64)
		visit = func(i, c int, sum int64) {
			if i == n {
				by[c] = max(by[c], sum)
				u := sum - int64(c*lambda)
				best = max(best, u)
				for j, x := range path {
					if x != got.States[j] {
						alt[j] = max(alt[j], u)
					}
				}
				return
			}
			for h, x := range s[i] {
				next := c
				if i > 0 && path[i-1] != h {
					next++
				}
				path[i] = h
				visit(i+1, next, sum+int64(x))
			}
		}
		visit(0, 0, 0)
		if !slices.Equal(by, got.BestBySwitchCount) || got.Utility != best {
			t.Fatal(trial, by, got)
		}
		var score int64
		for j, h := range got.States {
			score += int64(s[j][h])
			if j > 0 && h != got.States[j-1] {
				score -= int64(lambda)
			}
			loss := int64(-1)
			if alt[j] != pathAbsent {
				loss = best - alt[j]
			}
			if got.AlternativeLoss[j] != loss {
				t.Fatal(trial, j)
			}
		}
		if score != best {
			t.Fatal("wrong path")
		}
		for steps := 0; steps < 50; steps++ {
			p := float64(steps) / 3
			opt := math.Inf(-1)
			for c, x := range by {
				opt = max(opt, float64(x)-float64(c)*p)
			}
			for c, x := range by {
				v := got.PenaltyIntervals[c]
				inside := v.Possible && p >= v.Low && p <= v.High
				tie := math.Abs(float64(x)-float64(c)*p-opt) < 1e-10
				if inside != tie {
					t.Fatal(trial, c, p, v)
				}
			}
			if (p >= got.ConstantPenalty) != (math.Abs(float64(by[0])-opt) < 1e-10) {
				t.Fatal("constant penalty")
			}
		}
	}
}

func TestDecisionSegmentsOracle(t *testing.T) {
	r := rand.New(rand.NewSource(325))
	for trial := 0; trial < 90; trial++ {
		n := 1 + r.Intn(8)
		rows := make([][]float32, n)
		for i := range rows {
			rows[i] = basis(r.Intn(4))
		}
		penalty, limit := r.Float64()*2, r.Float64()*3
		g, e := SegmentDirections(rows, n, penalty, limit)
		if e != nil {
			t.Fatal(e)
		}
		costs := make([][]float64, n)
		worst := make([]float64, n)
		for i := range worst {
			worst[i] = math.Inf(1)
		}
		loss := func(start, end int) float64 {
			var sum [4]float64
			for _, v := range rows[start:end] {
				for j := range sum {
					sum[j] += float64(v[j])
				}
			}
			l := math.Sqrt(directionDot(sum[:], sum[:]))
			s := 0.
			for _, v := range rows[start:end] {
				for j, c := range sum {
					d := float64(v[j]) - c/l
					s += d * d
				}
			}
			return s
		}
		for mask := 0; mask < 1<<(n-1); mask++ {
			k, start := 0, 0
			total, mx := 0., 0.
			for i := 1; i <= n; i++ {
				if i == n || mask&(1<<(i-1)) != 0 {
					x := loss(start, i)
					total += x
					mx = max(mx, x)
					start = i
					k++
				}
			}
			costs[k-1] = append(costs[k-1], total)
			worst[k-1] = min(worst[k-1], mx)
		}
		choice, target := 0, 0
		best := math.Inf(1)
		for j, c := range costs {
			slices.Sort(c)
			fit := g.ByCount[j]
			near25(t, fit.Loss, c[0])
			near25(t, fit.WorstLoss, worst[j])
			second := math.Inf(1)
			if len(c) > 1 {
				second = c[1]
			}
			near25(t, fit.SecondLoss, second)
			total, start := 0., 0
			for _, end := range fit.Ends {
				if end <= start {
					t.Fatal("empty segment")
				}
				total += loss(start, end)
				start = end
			}
			near25(t, total, fit.Loss)
			if start != n || len(fit.Ends) != j+1 {
				t.Fatal(fit.Ends)
			}
			if v := c[0] + float64(j)*penalty; v < best {
				choice, best = j+1, v
			}
			if target == 0 && worst[j] <= limit {
				target = j + 1
			}
		}
		if choice != g.PenalizedCount || target != g.CountForBound {
			t.Fatal(choice, target, g)
		}
	}
}

func TestDecisionReviewOracle(t *testing.T) {
	r := rand.New(rand.NewSource(425))
	for trial := 0; trial < 140; trial++ {
		n := 1 + r.Intn(7)
		v := make([]ReviewVisit, n)
		for i := range v {
			v[i] = ReviewVisit{float64(r.Intn(3)), r.Intn(2) == 0}
		}
		budget, target := r.Intn(n+1), r.Intn(n+1)
		g, e := AnalyzeReview(v, budget, target, 1, 3)
		if e != nil {
			t.Fatal(e)
		}
		cap, least, bestCost := -1, n+1, math.MaxInt
		for _, p := range g.Curve {
			withheld, removed, left := 0, 0, 0
			for _, x := range v {
				if x.Priority >= p.Threshold {
					if x.Wrong {
						removed++
					} else {
						withheld++
					}
				} else if x.Wrong {
					left++
				}
			}
			if p.Reviewed != withheld+removed || p.RemainingWrong != left || p.WrongRemoved != removed || p.CorrectWithheld != withheld {
				t.Fatal(p)
			}
			if withheld <= budget {
				cap = max(cap, removed)
			}
			if left <= target {
				least = min(least, p.Reviewed)
			}
			bestCost = min(bestCost, p.Reviewed+3*left)
		}
		if cap != g.BestWithinCorrectBudget.WrongRemoved || least != g.LeastForErrorTarget.Reviewed || bestCost != g.MinimumCost.Reviewed+3*g.MinimumCost.RemainingWrong {
			t.Fatal(g)
		}
		// Enumerate all priority-respecting retained orders, including tied visits.
		sum := 0.
		orders := 0
		var perm func(int, int, int, float64, float64)
		perm = func(depth, used, wrong int, last, area float64) {
			if depth == n {
				orders++
				sum += area / float64(n)
				return
			}
			for i, x := range v {
				if used&(1<<i) != 0 || x.Priority < last {
					continue
				}
				err := wrong
				if x.Wrong {
					err++
				}
				perm(depth+1, used|(1<<i), err, x.Priority, area+float64(err)/float64(depth+1))
			}
		}
		perm(0, 0, 0, math.Inf(-1), 0)
		near25(t, g.RiskCoverageArea, sum/float64(orders))
	}
}

func TestDecisionVisitStatistics(t *testing.T) {
	for _, v := range []struct {
		n, k   int
		lo, hi float64
	}{{10, 3, .06673951117773447, .652452850060008}, {50, 0, 0, .07112173646420458}, {1000000, 12, .000006200589987895605, .000020961491123670386}, {1000000, 500000, .4990195191953118, .5009804808046883}} {
		got, e := VisitErrorInterval(v.n, v.k, .05)
		if e != nil {
			t.Fatal(e)
		}
		near25(t, got.Low, v.lo)
		near25(t, got.High, v.hi)
	}
	g, e := AuditPairedVisits(100, 8, 2, .05)
	if e != nil {
		t.Fatal(e)
	}
	near25(t, g.McNemarP, .109375)
	near25(t, g.DiscordantOdds.Low, .7982524004966308)
	near25(t, g.DiscordantOdds.High, 38.6656560797152)
	near25(t, g.ErrorReduction.Low, -.047882280472540936)
	near25(t, g.ErrorReduction.High, .1608739820078152)
	n, e := ZeroErrorVisits(.001, .05)
	if e != nil || n != 2995 {
		t.Fatal(n, e)
	}
	g, e = AuditPairedVisits(100, 0, 0, .05)
	if e != nil || g.McNemarP != 1 || !math.IsInf(g.DiscordantOdds.High, 1) {
		t.Fatal(g, e)
	}
}

func TestDecisionWorkloads(t *testing.T) {
	if os.Getenv("FACEX_GALLERY_SCALE") != "1" {
		t.Skip("scale workload")
	}
	r := rand.New(rand.NewSource(25))
	w := make([][]float64, 12)
	for i := range w {
		w[i] = make([]float64, 12)
		for j := range w[i] {
			w[i][j] = r.NormFloat64()
		}
	}
	now := time.Now()
	if _, e := InferAssignments(w); e != nil {
		t.Fatal(e)
	}
	t.Log("exact posterior 12 by 12", time.Since(now))
	s := make([][]int, 180)
	for i := range s {
		s[i] = make([]int, 20)
		for j := range s[i] {
			s[i][j] = r.Intn(101)
		}
	}
	now = time.Now()
	if _, e := SolveTrackPath(s, 25); e != nil {
		t.Fatal(e)
	}
	t.Log("track 180 frames 20 states", time.Since(now))
	rows := make([][]float32, 128)
	for i := range rows {
		rows[i] = basis(r.Intn(30))
	}
	now = time.Now()
	if _, e := SegmentDirections(rows, 12, 1, 3); e != nil {
		t.Fatal(e)
	}
	t.Log("segmentation 128 frames", time.Since(now))
	visits := make([]ReviewVisit, 100000)
	for i := range visits {
		visits[i] = ReviewVisit{float64(r.Intn(1000)), r.Intn(10) == 0}
	}
	now = time.Now()
	if _, e := AnalyzeReview(visits, 1000, 500, 1, 10); e != nil {
		t.Fatal(e)
	}
	t.Log("review 100000 visits", time.Since(now))
}

// Each fixture photo contributes one query. Ten people are enrolled, five
// are held out. This is a replay of still images, not independent visits.
func TestDecisionModelReplay(t *testing.T) {
	dir := os.Getenv("FACEX_GALLERY_FIXTURE")
	if dir == "" {
		t.Skip("requires model fixture")
	}
	rows, e := os.Open(filepath.Join(dir, "rows"))
	if e != nil {
		t.Fatal(e)
	}
	defer rows.Close()
	qf, e := os.Open(filepath.Join(dir, "queries"))
	if e != nil {
		t.Fatal(e)
	}
	defer qf.Close()
	axes := make([][]float64, 10)
	for i := range axes {
		if _, e = rows.Seek(int64(i)*4369*OutputSize*4, 0); e != nil {
			t.Fatal(e)
		}
		views := make([][]float32, 8)
		for j := range views {
			views[j] = make([]float32, OutputSize)
			if e = binary.Read(rows, binary.LittleEndian, views[j]); e != nil {
				t.Fatal(e)
			}
		}
		f, e := FitDirections(views, nil)
		if e != nil || f.Template == nil {
			t.Fatal(e)
		}
		axes[i] = f.Template
	}
	var visits []ReviewVisit
	var roster, oldPolicy, newPolicy strings.Builder
	roster.WriteString("visit_id,expected_id\n")
	oldPolicy.WriteString("visit_id,decision\n")
	newPolicy.WriteString("visit_id,decision\n")
	fixed, withheld, remaining := 0, 0, 0
	// Expected gaps for the five held out FP32 probes in fixture order.
	fp32Gaps := [...]float64{0.017933, 0.036219, 0.170846, 0.082843, 0.084267}
	fp32 := os.Getenv("FACEX_GALLERY_PROFILE") == "fp32"
	for truth := 0; truth < 15; truth++ {
		q := make([]float32, OutputSize)
		if e = binary.Read(qf, binary.LittleEndian, q); e != nil {
			t.Fatal(e)
		}
		u, e := unitViews([][]float32{q})
		if e != nil {
			t.Fatal(e)
		}
		best, next, id := -2., -2., -1
		for j, a := range axes {
			x := directionDot(a, u[0])
			if x > best {
				next, best, id = best, x, j
			} else if x > next {
				next = x
			}
		}
		gap := best - next
		expected := fmt.Sprint(truth)
		if truth >= len(axes) {
			expected = "unknown"
		}
		fmt.Fprintf(&roster, "%d,%s\n", truth, expected)
		fmt.Fprintf(&oldPolicy, "%d,%d\n", truth, id)
		candidate := fmt.Sprint(id)
		if gap <= .1 {
			candidate = "review"
		}
		fmt.Fprintf(&newPolicy, "%d,%s\n", truth, candidate)
		if truth >= len(axes) {
			t.Logf("probe=%d top_id=%d best=%.8f next=%.8f gap=%.8f decision=%s", truth, id, best, next, gap, candidate)
			if fp32 && math.Abs(gap-fp32Gaps[truth-len(axes)]) > 2e-5 {
				t.Errorf("FP32 gap changed for probe %d: %.8f", truth, gap)
			}
		}
		wrong := id != truth
		visits = append(visits, ReviewVisit{-gap, wrong})
		if gap <= .1 {
			if wrong {
				fixed++
			} else {
				withheld++
			}
		} else if wrong {
			remaining++
		}
	}
	a, e := AnalyzeReview(visits, 0, 0, 1, 3)
	if e != nil {
		t.Fatal(e)
	}
	comparison, err := CompareVisitLogs(strings.NewReader(roster.String()), strings.NewReader(oldPolicy.String()), strings.NewReader(newPolicy.String()))
	if err != nil || comparison.WrongRemoved != fixed || comparison.CorrectIDToReview != withheld || comparison.After.WrongKnownID+comparison.After.UnknownGranted != remaining {
		t.Fatal(comparison, err)
	}
	t.Logf("paired report visits=%d known=%d unknown=%d wrong_removed=%d introduced=%d correct_to_review=%d remaining=%d", comparison.Visits, comparison.Known, comparison.Unknown, comparison.WrongRemoved, comparison.WrongIntroduced, comparison.CorrectIDToReview, remaining)
	if fp32 && (fixed != 4 || withheld != 0 || remaining != 1) {
		t.Error("FP32 review decisions changed on the model fixture")
	}
	before := a.Curve[0].RemainingWrong
	if before != fixed+remaining {
		t.Fatal("replay mismatch")
	}
	t.Logf("fixture top1 replay queries=15 enrolled=10 held_out=5 wrong_before=%d wrong_removed=%d correct_withheld=%d wrong_remaining=%d gap_cutoff=0.1", before, fixed, withheld, remaining)
}

func TestDecisionInvalidAndExtreme(t *testing.T) {
	if _, e := InferAssignments([][]float64{{math.NaN()}}); e == nil {
		t.Fatal("invalid weight")
	}
	x, e := InferAssignments([][]float64{{math.Inf(-1), 100}, {math.Inf(-1), -100}})
	if e != nil {
		t.Fatal(e)
	}
	for _, v := range x.Marginals {
		near25(t, v[0], 0)
		near25(t, v[0]+v[1]+v[2], 1)
	}
	if _, e = SolveTrackPath([][]int{{0}, {}}, 1); e == nil {
		t.Fatal("jagged track")
	}
	if _, e = SegmentDirections([][]float32{make([]float32, OutputSize)}, 1, 0, 0); e == nil {
		t.Fatal("zero vector")
	}
	if _, e = AnalyzeReview([]ReviewVisit{{math.NaN(), false}}, 0, 0, 1, 1); e == nil {
		t.Fatal("invalid priority")
	}
	if _, e = AuditPairedVisits(4, 3, 2, .05); e == nil {
		t.Fatal("invalid counts")
	}
}
