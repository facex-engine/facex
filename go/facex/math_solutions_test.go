package facex

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func closeMath(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > 1e-10 {
		t.Fatalf("got %.16g want %.16g", got, want)
	}
}

func TestMathEnrollment(t *testing.T) {
	r := rand.New(rand.NewSource(2069))
	for trial := 0; trial < 50; trial++ {
		n := 2 + r.Intn(11)
		if trial == 0 {
			n = 65
		}
		rows := make([][]float32, n)
		weights := make([]float64, n)
		for i := range rows {
			rows[i] = make([]float32, OutputSize)
			for j := 0; j < 9; j++ {
				rows[i][j] = float32(r.NormFloat64())
			}
			weights[i] = .1 + r.Float64()
		}
		u, _ := unitViews(rows)
		fit, e := FitDirections(rows, weights)
		if e != nil || !fit.LeaveOneOutDefined {
			t.Fatal(fit, e)
		}
		sum := make([]float64, OutputSize)
		for i, v := range u {
			for j, x := range v {
				sum[j] += weights[i] * x
			}
		}
		length := math.Sqrt(directionDot(sum, sum))
		for j, x := range sum {
			closeMath(t, fit.Template[j], x/length)
		}
		medoid := 0
		medCost := math.Inf(1)
		angle := 0.
		for i, v := range u {
			cost := 0.
			without := make([]float64, OutputSize)
			for k, other := range u {
				cost += weights[k] * (1 - directionDot(v, other))
				if k != i {
					for j, x := range other {
						without[j] += weights[k] * x
					}
				}
			}
			if cost < medCost {
				medCost, medoid = cost, i
			}
			c := directionDot(fit.Template, without) / math.Sqrt(directionDot(without, without))
			angle = max(angle, math.Acos(max(-1., min(1., c))))
		}
		if fit.Medoid != medoid {
			t.Fatal(fit.Medoid, medoid)
		}
		closeMath(t, fit.WorstLeaveOneOutAngle, angle)
		for k := 1; k <= 2; k++ {
			got, e := CoverViews(rows, k)
			if e != nil {
				t.Fatal(e)
			}
			best := math.Inf(-1)
			var want []int
			dist := make([][]float64, n)
			for i := range u {
				dist[i] = make([]float64, n)
				for j := range u {
					dist[i][j] = directionDot(u[i], u[j])
					if i == j {
						dist[i][j] = 1
					}
				}
			}
			for i := range u {
				for j := i; j < n; j++ {
					if (k == 1) != (i == j) {
						continue
					}
					worst := 1.
					for a := range u {
						worst = min(worst, max(dist[i][a], dist[j][a]))
					}
					if worst > best {
						best = worst
						want = []int{i}
						if k == 2 {
							want = append(want, j)
						}
					}
				}
			}
			closeMath(t, got.MinCosine, best)
			if !slices.Equal(got.Indices, want) {
				t.Fatalf("cover got %v want %v", got, want)
			}
		}
	}
	a := basis(0)
	b := basis(0)
	b[0] = -1
	fit, e := FitDirections([][]float32{a, b}, nil)
	if e != nil || fit.Template != nil || fit.LeaveOneOutDefined {
		t.Fatal(fit, e)
	}
	fit, e = FitDirections([][]float32{a, b, b}, nil)
	if e != nil || fit.LeaveOneOutDefined {
		t.Fatal(fit, e)
	}
	if _, e = FitDirections([][]float32{a}, []float64{math.NaN()}); e == nil {
		t.Fatal("bad weight")
	}
}

func TestMathVerification(t *testing.T) {
	tied := []VerificationTrial{{1, true}, {1, false}}
	v, e := EvaluateVerification(tied, .2, 1, 1, 0)
	if e != nil {
		t.Fatal(e)
	}
	closeMath(t, v.AUC, .5)
	closeMath(t, v.EER, .5)
	closeMath(t, v.PartialAUC, .02)
	for i, high := range []bool{true, false} {
		v, e := EvaluateVerification([]VerificationTrial{{1, high}, {0, !high}}, .3, 1, 1, 0)
		if e != nil {
			t.Fatal(e)
		}
		closeMath(t, v.AUC, float64(1-i))
		closeMath(t, v.EER, float64(i))
		closeMath(t, v.PartialAUC, .3*float64(1-i))
	}
	r := rand.New(rand.NewSource(271))
	for trial := 0; trial < 300; trial++ {
		rows := append([]VerificationTrial(nil), tied...)
		for i := 0; i < 23; i++ {
			rows = append(rows, VerificationTrial{float64(r.Intn(7)), r.Intn(2) == 0})
		}
		limit := r.Intn(5)
		cost := .3 + r.Float64()
		far := .03 + .95*r.Float64()
		got, e := EvaluateVerification(rows, far, 1, cost, limit)
		if e != nil {
			t.Fatal(e)
		}
		var pos, neg []float64
		for _, x := range rows {
			if x.Genuine {
				pos = append(pos, x.Score)
			} else {
				neg = append(neg, x.Score)
			}
		}
		area, partial := 0., 0.
		for _, s := range neg {
			above, ties, ng, eq := 0, 0, 0, 0
			for _, q := range pos {
				if q > s {
					above++
				}
				if q == s {
					ties++
				}
			}
			for _, q := range neg {
				if q > s {
					ng++
				}
				if q == s {
					eq++
				}
			}
			cut := max(0., min(1., (far*float64(len(neg))-float64(ng))/float64(eq)))
			area += float64(above) + float64(ties)*.5
			partial += cut*float64(above) + .5*cut*cut*float64(ties)
		}
		closeMath(t, got.AUC, area/float64(len(pos)*len(neg)))
		closeMath(t, got.PartialAUC, partial/float64(len(pos)*len(neg)))
		minCost, bestTA := 1., 0
		for _, cut := range rows {
			tp, fp := 0, 0
			for _, x := range rows {
				if x.Score >= cut.Score {
					if x.Genuine {
						tp++
					} else {
						fp++
					}
				}
			}
			minCost = min(minCost, 1-float64(tp)/float64(len(pos))+cost*float64(fp)/float64(len(neg)))
			if fp <= limit {
				bestTA = max(bestTA, tp)
			}
		}
		// Solve FAR = FRR by bisecting the common rate. At the candidate
		// FAR, allocate the available impostor mass to descending score ties.
		lo, hi := 0., 1.
		for it := 0; it < 55; it++ {
			target := (lo + hi) / 2
			tpr := 0.
			for _, sp := range pos {
				higher, tied := 0, 0
				for _, sn := range neg {
					if sn > sp {
						higher++
					}
					if sn == sp {
						tied++
					}
				}
				available := target*float64(len(neg)) - float64(higher)
				if tied == 0 {
					if available >= 0 {
						tpr++
					}
				} else {
					tpr += min(1., max(0., available/float64(tied)))
				}
			}
			if 1-tpr/float64(len(pos)) > target {
				lo = target
			} else {
				hi = target
			}
		}
		closeMath(t, got.EER, (lo+hi)/2)
		closeMath(t, got.MinCost, minCost)
		if got.BudgetPoint.TrueAccept != bestTA || got.BudgetPoint.FalseAccept > limit {
			t.Fatal(got)
		}
	}
}

func TestMathAssociationExhaustive(t *testing.T) {
	r := rand.New(rand.NewSource(639))
	for trial := 0; trial < 600; trial++ {
		n, m := 1+r.Intn(5), r.Intn(6)
		scores := make([][]int, n)
		for i := range scores {
			scores[i] = make([]int, m)
			for j := range scores[i] {
				v := r.Intn(8) - 3
				if v == 4 {
					v = ForbiddenAssociation
				}
				scores[i][j] = v
			}
		}
		got, e := AnalyzeAssociations(scores)
		if e != nil {
			t.Fatal(e)
		}
		base, loss, e := AssociationRegrets(scores)
		if e != nil {
			t.Fatal(e)
		}
		if !slices.Equal(base.Columns, got.Best.Columns) {
			t.Fatal("unstable optimum")
		}
		by, weak := make([]int64, min(n, m)+1), make([]int, min(n, m)+1)
		for k := range by {
			by[k] = math.MinInt64
			weak[k] = ForbiddenAssociation
		}
		alt := make([]int64, n)
		for i := range alt {
			alt[i] = math.MinInt64
		}
		assign := unmatched(n)
		var visit func(int, int, int, int, int64)
		visit = func(i, used, k, bottleneck int, sum int64) {
			if i == n {
				by[k] = max(by[k], sum)
				weak[k] = max(weak[k], bottleneck)
				for row, x := range assign {
					if x != base.Columns[row] {
						alt[row] = max(alt[row], sum)
					}
				}
				return
			}
			assign[i] = -1
			visit(i+1, used, k, bottleneck, sum)
			for j, v := range scores[i] {
				if v != ForbiddenAssociation && used&(1<<j) == 0 {
					assign[i] = j
					visit(i+1, used|(1<<j), k+1, min(bottleneck, v), sum+int64(v))
				}
			}
		}
		visit(0, 0, 0, 1000000, 0)
		if base.Utility != slices.Max(by) {
			t.Fatalf("trial %d best %v scores %v", trial, base, scores)
		}
		for k, v := range by {
			if k < len(got.ByCount) {
				item := got.ByCount[k]
				seen := 0
				count := 0
				var sum int64
				for i, j := range item.Columns {
					if j < 0 {
						continue
					}
					if seen&(1<<j) != 0 || scores[i][j] == ForbiddenAssociation {
						t.Fatal(item)
					}
					seen |= 1 << j
					count++
					sum += int64(scores[i][j])
				}
				if item.Utility != v || sum != v || count != k {
					t.Fatal("frontier", trial, k, item, v)
				}
			} else if v != math.MinInt64 {
				t.Fatal("missing matching", k)
			}
			if k > 0 {
				b, ok, e := AssociationBottleneck(scores, k)
				if e != nil || ok != (v != math.MinInt64) || ok && b != weak[k] {
					t.Fatal("bottleneck", k, b, weak[k], e)
				}
			}
		}
		if got.MostMatches.Utility != by[len(got.ByCount)-1] {
			t.Fatal("cardinality")
		}
		for i, v := range alt {
			want := int64(-1)
			if v != math.MinInt64 {
				want = base.Utility - v
			}
			if loss[i] != want {
				t.Fatal("regret", trial, i, loss[i], want, scores, base)
			}
		}
	}
}

func TestMathWorkloads(t *testing.T) {
	if os.Getenv("FACEX_GALLERY_SCALE") != "1" {
		t.Skip("requires scale run")
	}
	r := rand.New(rand.NewSource(15))
	rows := make([][]float32, 128)
	for i := range rows {
		rows[i] = make([]float32, OutputSize)
		for j := range rows[i] {
			rows[i][j] = float32(r.NormFloat64())
		}
	}
	start := time.Now()
	if _, e := CoverViews(rows, 2); e != nil {
		t.Fatal(e)
	}
	t.Log("exact two exemplars, 128 views", time.Since(start))
	trials := make([]VerificationTrial, 1000000)
	for i := range trials {
		trials[i] = VerificationTrial{float64(r.Intn(10000)), i%2 == 0}
	}
	start = time.Now()
	if _, e := EvaluateVerification(trials, .001, 1, 1, 50); e != nil {
		t.Fatal(e)
	}
	t.Log("verification, 1000000 pairs", time.Since(start))
	s := make([][]int, 32)
	for i := range s {
		s[i] = make([]int, 48)
		for j := range s[i] {
			s[i][j] = r.Intn(2000001) - 1000000
		}
	}
	start = time.Now()
	if _, _, e := AssociationRegrets(s); e != nil {
		t.Fatal(e)
	}
	t.Log("assignment and all regrets, 32 by 48", time.Since(start))
}

// Assemble a fixed batch of 15 held-out crop embeddings; rows here do not
// represent a camera frame or independent visits.
func TestMathModelFixture(t *testing.T) {
	dir := os.Getenv("FACEX_GALLERY_FIXTURE")
	if dir == "" {
		t.Skip("requires gallery fixture")
	}
	f, e := os.Open(filepath.Join(dir, "rows"))
	if e != nil {
		t.Fatal(e)
	}
	defer f.Close()
	qf, e := os.Open(filepath.Join(dir, "queries"))
	if e != nil {
		t.Fatal(e)
	}
	defer qf.Close()
	const persons = 15
	templates := make([][]float64, persons)
	for i := range templates {
		if _, e = f.Seek(int64(i)*4369*OutputSize*4, 0); e != nil {
			t.Fatal(e)
		}
		views := make([][]float32, 8)
		for j := range views {
			views[j] = make([]float32, OutputSize)
			if e = binary.Read(f, binary.LittleEndian, views[j]); e != nil {
				t.Fatal(e)
			}
		}
		fit, e := FitDirections(views, nil)
		if e != nil || fit.Template == nil {
			t.Fatal(e)
		}
		templates[i] = fit.Template
		one, e := CoverViews(views, 1)
		if e != nil {
			t.Fatal(e)
		}
		two, e := CoverViews(views, 2)
		if e != nil || two.MinCosine < one.MinCosine {
			t.Fatal(i, e)
		}
	}
	matrix := make([][]int, persons)
	var pairs []VerificationTrial
	for i := range matrix {
		q := make([]float32, OutputSize)
		if e = binary.Read(qf, binary.LittleEndian, q); e != nil {
			t.Fatal(e)
		}
		unit, e := unitViews([][]float32{q})
		if e != nil {
			t.Fatal(e)
		}
		matrix[i] = make([]int, persons)
		for j, template := range templates {
			s := max(-1., min(1., directionDot(unit[0], template)))
			pairs = append(pairs, VerificationTrial{s, i == j})
			matrix[i][j] = int(math.Round(s * 1e6))
		}
	}
	match, regrets, e := AssociationRegrets(matrix)
	if e != nil {
		t.Fatal(e)
	}
	right := 0
	for i, j := range match.Columns {
		if i == j {
			right++
		}
	}
	curve, e := EvaluateVerification(pairs, .01, 1, 1, 0)
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("fixture assigned_correct=%d/%d min_regret=%d auc=%.6f eer=%.6f budget0_true=%d/%d", right, persons, slices.Min(regrets), curve.AUC, curve.EER, curve.BudgetPoint.TrueAccept, persons)
}
