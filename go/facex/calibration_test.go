package facex

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"
)

func replayThresholds(probes []ScoredProbe, score, margin float64) (c CalibrationCounts) {
	for _, p := range probes {
		if p.BestID < 0 || p.Score < score || p.Margin < margin {
			continue
		}
		switch {
		case p.ExpectedID == -1:
			c.UnknownAccepted++
		case p.ExpectedID != p.BestID:
			c.KnownWrong++
		default:
			c.Correct++
		}
	}
	return c
}

func bruteThresholds(probes []ScoredProbe, unknown, wrong int) (CalibrationResult, bool) {
	scores, margins := []float64{1, -1}, []float64{2, 0}
	for _, v := range probes {
		scores = append(scores, v.Score)
		margins = append(margins, v.Margin)
	}
	var best CalibrationResult
	found := false
	for _, s := range scores {
		for _, m := range margins {
			c := replayThresholds(probes, s, m)
			if c.UnknownAccepted > unknown || c.KnownWrong > wrong {
				continue
			}
			loss := c.UnknownAccepted + c.KnownWrong
			prev := best.Counts.UnknownAccepted + best.Counts.KnownWrong
			if !found || c.Correct > best.Counts.Correct ||
				c.Correct == best.Counts.Correct && (loss < prev ||
					loss == prev && (s > best.MinScore || s == best.MinScore && m > best.MinMargin)) {
				found, best = true, CalibrationResult{s, m, c}
			}
		}
	}
	return best, found
}

func TestThresholdSweepAgainstExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(2069))
	for trial := 0; trial < 800; trial++ {
		probes := make([]ScoredProbe, 1+rng.Intn(18))
		for i := range probes {
			probes[i] = ScoredProbe{ExpectedID: int64(rng.Intn(4) - 1), BestID: int64(rng.Intn(4) - 1),
				Score: float64(rng.Intn(9)-4) / 4, Margin: float64(rng.Intn(9)) / 4}
		}
		u, w := rng.Intn(4), rng.Intn(4)
		want, ok := bruteThresholds(probes, u, w)
		got, err := FitThresholds(probes, u, w)
		if !ok {
			if !errors.Is(err, ErrNoFeasibleThresholds) {
				t.Fatal(trial, err)
			}
		} else if err != nil || got != want {
			t.Fatalf("trial %d got %+v want %+v err %v", trial, got, want, err)
		}
	}
}

func TestThresholdLimits(t *testing.T) {
	for _, p := range []ScoredProbe{{Score: math.NaN()}, {Margin: math.Inf(1)}, {BestID: -2}} {
		if _, err := FitThresholds([]ScoredProbe{p}, 0, 0); err == nil {
			t.Fatal("invalid probe accepted")
		}
	}
	if _, err := FitThresholds([]ScoredProbe{{ExpectedID: -1, BestID: 4, Score: 1, Margin: 2}}, 0, 0); !errors.Is(err, ErrNoFeasibleThresholds) {
		t.Fatal(err)
	}
	if _, err := FitThresholds(nil, 0, 0); err == nil {
		t.Fatal("empty input accepted")
	}
}

func TestCalibrationAgreesWithIdentify(t *testing.T) {
	a, b, c := basis(0), basis(1), basis(2)
	rows := [][]float32{a, b, c, a}
	labels := []int64{91, 28, 35, 91}
	for _, n := range []int{1, 3, 4} {
		g, err := NewIdentityGallery(rows[:n], labels[:n])
		if err != nil {
			t.Fatal(err)
		}
		mid := basis(0)
		mid[1] = 1
		queries := [][]float32{a, b, c, mid, make([]float32, OutputSize), basis(3)}
		truth := []int64{91, 28, 999, 35, -1, 17}
		probes, err := g.ScoreProbes(queries, truth)
		if err != nil || probes[2].ExpectedID != -1 {
			t.Fatal(probes, err)
		}
		for _, s := range []float64{-1, 0, .65, 1} {
			for _, m := range []float64{0, .05, 1, 2} {
				for i, p := range probes {
					got, err := g.Identify(queries[i], s, m)
					want := int64(-1)
					if p.Score >= s && p.Margin >= m {
						want = p.BestID
					}
					if err != nil || got != want {
						t.Fatal(n, i, s, m, got, want, err)
					}
				}
			}
		}
	}
}

func TestCalibrationMillion(t *testing.T) {
	if os.Getenv("FACEX_GALLERY_SCALE") != "1" {
		t.Skip("requires scale run")
	}
	rng := rand.New(rand.NewSource(77))
	probes := make([]ScoredProbe, 1_000_000)
	for i := range probes {
		s, m := rng.Float64(), rng.Float64()
		p := ScoredProbe{ExpectedID: 1, BestID: 1, Score: .5 + .5*s, Margin: .1 + .8*m}
		switch i % 5 {
		case 0:
			p.ExpectedID, p.Score, p.Margin = -1, .25+.65*s, .65*m
		case 1:
			p.BestID, p.Score, p.Margin = 2, .5+.48*s, .35*m
		}
		probes[i] = p
	}
	start := time.Now()
	got, err := FitThresholds(probes, 100, 100)
	elapsed := time.Since(start)
	if err != nil || got.Counts != replayThresholds(probes, got.MinScore, got.MinMargin) ||
		got.Counts.UnknownAccepted > 100 || got.Counts.KnownWrong > 100 {
		t.Fatal(got, err)
	}
	// Independent one-dimensional optimization. With margin fixed at zero
	// the first threshold violating either budget marks the end of the sweep.
	sort.Slice(probes, func(i, j int) bool { return probes[i].Score > probes[j].Score })
	var correct, u, w, best int
	for i := 0; i < len(probes); {
		score := probes[i].Score
		for i < len(probes) && probes[i].Score == score {
			p := probes[i]
			switch {
			case p.ExpectedID < 0:
				u++
			case p.ExpectedID != p.BestID:
				w++
			default:
				correct++
			}
			i++
		}
		if u > 100 || w > 100 {
			break
		}
		best = correct
	}
	if got.Counts.Correct < best {
		t.Fatal("two thresholds lost feasible matches")
	}
	t.Logf("synthetic probes=%d fit=%v score=%.8f margin=%.8f correct=%d unknown_accepted=%d known_wrong=%d score_only_correct=%d",
		len(probes), elapsed, got.MinScore, got.MinMargin, got.Counts.Correct, got.Counts.UnknownAccepted, got.Counts.KnownWrong, best)
}
