package facex

import (
	"context"
	"errors"
	"fmt"
	"math"
)

const (
	rejectedID      int64 = -1
	maxIdentityCaps       = 8
	cosinePad             = 1e-12
)

// IdentityGallery is a fixed snapshot of enrollment embeddings, grouped by person.
// Once constructed, it supports concurrent calls to Identify and Screen.
type IdentityGallery struct {
	people []identityGroup
}

type identityGroup struct {
	id    int64
	rows  []float32
	norms []float64
	caps  []identityCap
}

type identityCap struct {
	row                  int
	cosRadius, sinRadius float64
	members []int
}

func vectorNorm(row []float32) (float64, error) {
	if len(row) != OutputSize {
		return 0, fmt.Errorf("facex: embedding has %d components, want %d", len(row), OutputSize)
	}
	var sum float64
	for _, x := range row {
		sum += float64(x) * float64(x)
	}
	if math.IsNaN(sum) || math.IsInf(sum, 0) {
		return 0, errors.New("facex: nonfinite embedding")
	}
	return math.Sqrt(sum), nil
}

func (p *identityGroup) row(i int) []float32 {
	return p.rows[i*OutputSize : (i+1)*OutputSize]
}

func (p *identityGroup) cosine(i int, query []float32, norm float64) float64 {
	var dot float64
	for j, v := range p.row(i) {
		dot += float64(v) * float64(query[j])
	}
	return max(-1., min(1., dot/(p.norms[i]*norm)))
}

// buildCaps chooses actual enrolled rows as witnesses. Each step updates the
// closest witness for every row, then selects the least well covered row.
func (p *identityGroup) buildCaps() {
	n := len(p.norms)
	nearest := make([]float64, n)
	owner := make([]int, n)
	for i := range nearest {
		nearest[i] = -2
	}
	seed := 0
	for len(p.caps) < min(n, maxIdentityCaps) {
		label := len(p.caps)
		p.caps = append(p.caps, identityCap{row: seed, cosRadius: 1})
		next := 0
		for i := range nearest {
			d := p.cosine(i, p.row(seed), p.norms[seed])
			if d > nearest[i] {
				nearest[i], owner[i] = d, label
			}
			if nearest[i] < nearest[next] {
				next = i
			}
		}
		if nearest[next] >= 1-cosinePad {
			break
		}
		seed = next
	}
	for i, label := range owner {
		c := &p.caps[label]
		c.cosRadius = min(c.cosRadius, nearest[i])
		c.members = append(c.members, i)
	}
	for i := range p.caps {
		c := &p.caps[i]
		c.cosRadius = max(-1., c.cosRadius-cosinePad)
		c.sinRadius = math.Sqrt((1-c.cosRadius)*(1+c.cosRadius))
	}
}

func newIdentityGroup(id int64, count int) identityGroup {
	return identityGroup{
		id:    id,
		rows:  make([]float32, 0, count*OutputSize),
		norms: make([]float64, 0, count),
	}
}

func (p *identityGroup) addView(row []float32) error {
	norm, err := vectorNorm(row)
	if err != nil {
		return err
	}
	if norm == 0 {
		return errors.New("facex: zero enrollment vector")
	}
	p.rows = append(p.rows, row...)
	p.norms = append(p.norms, norm)
	return nil
}

// NewIdentityGallery copies the enrollment vectors. Repeat an ID for additional
// photos of the same person. IDs must be nonnegative and vectors finite and nonzero.
func NewIdentityGallery(embeddings [][]float32, ids []int64) (*IdentityGallery, error) {
	if len(ids) == 0 || len(ids) != len(embeddings) {
		return nil, errors.New("facex: expected one ID per row of a nonempty gallery")
	}
	index := make(map[int64]int)
	g := &IdentityGallery{}
	counts := []int{}
	for _, id := range ids {
		if id < 0 {
			return nil, errors.New("facex: negative identity ID")
		}
		i, ok := index[id]
		if !ok {
			i = len(g.people)
			index[id] = i
			g.people = append(g.people, identityGroup{id: id})
			counts = append(counts, 0)
		}
		counts[i]++
	}
	for i, count := range counts {
		g.people[i] = newIdentityGroup(g.people[i].id, count)
	}
	for i, row := range embeddings {
		if err := g.people[index[ids[i]]].addView(row); err != nil {
			return nil, fmt.Errorf("enrollment row %d: %w", i, err)
		}
	}
	for i := range g.people {
		g.people[i].buildCaps()
	}
	return g, nil
}

func (p *identityGroup) capBounds(c *identityCap, query []float32, norm float64) (float64, float64) {
	dot := p.cosine(c.row, query, norm)
	d := min(1., dot+cosinePad)
	upper := 1.
	if d < c.cosRadius {
		upper = d*c.cosRadius + math.Sqrt((1-d)*(1+d))*c.sinRadius
	}
	return dot-cosinePad, upper+cosinePad
}

func (g *IdentityGallery) validate(score, margin float64) error {
	if g == nil || g.people == nil {
		return errors.New("facex: empty identity gallery")
	}
	if !(score >= -1 && score <= 1 && margin >= 0 && margin <= 2) {
		return errors.New("facex: invalid cosine or margin threshold")
	}
	return nil
}

func (g *IdentityGallery) screen(ctx context.Context, query []float32, norm, score, margin float64) (int64, bool, float64) {
	if len(g.people) == 0 || norm == 0 || (len(g.people) == 1 && margin > 0) {
		return rejectedID, true, 0
	}
	bestLower, first, second := math.Inf(-1), math.Inf(-1), math.Inf(-1)
	winner, upperWinner := 0, 0
	for i := range g.people {
		if i%64 == 0 && ctx.Err() != nil {
			return rejectedID, false, 0
		}
		p := &g.people[i]
		lo, hi := math.Inf(-1), math.Inf(-1)
		for j := range p.caps {
			lower, upper := p.capBounds(&p.caps[j], query, norm)
			lo, hi = max(lo, lower), max(hi, upper)
		}
		if lo > bestLower {
			bestLower, winner = lo, i
		}
		if hi > first {
			first, second, upperWinner = hi, first, i
		} else {
			second = max(second, hi)
		}
	}
	if first < score {
		return rejectedID, true, bestLower
	}
	rival := first
	if winner == upperWinner {
		rival = second
	}
	if bestLower >= score && bestLower > rival && bestLower-rival >= margin {
		return g.people[winner].id, true, bestLower
	}
	return rejectedID, false, bestLower
}

// Screen uses enrolled witnesses to bound the decision. A resolved ID of -1
// means rejection. With resolved=false, Identify can complete the exact lookup.
func (g *IdentityGallery) Screen(query []float32, minScore, minMargin float64) (id int64, resolved bool, err error) {
	if err = g.validate(minScore, minMargin); err != nil {
		return rejectedID, false, err
	}
	norm, err := vectorNorm(query)
	if err != nil {
		return rejectedID, false, err
	}
	id, resolved, _ = g.screen(context.Background(), query, norm, minScore, minMargin)
	return id, resolved, nil
}

// Identify returns the best person's ID or -1. Both thresholds are inclusive.
// A tied best score across different people always rejects.
func (g *IdentityGallery) Identify(query []float32, minScore, minMargin float64) (int64, error) {
	if err := g.validate(minScore, minMargin); err != nil {
		return rejectedID, err
	}
	norm, err := vectorNorm(query)
	if err != nil {
		return rejectedID, err
	}
	return g.identify(context.Background(), query, norm, minScore, minMargin), nil
}

func (g *IdentityGallery) identify(ctx context.Context, query []float32, norm, minScore, minMargin float64) int64 {
	id, resolved, lower := g.screen(ctx, query, norm, minScore, minMargin)
	if resolved {
		return id
	}
	best, second := math.Inf(-1), math.Inf(-1)
	winner := rejectedID
	for i := range g.people {
		if ctx.Err() != nil {
			return rejectedID
		}
		p := &g.people[i]
		value := math.Inf(-1)
		// Every acceptable winner reaches minScore and at least the best
		// witness score. A cap below this floor minus the required margin
		// cannot change any accepted decision, even as a runner-up.
		floor := max(minScore, max(best, lower)) - minMargin - cosinePad
		for j := range p.caps {
			c := &p.caps[j]
			_, upper := p.capBounds(c, query, norm)
			if upper < floor || upper < value {
				continue
			}
			for n, row := range c.members {
				if n%256 == 0 && ctx.Err() != nil {
					return rejectedID
				}
				value = max(value, p.cosine(row, query, norm))
			}
		}
		if value > best {
			best, second, winner = value, best, p.id
		} else {
			second = max(second, value)
		}
	}
	if best >= minScore && best > second && best-second >= minMargin {
		return winner
	}
	return rejectedID
}
