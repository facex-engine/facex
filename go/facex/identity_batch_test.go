package facex

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

func TestIdentifyBatchContract(t *testing.T) {
	rows := [][]float32{basis(0), basis(1), basis(2), basis(0)}
	ids := []int64{100, 200, 300, 100}
	g, err := NewIdentityGallery(rows, ids)
	if err != nil {
		t.Fatal(err)
	}
	q := [][]float32{rows[2], rows[0], rows[1], make([]float32, OutputSize)}
	for _, n := range []int{1, 2, 1024} {
		got, err := g.IdentifyBatch(context.Background(), q, 1, 1, n)
		if err != nil {
			t.Fatal(err)
		}
		for i, want := range []int64{300, 100, 200, -1} {
			if got[i] != want {
				t.Fatalf("query %d got %d want %d", i, got[i], want)
			}
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got, err := g.IdentifyBatch(ctx, q, .7, .05, 2)
	if got != nil || !errors.Is(err, context.Canceled) {
		t.Fatal(got, err)
	}
	if got, err = g.IdentifyBatch(context.Background(), [][]float32{q[0], {1}}, .7, .05, 2); err == nil || got != nil {
		t.Fatal("expected atomic validation failure")
	}
	if got, err = g.IdentifyBatch(context.Background(), nil, .7, .05, 2); err != nil || len(got) != 0 {
		t.Fatal(got, err)
	}
}

// fullFallback mirrors the previous refinement using cached row norms.
func fullFallback(g *IdentityGallery, q []float32, score, margin float64) int64 {
	norm, _ := vectorNorm(q)
	id, resolved, _ := g.screen(context.Background(), q, norm, score, margin)
	if resolved {
		return id
	}
	best, second := math.Inf(-1), math.Inf(-1)
	for i := range g.people {
		p := &g.people[i]
		value := math.Inf(-1)
		for j := range p.norms {
			value = max(value, p.cosine(j, q, norm))
		}
		if value > best {
			best, second, id = value, best, p.id
		} else {
			second = max(second, value)
		}
	}
	if best >= score && best > second && best-second >= margin {
		return id
	}
	return rejectedID
}

func TestIdentityGalleryScale(t *testing.T) {
	if os.Getenv("FACEX_GALLERY_SCALE") != "1" {
		t.Skip("set FACEX_GALLERY_SCALE=1 to run the large gallery")
	}
	const people, per = 15, 4369
	rng := rand.New(rand.NewSource(2069))
	rows := make([][]float32, people*per)
	ids := make([]int64, len(rows))
	queries := make([][]float32, people*3)
	for i := range rows {
		person, view := i/per, i%8
		row := make([]float32, OutputSize)
		for j := range row {
			row[j] = float32(.004 * rng.NormFloat64())
		}
		row[person] += 1
		row[people+person*8+view] += .14
		rows[i], ids[i] = row, int64(person)
	}
	for i := 0; i < people; i++ {
		queries[i] = basis(i)
		queries[i+people] = basis(i)
		queries[i+people][(i+1)%people] = 1
		queries[i+2*people] = basis(200+i)
	}
	if dir := os.Getenv("FACEX_GALLERY_FIXTURE"); dir != "" {
		readRows := func(name string, vectors [][]float32) {
			f, err := os.Open(filepath.Join(dir, name))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			stat, err := f.Stat()
			if err != nil || stat.Size() != int64(len(vectors)*OutputSize*4) {
				t.Fatalf("incorrect fixture size for %s: %v", name, err)
			}
			reader := bufio.NewReaderSize(f, 1<<20)
			for _, row := range vectors {
				if err := binary.Read(reader, binary.LittleEndian, row); err != nil {
					t.Fatal(err)
				}
			}
		}
		readRows("rows", rows)
		readRows("queries", queries)
	}
	g, err := NewIdentityGallery(rows, ids)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	want := make([]int64, len(queries))
	for i, q := range queries {
		want[i] = referenceID(rows, ids, q, .65, .05)
	}
	rows, ids = nil, nil
	var timings [2][]time.Duration
	fallback := 0
	for pass := 0; pass < 4; pass++ {
		for i, q := range queries {
			_, resolved, err := g.Screen(q, .65, .05)
			if err != nil {
				t.Fatal(err)
			}
			if !resolved && pass == 0 {
				fallback++
			}
			for k := 0; k < 2; k++ {
				var result int64
				method := (pass+i+k)%2
				start := time.Now()
				if method == 0 {
					result = fullFallback(g, q, .65, .05)
				} else {
					result, err = g.Identify(q, .65, .05)
				}
				elapsed := time.Since(start)
				if err != nil || result != want[i] {
					t.Fatalf("method %d query %d got %d want %d: %v", method, i, result, want[i], err)
				}
				if !resolved && pass > 0 {
					timings[method] = append(timings[method], elapsed)
				}
			}
		}
	}
	if fallback == 0 {
		t.Fatal("workload did not exercise fallback")
	}
	for method, samples := range timings {
		sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
		t.Logf("fallback method=%d samples=%d p50=%v p95=%v", method, len(samples), samples[(len(samples)-1)/2], samples[(95*len(samples)+99)/100-1])
	}
	for _, n := range []int{1, 4} {
		start := time.Now()
		result, err := g.IdentifyBatch(ctx, queries, .65, .05, n)
		if err != nil {
			t.Fatal(err)
		}
		elapsed := time.Since(start)
		for i, id := range result {
			if id != want[i] {
				t.Fatalf("batch %d query %d got %d want %d", n, i, id, want[i])
			}
		}
		t.Logf("batch workers=%d queries=%d wall=%v", n, len(result), elapsed)
	}
	t.Logf("enrollment=%d distinct_queries=%d fallback_queries=%d", people*per, len(queries), fallback)
}
