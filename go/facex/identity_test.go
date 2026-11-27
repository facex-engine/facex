package facex

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

func basis(i int) []float32 {
	row := make([]float32, OutputSize)
	row[i] = 1
	return row
}

func ExampleIdentityGallery_Identify() {
	// In the application each row and the query come from FaceX.Embed.
	front, side, other := make([]float32, OutputSize), make([]float32, OutputSize), make([]float32, OutputSize)
	front[0], side[1], other[2] = 1, 1, 1
	g, err := NewIdentityGallery([][]float32{front, side, other}, []int64{41, 41, 7})
	if err != nil {
		panic(err)
	}
	id, err := g.Identify(side, 0.85, 0.05)
	fmt.Println(id, err)
	// Output: 41 <nil>
}

func TestIdentityBoundaries(t *testing.T) {
	a, b := basis(0), basis(1)
	g, err := NewIdentityGallery([][]float32{a, b, a}, []int64{41, 7, 41})
	if err != nil {
		t.Fatal(err)
	}
	a[0] = 0
	for _, tc := range []struct {
		score, margin float64
		want          int64
		resolved      bool
	}{
		{.9, .1, 41, true}, {1, 1, 41, false}, {1, math.Nextafter(1, 2), -1, false},
	} {
		id, resolved, err := g.Screen(basis(0), tc.score, tc.margin)
		if err != nil || resolved != tc.resolved || (resolved && id != tc.want) {
			t.Fatal(tc, id, resolved, err)
		}
		id, err = g.Identify(basis(0), tc.score, tc.margin)
		if err != nil || id != tc.want {
			t.Fatal(tc, id, err)
		}
	}
	for _, tc := range []struct {
		ids    []int64
		margin float64
		want   int64
	}{
		{[]int64{1, 2}, 0, -1}, {[]int64{1, 1}, 0, 1}, {[]int64{1, 1}, .01, -1},
	} {
		g, err := NewIdentityGallery([][]float32{b, b}, tc.ids)
		if err != nil {
			t.Fatal(err)
		}
		id, err := g.Identify(b, 1, tc.margin)
		if err != nil || id != tc.want {
			t.Fatal(tc, id, err)
		}
	}
}

func TestIdentityValidation(t *testing.T) {
	zero, bad := make([]float32, OutputSize), basis(0)
	bad[2] = float32(math.NaN())
	for _, row := range [][]float32{zero, bad, {1}} {
		if _, err := NewIdentityGallery([][]float32{row}, []int64{5}); err == nil {
			t.Fatal("invalid enrollment accepted")
		}
	}
	if _, err := NewIdentityGallery([][]float32{basis(0)}, []int64{-1}); err == nil {
		t.Fatal("negative ID accepted")
	}
	g, err := NewIdentityGallery([][]float32{basis(0)}, []int64{5})
	if err != nil {
		t.Fatal(err)
	}
	if id, err := g.Identify(zero, -1, 0); err != nil || id != -1 {
		t.Fatal(id, err)
	}
	if _, err := g.Identify(bad, 0, 0); err == nil {
		t.Fatal("invalid query accepted")
	}
	if _, err := g.Identify(zero, math.NaN(), 0); err == nil {
		t.Fatal("invalid threshold accepted")
	}
}

// Exhaustive reference uses every row and compares only distinct people.
func referenceID(rows [][]float32, ids []int64, q []float32, score, margin float64) int64 {
	scores := map[int64]float64{}
	for i, row := range rows {
		var dot, a, b float64
		for j, x := range row {
			f, v := float64(x), float64(q[j])
			dot += f * v
			a += f * f
			b += v * v
		}
		value := max(-1., min(1., dot/(math.Sqrt(a)*math.Sqrt(b))))
		if prev, ok := scores[ids[i]]; !ok || value > prev {
			scores[ids[i]] = value
		}
	}
	for id, v := range scores {
		if v < score {
			continue
		}
		win := true
		for other, w := range scores {
			if other != id && (v <= w || v-w < margin) {
				win = false
				break
			}
		}
		if win {
			return id
		}
	}
	return -1
}

func TestIdentityConcurrentReference(t *testing.T) {
	r := rand.New(rand.NewSource(2069))
	random := func() []float32 {
		row := make([]float32, OutputSize)
		for i := range row {
			row[i] = float32(r.NormFloat64())
		}
		return row
	}
	rows, ids := make([][]float32, 96), make([]int64, 96)
	for i := range rows {
		rows[i] = random()
		ids[i] = int64(i%4) * 10000000000
	}
	rows[1][0] = math.MaxFloat32
	g, err := NewIdentityGallery(rows, ids)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		q := random()
		if i%2 == 0 {
			q = rows[i]
		}
		wg.Add(1)
		go func(q []float32) {
			defer wg.Done()
			for _, params := range [][2]float64{{.85, .05}, {.05, 0}, {1, 0}, {-1, .1}} {
				want := referenceID(rows, ids, q, params[0], params[1])
				got, err := g.Identify(q, params[0], params[1])
				if err != nil || want != got {
					t.Errorf("want %d got %d err %v", want, got, err)
					return
				}
				got, ok, err := g.Screen(q, params[0], params[1])
				if err != nil || ok && got != want {
					t.Errorf("incorrect screen: %d %t %v", got, ok, err)
					return
				}
			}
		}(q)
	}
	wg.Wait()
}
