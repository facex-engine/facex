package facex

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"sync"
	"testing"
	"time"
)

func ExampleLiveGallery() {
	var live LiveGallery
	front := make([]float32, OutputSize)
	front[0] = 1 // In the application use the output of FaceX.Embed.
	_, err := live.Apply(0, []EnrollmentChange{{ID: 42, Views: [][]float32{front}}})
	if err != nil {
		panic(err)
	}
	gallery, revision := live.Snapshot()
	id, err := gallery.Identify(front, .85, 0)
	fmt.Println(revision, id, err)
	// Output: 1 42 <nil>
}

func identifyView(t *testing.T, g IdentityGallery, row []float32, want int64) {
	t.Helper()
	got, err := g.Identify(row, .8, 0)
	if err != nil || got != want {
		t.Fatalf("got %d want %d: %v", got, want, err)
	}
}

func TestLiveGalleryTransaction(t *testing.T) {
	var live LiveGallery
	first, revision := live.Snapshot()
	if revision != 0 {
		t.Fatal(revision)
	}
	identifyView(t, first, basis(0), -1)
	a, b := basis(0), basis(1)
	initial := []EnrollmentChange{{ID: 7, Views: [][]float32{a}}, {ID: 8, Views: [][]float32{b}}}
	if rev, err := live.Apply(0, initial); err != nil || rev != 1 {
		t.Fatal(rev, err)
	}
	initial[0].Views[0][0] = 0
	pinned, _ := live.Snapshot()
	identifyView(t, pinned, basis(0), 7)
	for _, invalid := range [][]EnrollmentChange{
		{{ID: 7, Remove: true}, {ID: 9, Views: [][]float32{{1}}}},
		{{ID: 7, Remove: true}, {ID: 7, Views: [][]float32{basis(0)}}},
		{{ID: -1, Remove: true}}, {{ID: 8}}, {{ID: 8, Views: [][]float32{b}, Remove: true}},
	} {
		rev, err := live.Apply(1, invalid)
		if err == nil || rev != 1 {
			t.Fatal(rev, err)
		}
		g, r := live.Snapshot()
		if r != 1 {
			t.Fatal(r)
		}
		identifyView(t, g, basis(0), 7)
	}
	if rev, err := live.Apply(1, []EnrollmentChange{{ID: 90, Remove: true}}); err != nil || rev != 1 {
		t.Fatal(rev, err)
	}
	if _, err := live.Apply(1, []EnrollmentChange{{ID: 7, Remove: true}, {ID: 9, Views: [][]float32{basis(0)}}}); err != nil {
		t.Fatal(err)
	}
	next, rev := live.Snapshot()
	if rev != 2 || &next.people[0].rows[0] != &pinned.people[1].rows[0] {
		t.Fatal("unchanged enrollment was rebuilt")
	}
	identifyView(t, next, basis(0), 9)
	identifyView(t, pinned, basis(0), 7)
	if _, err := live.Apply(rev, []EnrollmentChange{{ID: 8, Remove: true}, {ID: 9, Remove: true}}); err != nil {
		t.Fatal(err)
	}
	empty, _ := live.Snapshot()
	out, err := empty.IdentifyBatch(context.Background(), [][]float32{b}, .8, 0, 1)
	if err != nil || !slices.Equal(out, []int64{-1}) {
		t.Fatal(out, err)
	}
}

func TestLiveConcurrentWriters(t *testing.T) {
	var live LiveGallery
	start := make(chan struct{})
	results := make(chan error, 16)
	for i := 0; i < cap(results); i++ {
		go func(id int64) {
			<-start
			_, err := live.Apply(0, []EnrollmentChange{{ID: id, Views: [][]float32{basis(0)}}})
			results <- err
		}(int64(i))
	}
	close(start)
	wins := 0
	for i := 0; i < cap(results); i++ {
		if err := <-results; err == nil {
			wins++
		} else if !errors.Is(err, ErrGalleryConflict) {
			t.Fatal(err)
		}
	}
	if wins != 1 {
		t.Fatal("unexpected successful writers", wins)
	}
}

func TestLiveConcurrentReaders(t *testing.T) {
	var live LiveGallery
	a, b := basis(0), basis(1)
	apply := func(expected uint64) {
		x, y := a, b
		if expected%2 == 1 {
			x, y = b, a
		}
		_, err := live.Apply(expected, []EnrollmentChange{
			{ID: 1, Views: [][]float32{x}}, {ID: 2, Views: [][]float32{y}},
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	apply(0)
	var readers, pinned sync.WaitGroup
	release := make(chan struct{})
	for i := 0; i < 8; i++ {
		readers.Add(1)
		pinned.Add(1)
		go func() {
			defer readers.Done()
			for n := 0; n < 60; n++ {
				g, rev := live.Snapshot()
				if n == 0 {
					pinned.Done()
					<-release
				}
				want := []int64{1, 2}
				if rev%2 == 0 {
					want = []int64{2, 1}
				}
				got, err := g.IdentifyBatch(context.Background(), [][]float32{a, b}, .9, .05, 2)
				if err != nil || !slices.Equal(got, want) {
					t.Errorf("mixed revision %d: %v want %v: %v", rev, got, want, err)
					return
				}
			}
		}()
	}
	pinned.Wait()
	apply(1)
	close(release)
	for rev := uint64(2); rev < 60; rev++ {
		apply(rev)
	}
	readers.Wait()
}

func TestLiveGalleryScale(t *testing.T) {
	dir := os.Getenv("FACEX_GALLERY_FIXTURE")
	if dir == "" {
		t.Skip("requires the model fixture gallery")
	}
	const people, per = 15, 4369
	rows, ids := make([][]float32, people*per), make([]int64, people*per)
	f, err := os.Open(filepath.Join(dir, "rows"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	input := bufio.NewReaderSize(f, 1<<20)
	for i := range rows {
		rows[i] = make([]float32, OutputSize)
		if err := binary.Read(input, binary.LittleEndian, rows[i]); err != nil {
			t.Fatal(err)
		}
		ids[i] = int64(i/per)
	}
	var live LiveGallery
	initial := make([]EnrollmentChange, people)
	for i := range initial {
		initial[i] = EnrollmentChange{ID: int64(i), Views: rows[i*per:(i+1)*per]}
	}
	rev, err := live.Apply(0, initial)
	if err != nil {
		t.Fatal(err)
	}
	var times [2][]time.Duration
	var allocated [2][]uint64
	for trial := 0; trial < 5; trial++ {
		owner := trial
		// Replace one photo, keeping the other 4368 views for this person.
		rows[owner*per] = basis(400+trial)
		change := []EnrollmentChange{{ID: int64(owner), Views: rows[owner*per:(owner+1)*per]}}
		var rebuilt *IdentityGallery
		for k := 0; k < 2; k++ {
			m := (trial+k)%2
			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			start := time.Now()
			if m == 0 {
				rebuilt, err = NewIdentityGallery(rows, ids)
			} else {
				rev, err = live.Apply(rev, change)
			}
			elapsed := time.Since(start)
			runtime.ReadMemStats(&after)
			if err != nil {
				t.Fatal(err)
			}
			times[m] = append(times[m], elapsed)
			allocated[m] = append(allocated[m], after.TotalAlloc-before.TotalAlloc)
		}
		current, _ := live.Snapshot()
		samples := [][]float32{rows[owner*per], rows[len(rows)-1], basis(300)}
		for i := 0; i < people; i++ {
			samples = append(samples, rows[i*per+1])
		}
		for _, q := range samples {
			want, err := rebuilt.Identify(q, .85, .05)
			if err != nil {
				t.Fatal(err)
			}
			got, err := current.Identify(q, .85, .05)
			if err != nil || got != want {
				t.Fatalf("trial %d got %d want %d: %v", trial, got, want, err)
			}
		}
		identifyView(t, current, rows[owner*per], int64(owner))
	}
	for m, name := range []string{"rebuild", "live"} {
		sort.Slice(times[m], func(i, j int) bool { return times[m][i] < times[m][j] })
		slices.Sort(allocated[m])
		t.Logf("%s rows=%d trials=5 median=%v median_allocated_bytes=%d", name, len(rows), times[m][2], allocated[m][2])
	}
}
