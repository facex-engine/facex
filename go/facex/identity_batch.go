package facex

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
)

// IdentifyBatch identifies ready embeddings in input order using at most workers
// goroutines and GOMAXPROCS CPUs. Cancellation or invalid input returns no IDs.
// Callers must keep queries unchanged until the call returns.
func (g *IdentityGallery) IdentifyBatch(ctx context.Context, queries [][]float32, minScore, minMargin float64, workers int) ([]int64, error) {
	if ctx == nil || workers < 1 {
		return nil, errors.New("facex: expected a context and a positive worker limit")
	}
	if err := g.validate(minScore, minMargin); err != nil {
		return nil, err
	}
	norms := make([]float64, len(queries))
	for i, q := range queries {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		norm, err := vectorNorm(q)
		if err != nil {
			return nil, fmt.Errorf("query %d: %w", i, err)
		}
		norms[i] = norm
	}
	ids := make([]int64, len(queries))
	var next atomic.Int64
	work := func() {
		for ctx.Err() == nil {
			i := int(next.Add(1)-1)
			if i >= len(queries) {
				break
			}
			ids[i] = g.identify(ctx, queries[i], norms[i], minScore, minMargin)
		}
	}
	count := min(workers, min(len(queries), runtime.GOMAXPROCS(0)))
	var wg sync.WaitGroup
	for i := 1; i < count; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			work()
		}()
	}
	work()
	wg.Wait()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return ids, nil
}
