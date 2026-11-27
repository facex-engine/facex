package facex

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
)

var (
	// ErrPoolBusy means all admitted work and waiting places are occupied.
	ErrPoolBusy = errors.New("facex: embedding pool busy")
	// ErrPoolClosed means pool shutdown has started.
	ErrPoolClosed = errors.New("facex: embedding pool closed")
)

// EmbedPool leases each binary stream to exactly one request at a time.
type EmbedPool struct {
	cfg     Config
	slots   chan *FaceX
	permits chan struct{}
	closed  chan struct{}
	once    sync.Once
}

// NewEmbedPool starts workers persistent processes and admits at most
// workers + maxWaiting calls. Use zero waiting places for immediate backpressure.
func NewEmbedPool(cfg Config, workers, maxWaiting int) (*EmbedPool, error) {
	if workers < 1 || maxWaiting < 0 || maxWaiting > math.MaxInt-workers {
		return nil, errors.New("facex: invalid embedding pool capacity")
	}
	p := &EmbedPool{
		cfg: cfg, slots: make(chan *FaceX, workers),
		permits: make(chan struct{}, workers+maxWaiting), closed: make(chan struct{}),
	}
	for i := 0; i < workers; i++ {
		fx, err := New(cfg)
		if err != nil {
			for j := 0; j < i; j++ {
				worker := <-p.slots
				stopWorker(worker)
				_ = worker.Close()
			}
			return nil, err
		}
		p.slots <- fx
	}
	return p, nil
}

// Close both descriptors to unblock either half of an in-flight exchange.
// The lease owner calls Wait through FaceX.Close after Embed has returned.
func stopWorker(fx *FaceX) {
	_ = fx.cmd.Process.Kill()
	_ = fx.stdin.Close()
	_ = fx.stdout.Close()
}

// Embed returns one 512-component embedding. An interrupted or broken stream
// is discarded before its slot can serve another caller. Failed requests are
// returned to the caller; a later request starts a fresh process in that slot.
func (p *EmbedPool) Embed(ctx context.Context, input []float32) ([]float32, error) {
	if ctx == nil || len(input) != InputSize {
		return nil, fmt.Errorf("facex: expected a context and %d input floats", InputSize)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case <-p.closed:
		return nil, ErrPoolClosed
	default:
	}
	select {
	case p.permits <- struct{}{}:
		defer func() { <-p.permits }()
	default:
		return nil, ErrPoolBusy
	}
	var fx *FaceX
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-p.closed:
		return nil, ErrPoolClosed
	case fx = <-p.slots:
	}
	defer func() { p.slots <- fx }()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case <-p.closed:
		return nil, ErrPoolClosed
	default:
	}
	if fx == nil {
		var err error
		fx, err = New(p.cfg)
		if err != nil {
			return nil, err
		}
	}

	type reply struct {
		vector []float32
		err    error
	}
	finished := make(chan reply, 1)
	go func(worker *FaceX) {
		v, err := worker.Embed(input)
		finished <- reply{v, err}
	}(fx)

	var failure error
	select {
	case result := <-finished:
		if result.err == nil {
			return result.vector, nil
		}
		failure = result.err
		stopWorker(fx)
	case <-ctx.Done():
		failure = ctx.Err()
		stopWorker(fx)
		<-finished
	case <-p.closed:
		failure = ErrPoolClosed
		stopWorker(fx)
		<-finished
	}
	_ = fx.Close()
	fx = nil
	return nil, failure
}

// Close interrupts active requests, wakes waiters and reaps every subprocess.
// Multiple callers can close the pool concurrently.
func (p *EmbedPool) Close() {
	p.once.Do(func() {
		close(p.closed)
		for i := 0; i < cap(p.slots); i++ {
			if fx := <-p.slots; fx != nil {
				stopWorker(fx)
				_ = fx.Close()
			}
		}
	})
}
