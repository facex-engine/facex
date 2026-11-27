package facex

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"
)

// The helper speaks the real fixed-size wire protocol. A file marker lets
// tests cancel only after a partial reply has actually reached the pipe.
func TestMain(m *testing.M) {
	if os.Getenv("FACEX_POOL_TEST_CHILD") == "1" && len(os.Args) == 3 && os.Args[2] == "--server" {
		runPoolChild()
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func runPoolChild() {
	input, output := make([]byte, InputBytes), make([]byte, OutputBytes)
	for {
		if _, err := io.ReadFull(os.Stdin, input); err != nil {
			return
		}
		switch math.Float32frombits(binary.LittleEndian.Uint32(input)) {
		case -1:
			_, _ = os.Stdout.Write(output[:13])
			if err := os.WriteFile("partial", []byte("ready"), 0600); err != nil {
				os.Exit(2)
			}
			time.Sleep(time.Minute)
			return
		case -2:
			_, _ = os.Stdout.Write(output[:13])
			return
		case -3:
			_, _ = os.Stderr.Write(bytes.Repeat([]byte{'x'}, 256<<10))
		}
		for j := 0; j < OutputSize; j++ {
			copy(output[4*j:4*j+4], input[4*j:4*j+4])
		}
		if _, err := os.Stdout.Write(output); err != nil {
			return
		}
	}
}

func poolInput(v float32) []float32 {
	data := make([]float32, InputSize)
	for i := range data {
		data[i] = v
	}
	return data
}

func newHelperPool(t *testing.T, workers, waiting int) (*EmbedPool, string) {
	t.Helper()
	t.Setenv("FACEX_POOL_TEST_CHILD", "1")
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	p, err := NewEmbedPool(Config{Exe: exe, Workdir: dir}, workers, waiting)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(p.Close)
	return p, dir
}

func waitPool(t *testing.T, ready func() bool) {
	t.Helper()
	limit := time.Now().Add(10 * time.Second)
	for !ready() {
		if time.Now().After(limit) {
			t.Fatal("timed out waiting for worker state")
		}
		time.Sleep(time.Millisecond)
	}
}

func TestPoolInterrupt(t *testing.T) {
	for _, shutdown := range []bool{false, true} {
		t.Run(fmt.Sprintf("shutdown=%t", shutdown), func(t *testing.T) {
			p, dir := newHelperPool(t, 1, 1)
			active, cancelActive := context.WithCancel(context.Background())
			defer cancelActive()
			busy := make(chan error, 1)
			go func() {
				_, err := p.Embed(active, poolInput(-1))
				busy <- err
			}()
			waitPool(t, func() bool {
				_, err := os.Stat(filepath.Join(dir, "partial"))
				return err == nil
			})
			waiting, cancelWaiting := context.WithCancel(context.Background())
			defer cancelWaiting()
			queued := make(chan error, 1)
			go func() {
				_, err := p.Embed(waiting, poolInput(7))
				queued <- err
			}()
			waitPool(t, func() bool { return len(p.permits) == 2 })
			if _, err := p.Embed(context.Background(), poolInput(8)); !errors.Is(err, ErrPoolBusy) {
				t.Fatalf("overflow: %v", err)
			}
			if shutdown {
				p.Close()
				if !errors.Is(<-busy, ErrPoolClosed) || !errors.Is(<-queued, ErrPoolClosed) {
					t.Fatal("shutdown did not interrupt active and queued calls")
				}
				return
			}
			cancelWaiting()
			if !errors.Is(<-queued, context.Canceled) {
				t.Fatal("queued context was ignored")
			}
			cancelActive()
			if !errors.Is(<-busy, context.Canceled) {
				t.Fatal("active context was ignored")
			}
			input := poolInput(11)
			out, err := p.Embed(context.Background(), input)
			if err != nil || !slices.Equal(out, input[:OutputSize]) {
				t.Fatal("partial reply leaked into the replacement stream", err)
			}
		})
	}
}

func TestPoolRecoveryAndConcurrentOwnership(t *testing.T) {
	p, _ := newHelperPool(t, 2, 32)
	if _, err := p.Embed(context.Background(), poolInput(-2)); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("truncated reply: %v", err)
	}
	input := poolInput(-3)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if out, err := p.Embed(ctx, input); err != nil || !slices.Equal(out, input[:OutputSize]) {
		t.Fatal("stderr blocked the exchange", err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			input := poolInput(float32(i+1))
			out, err := p.Embed(ctx, input)
			if err != nil || !slices.Equal(out, input[:OutputSize]) {
				t.Errorf("request %d received a wrong reply: %v", i, err)
			}
		}(i)
	}
	wg.Wait()
}

func TestPoolRealEngine(t *testing.T) {
	weights := os.Getenv("FACEX_POOL_TEST_WEIGHTS")
	if weights == "" {
		t.Skip("set FACEX_POOL_TEST_WEIGHTS and FACEX_POOL_TEST_EXE to run the model check")
	}
	cfg := Config{Exe: os.Getenv("FACEX_POOL_TEST_EXE"), Weights: weights}
	serial, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer serial.Close()
	const count = 16
	inputs, want := make([][]float32, count), make([][]float32, count)
	for i := range inputs {
		input := make([]float32, InputSize)
		for j := range input {
			input[j] = float32(math.Sin(float64((j+1)*(i+1))*.001))
		}
		inputs[i] = input
		want[i], err = serial.Embed(input)
		if err != nil {
			t.Fatal(err)
		}
		if i > 0 && slices.Equal(want[0], want[i]) {
			t.Fatal("reference inputs produced identical embeddings")
		}
	}
	p, err := NewEmbedPool(cfg, 2, count)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	var wg sync.WaitGroup
	for i := range inputs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			out, err := p.Embed(context.Background(), inputs[i])
			if err != nil || !slices.Equal(want[i], out) {
				t.Errorf("engine request %d differs from the serial reference: %v", i, err)
			}
		}(i)
	}
	wg.Wait()
}
