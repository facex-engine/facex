// Package facex wraps the FaceX embedding engine via its --server subprocess.
// Stdlib-only (no cgo). One binary, one weights file, stdin/stdout protocol.
//
// Usage:
//
//	fx, err := facex.New(facex.Config{
//	    Exe:     "./facex-cli",
//	    Weights: "data/edgeface_xs_fp32.bin",
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer fx.Close()
//
//	// input: HWC float32 [-1, 1], exactly 3*112*112 = 37632 floats.
//	emb, err := fx.Embed(input)  // []float32 length 512
//
// A FaceX instance serializes its calls. EmbedPool supports concurrent engines
// and context cancellation.
package facex

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
)

// Input face layout constants.
const (
	InputSize   = 3 * 112 * 112 // 37632 floats
	InputBytes  = InputSize * 4
	OutputSize  = 512
	OutputBytes = OutputSize * 4
)

// Config for starting the subprocess.
type Config struct {
	Exe     string // path to facex-cli[.exe]; auto-detected if empty
	Weights string // path to edgeface_xs_fp32.bin (default "data/edgeface_xs_fp32.bin")
	Workdir string // cwd for subprocess; default = dir of Exe
}

// FaceX is a persistent subprocess wrapping facex-cli --server.
type FaceX struct {
	inputBuf  []byte
	outputBuf [OutputBytes]byte
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	stdout    io.ReadCloser
	mu        sync.Mutex
	closed    bool
}

// defaultExe returns the right binary name for the current OS.
func defaultExe() string {
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	for _, p := range []string{"./facex-cli" + suffix, "./facex-cli"} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return "./facex-cli" + suffix
}

// New starts a subprocess and returns a FaceX instance ready to Embed.
func New(cfg Config) (*FaceX, error) {
	if cfg.Exe == "" {
		cfg.Exe = defaultExe()
	}
	if cfg.Weights == "" {
		cfg.Weights = "data/edgeface_xs_fp32.bin"
	}
	if cfg.Workdir == "" && filepath.Base(cfg.Exe) != cfg.Exe {
		exe, err := filepath.Abs(cfg.Exe)
		if err != nil {
			return nil, err
		}
		cfg.Exe = exe
	}
	if cfg.Workdir == "" {
		cfg.Workdir = filepath.Dir(cfg.Exe)
	}

	cmd := exec.Command(cfg.Exe, cfg.Weights, "--server")
	cmd.Dir = cfg.Workdir
	cmd.Env = os.Environ()

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, err
	}
	// Leave Stderr nil so engine diagnostics cannot fill an unread pipe.
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		return nil, err
	}
	return &FaceX{cmd: cmd, stdin: stdin, stdout: stdout, inputBuf: make([]byte, InputBytes)}, nil
}

// Embed runs a single forward pass. input must be exactly 3*112*112 floats
// in HWC order, range [-1, 1]. Returns 512 raw EdgeFace XS components. Use CosSim for cosine scoring.
func (fx *FaceX) Embed(input []float32) ([]float32, error) {
	if len(input) != InputSize {
		return nil, fmt.Errorf("input length %d != expected %d", len(input), InputSize)
	}
	fx.mu.Lock()
	defer fx.mu.Unlock()
	if fx.closed {
		return nil, fmt.Errorf("facex: subprocess already closed")
	}
	buf := fx.inputBuf
	for i, v := range input {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	if _, err := fx.stdin.Write(buf); err != nil {
		return nil, fmt.Errorf("stdin write: %w", err)
	}
	outBuf := fx.outputBuf[:]
	if _, err := io.ReadFull(fx.stdout, outBuf); err != nil {
		return nil, fmt.Errorf("stdout read: %w", err)
	}
	out := make([]float32, OutputSize)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(outBuf[i*4:]))
	}
	return out, nil
}

// CosSim returns cosine similarity, zero for a zero vector, and NaN for mismatched lengths.
func CosSim(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.NaN())
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

// Close terminates the subprocess and releases resources.
func (fx *FaceX) Close() error {
	fx.mu.Lock()
	defer fx.mu.Unlock()
	if fx.closed {
		return nil
	}
	fx.closed = true
	_ = fx.stdin.Close()
	return fx.cmd.Wait()
}
