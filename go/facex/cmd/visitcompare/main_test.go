package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCLI(t *testing.T) {
	dir := t.TempDir()
	var args []string
	for _, v := range [][2]string{{"roster", "visit_id,expected_id\na,unknown\nb,8\n"}, {"before", "visit_id,decision\na,8\nb,8\n"}, {"after", "visit_id,decision\nb,8\na,review\n"}} {
		path := filepath.Join(dir, v[0]+".csv")
		if e := os.WriteFile(path, []byte(v[1]), 0600); e != nil {
			t.Fatal(e)
		}
		args = append(args, "-"+v[0], path)
	}
	for _, stats := range []bool{false, true} {
		var out, diag bytes.Buffer
		current := append([]string{}, args...)
		if stats {
			current = append(current, "-independent")
		}
		if e := run(current, &out, &diag); e != nil {
			t.Fatal(e)
		}
		var report map[string]any
		if e := json.Unmarshal(out.Bytes(), &report); e != nil {
			t.Fatal(e)
		}
		_, ok := report["independent_visit_statistics"]
		if ok != stats || report["wrong_removed"] != float64(1) {
			t.Fatal(report)
		}
	}
}
