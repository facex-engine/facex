// Command visitcompare audits two policy decisions against a common visit roster.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/facex-engine/facex/go/facex"
)

func run(args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("visitcompare", flag.ContinueOnError)
	flags.SetOutput(stderr)
	roster := flags.String("roster", "", "CSV with visit_id,expected_id")
	before := flags.String("before", "", "baseline CSV with visit_id,decision")
	after := flags.String("after", "", "candidate CSV with visit_id,decision")
	independent := flags.Bool("independent", false, "compute paired statistics for independently sampled visits and fixed rules")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *roster == "" || *before == "" || *after == "" || flags.NArg() != 0 {
		return fmt.Errorf("supply -roster, -before and -after")
	}
	var readers [3]io.Reader
	for i, path := range []string{*roster, *before, *after} {
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		readers[i] = f
	}
	result, err := facex.CompareVisitLogs(readers[0], readers[1], readers[2])
	if err != nil {
		return err
	}
	type inference struct {
		McNemarP    float64    `json:"mcnemar_p"`
		Reduction95 [2]float64 `json:"wrong_grant_reduction_95"`
	}
	report := struct {
		facex.VisitComparison
		Inference *inference `json:"independent_visit_statistics,omitempty"`
	}{VisitComparison: result}
	if *independent {
		audit, err := facex.AuditPairedVisits(result.Visits, result.WrongRemoved, result.WrongIntroduced, .05)
		if err != nil {
			return err
		}
		report.Inference = &inference{audit.McNemarP, [2]float64{audit.ErrorReduction.Low, audit.ErrorReduction.High}}
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}
func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
