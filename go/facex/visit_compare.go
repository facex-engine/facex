package facex

import (
	"encoding/csv"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
)

const (
	visitUnknown int64 = -1
	visitReview  int64 = -2
)

type VisitOutcomes struct {
	CorrectID       int `json:"correct_id"`
	WrongKnownID    int `json:"wrong_known_id"`
	UnknownGranted  int `json:"unknown_granted"`
	KnownRejected   int `json:"known_rejected"`
	UnknownRejected int `json:"unknown_rejected"`
	KnownReviewed   int `json:"known_reviewed"`
	UnknownReviewed int `json:"unknown_reviewed"`
}

type VisitComparison struct {
	Visits            int           `json:"visits"`
	Known             int           `json:"known"`
	Unknown           int           `json:"unknown"`
	Before            VisitOutcomes `json:"before"`
	After             VisitOutcomes `json:"after"`
	WrongRemoved      int           `json:"wrong_removed"`
	WrongIntroduced   int           `json:"wrong_introduced"`
	WrongBoth         int           `json:"wrong_both"`
	CorrectIDToReview int           `json:"correct_id_to_review"`
	// Keys describe the old outcome, then the new outcome. Absent pairs have count zero.
	Transitions map[string]map[string]int `json:"transitions"`
}

func (o *VisitOutcomes) record(expected, actual int64) (outcome string, wrong bool) {
	switch {
	case actual == visitReview && expected == visitUnknown:
		o.UnknownReviewed++
		return "unknown_reviewed", false
	case actual == visitReview:
		o.KnownReviewed++
		return "known_reviewed", false
	case actual == visitUnknown && expected == visitUnknown:
		o.UnknownRejected++
		return "unknown_rejected", false
	case actual == visitUnknown:
		o.KnownRejected++
		return "known_rejected", false
	case expected == visitUnknown:
		o.UnknownGranted++
		return "unknown_granted", true
	case actual != expected:
		o.WrongKnownID++
		return "wrong_known_id", true
	default:
		o.CorrectID++
		return "correct_id", false
	}
}

func visitLogValue(value string, roster bool) (int64, error) {
	if roster && value == "unknown" || !roster && value == "reject" {
		return visitUnknown, nil
	}
	if !roster && value == "review" {
		return visitReview, nil
	}
	id, err := strconv.ParseInt(value, 10, 64)
	if err != nil || id < 0 {
		return 0, fmt.Errorf("invalid ID or status %q", value)
	}
	return id, nil
}

func readVisitLog(r io.Reader, role string) (map[string]int64, error) {
	if r == nil {
		return nil, fmt.Errorf("facex: missing %s input", role)
	}
	roster := role == "roster"
	header := []string{"visit_id", "decision"}
	if roster {
		header[1] = "expected_id"
	}
	reader := csv.NewReader(r)
	fields, err := reader.Read()
	if err != nil || !slices.Equal(fields, header) {
		return nil, fmt.Errorf("facex: %s requires header %s", role, strings.Join(header, ","))
	}
	values := make(map[string]int64)
	for line := 2; ; line++ {
		fields, err = reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("%s record %d: %w", role, line, err)
		}
		key := fields[0]
		if key == "" || key != strings.TrimSpace(key) {
			return nil, fmt.Errorf("%s record %d: invalid visit ID", role, line)
		}
		if _, exists := values[key]; exists {
			return nil, fmt.Errorf("%s record %d: repeated visit %q", role, line, key)
		}
		id, err := visitLogValue(fields[1], roster)
		if err != nil {
			return nil, fmt.Errorf("%s record %d: %w", role, line, err)
		}
		values[key] = id
	}
	return values, nil
}

// CompareVisitLogs joins independent truth and two policy exports by visit_id.
// A wrong outcome is an automatic grant to the wrong or an unenrolled person.
// A rejected or reviewed visit never earns credit as a correct identification.
func CompareVisitLogs(roster, before, after io.Reader) (VisitComparison, error) {
	truth, e := readVisitLog(roster, "roster")
	if e != nil {
		return VisitComparison{}, e
	}
	if len(truth) == 0 {
		return VisitComparison{}, fmt.Errorf("facex: empty roster")
	}
	old, e := readVisitLog(before, "before")
	if e != nil {
		return VisitComparison{}, e
	}
	next, e := readVisitLog(after, "after")
	if e != nil {
		return VisitComparison{}, e
	}
	out := VisitComparison{Visits: len(truth), Transitions: make(map[string]map[string]int)}
	for key, expected := range truth {
		a, ok := old[key]
		if !ok {
			return VisitComparison{}, fmt.Errorf("facex: before missing visit %q", key)
		}
		b, ok := next[key]
		if !ok {
			return VisitComparison{}, fmt.Errorf("facex: after missing visit %q", key)
		}
		if expected == visitUnknown {
			out.Unknown++
		} else {
			out.Known++
		}
		from, wasWrong := out.Before.record(expected, a)
		to, isWrong := out.After.record(expected, b)
		if out.Transitions[from] == nil {
			out.Transitions[from] = make(map[string]int)
		}
		out.Transitions[from][to]++
		switch {
		case wasWrong && isWrong:
			out.WrongBoth++
		case wasWrong:
			out.WrongRemoved++
		case isWrong:
			out.WrongIntroduced++
		}
		if from == "correct_id" && to == "known_reviewed" {
			out.CorrectIDToReview++
		}
	}
	if len(old) != len(truth) || len(next) != len(truth) {
		return VisitComparison{}, fmt.Errorf("facex: policy log contains a visit outside the roster")
	}
	return out, nil
}
