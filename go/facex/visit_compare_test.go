package facex

import (
	"strings"
	"testing"
)

func TestVisitCompareMixedChanges(t *testing.T) {
	r := "visit_id,expected_id\na,10\nb,11\nc,unknown\nd,unknown\ne,15\nf,16\ng,unknown\nh,0\n"
	a := "visit_id,decision\na,10\nb,3\nc,8\nd,reject\ne,8\nf,review\ng,review\nh,0\n"
	b := "visit_id,decision\nh,0\ng,reject\nf,reject\ne,9\nd,7\nc,review\nb,11\na,review\n"
	got, err := CompareVisitLogs(strings.NewReader(r), strings.NewReader(a), strings.NewReader(b))
	if err != nil || got.Visits != 8 || got.Known != 5 || got.Unknown != 3 || got.WrongRemoved != 2 || got.WrongIntroduced != 1 || got.WrongBoth != 1 || got.CorrectIDToReview != 1 {
		t.Fatal(got, err)
	}
	if got.Before != (VisitOutcomes{CorrectID: 2, WrongKnownID: 2, UnknownGranted: 1, UnknownRejected: 1, KnownReviewed: 1, UnknownReviewed: 1}) {
		t.Fatal(got.Before)
	}
	if got.After != (VisitOutcomes{CorrectID: 2, WrongKnownID: 1, UnknownGranted: 1, KnownRejected: 1, UnknownRejected: 1, KnownReviewed: 1, UnknownReviewed: 1}) {
		t.Fatal(got.After)
	}
	if got.Transitions["wrong_known_id"]["correct_id"] != 1 || got.Transitions["unknown_rejected"]["unknown_granted"] != 1 {
		t.Fatal(got.Transitions)
	}
}
func TestVisitCompareFailsClosed(t *testing.T) {
	roster := "visit_id,expected_id\na,unknown\n"
	policy := "visit_id,decision\na,review\n"
	for _, p := range []string{"visit_id,decision\n", "visit_id,decision\nb,reject\n", policy + "b,1\n", policy + "a,reject\n", "visit_id,decision\na,-1\n", "visit_id,decision\na,\n", "visit_id,decision\na,review,1\n", "visit_id,decision\n a,review\n"} {
		out, e := CompareVisitLogs(strings.NewReader(roster), strings.NewReader(policy), strings.NewReader(p))
		if e == nil || out.Visits != 0 {
			t.Fatal(p, out, e)
		}
	}
	for _, r := range []string{"", "visit_id,expected_id\n", roster + "a,8\n", "visit_id,expected_id\na,reject\n"} {
		if _, e := CompareVisitLogs(strings.NewReader(r), strings.NewReader(policy), strings.NewReader(policy)); e == nil {
			t.Fatal(r)
		}
	}
}
