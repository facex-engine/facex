package facex

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
)

// ErrGalleryConflict means the enrollment revision changed before publication.
var ErrGalleryConflict = errors.New("facex: enrollment revision conflict")

// EnrollmentChange replaces the complete view list for ID. Set Remove with no
// views to remove the identity. Changes for one ID must occur once per update.
type EnrollmentChange struct {
	ID     int64
	Views  [][]float32
	Remove bool
}

type enrollmentVersion struct {
	gallery  IdentityGallery
	revision uint64
}

// LiveGallery publishes immutable enrollment snapshots. Its zero value is an
// empty gallery at revision zero. Do not copy a LiveGallery after first use.
type LiveGallery struct {
	writers sync.Mutex
	current atomic.Pointer[enrollmentVersion]
}

// Snapshot pins both the enrollment data and its revision with one atomic load.
// Use the same returned gallery for all stages of a request or an entire batch.
// Older snapshots keep their view of enrollment, including subsequently removed
// identities, until callers release them.
func (g *LiveGallery) Snapshot() (IdentityGallery, uint64) {
	if current := g.current.Load(); current != nil {
		return current.gallery, current.revision
	}
	return IdentityGallery{people: []identityGroup{}}, 0
}

// Apply commits all changes together if expected is the current revision.
// On any error it publishes nothing and returns the current revision. Removing
// a missing ID is a no-op. The caller owns the input and may reuse it on return.
func (g *LiveGallery) Apply(expected uint64, changes []EnrollmentChange) (uint64, error) {
	g.writers.Lock()
	defer g.writers.Unlock()

	old, revision := g.Snapshot()
	if expected != revision {
		return revision, ErrGalleryConflict
	}
	if len(changes) == 0 {
		return revision, nil
	}

	byID := make(map[int64]int, len(changes))
	replacements := make([]identityGroup, len(changes))
	for i, edit := range changes {
		if edit.ID < 0 || (edit.Remove && len(edit.Views) > 0) || (!edit.Remove && len(edit.Views) == 0) {
			return revision, fmt.Errorf("facex: invalid enrollment change %d", i)
		}
		if _, duplicate := byID[edit.ID]; duplicate {
			return revision, fmt.Errorf("facex: repeated enrollment ID %d", edit.ID)
		}
		byID[edit.ID] = i
		if edit.Remove {
			continue
		}
		p := newIdentityGroup(edit.ID, len(edit.Views))
		for j, row := range edit.Views {
			if err := p.addView(row); err != nil {
				return revision, fmt.Errorf("identity %d view %d: %w", edit.ID, j, err)
			}
		}
		p.buildCaps()
		replacements[i] = p
	}

	// The directory is private to this version. Unchanged group descriptors
	// point to immutable view, norm and cap arrays shared with older versions.
	next := make([]identityGroup, 0, len(old.people))
	changed := false
	for _, p := range old.people {
		i, edited := byID[p.id]
		if !edited {
			next = append(next, p)
			continue
		}
		changed = true
		if !changes[i].Remove {
			next = append(next, replacements[i])
		}
		delete(byID, p.id)
	}
	for i, edit := range changes {
		if _, added := byID[edit.ID]; added && !edit.Remove {
			next = append(next, replacements[i])
			changed = true
		}
	}
	if !changed {
		return revision, nil
	}
	if revision == math.MaxUint64 {
		return revision, errors.New("facex: enrollment revision exhausted")
	}
	revision++
	g.current.Store(&enrollmentVersion{gallery: IdentityGallery{people: next}, revision: revision})
	return revision, nil
}
