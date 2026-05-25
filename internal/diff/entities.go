package diff

// EntityReport summarises NER disagreement plus standard P/R/F1 (exact-span match).
type EntityReport struct {
	Missing   []Entity // in want but not in got (false negatives)
	Spurious  []Entity // in got but not in want (false positives)
	Precision float64
	Recall    float64
	F1        float64
}

// Equal reports whether the comparison found no missing or spurious entities
// (precision = recall = F1 = 1.0).
func (r EntityReport) Equal() bool {
	return len(r.Missing) == 0 && len(r.Spurious) == 0
}

// CompareEntities diffs two entity-span slices using exact match on
// (Start, End, Label). Ignores the Text field (which is derivable).
//
// Definitions:
//
//	Precision = TP / (TP + FP); if no predictions, P = 1.0 (avoid NaN)
//	Recall    = TP / (TP + FN); if no gold,         R = 1.0
//	F1        = 2PR/(P+R); if both 0, F1 = 0
func CompareEntities(want, got []Entity) EntityReport {
	var r EntityReport
	wantSet := make(map[Entity]bool, len(want))
	for _, e := range want {
		wantSet[keyOf(e)] = true
	}
	gotSet := make(map[Entity]bool, len(got))
	for _, e := range got {
		gotSet[keyOf(e)] = true
	}
	tp := 0
	for k := range gotSet {
		if wantSet[k] {
			tp++
		} else {
			r.Spurious = append(r.Spurious, k)
		}
	}
	for k := range wantSet {
		if !gotSet[k] {
			r.Missing = append(r.Missing, k)
		}
	}
	if len(gotSet) == 0 {
		r.Precision = 1.0
	} else {
		r.Precision = float64(tp) / float64(len(gotSet))
	}
	if len(wantSet) == 0 {
		r.Recall = 1.0
	} else {
		r.Recall = float64(tp) / float64(len(wantSet))
	}
	if r.Precision+r.Recall == 0 {
		r.F1 = 0
	} else {
		r.F1 = 2 * r.Precision * r.Recall / (r.Precision + r.Recall)
	}
	return r
}

// keyOf returns a comparable Entity (Text field zeroed so dedup uses span+label).
func keyOf(e Entity) Entity {
	return Entity{Start: e.Start, End: e.End, Label: e.Label}
}
