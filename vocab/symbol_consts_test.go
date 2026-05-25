package vocab

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestSymbolConsts_MatchHash — every exported POS / Dep ID must equal
// StringStore.Hash(name). If symbolsByStr ever shifts, this test fails
// loudly so the constants get updated rather than silently diverging.
// Also locks the spacy.symbols parity claim: the integers here are the
// same integers Python's spaCy uses.
func TestSymbolConsts_MatchHash(t *testing.T) {
	ss := NewStringStore()
	cases := []struct {
		name string
		want uint64
	}{
		{"ADJ", POSAdj}, {"ADP", POSAdp}, {"ADV", POSAdv}, {"AUX", POSAux},
		{"CONJ", POSConj}, {"CCONJ", POSCConj}, {"DET", POSDet}, {"INTJ", POSIntj},
		{"NOUN", POSNoun}, {"NUM", POSNum}, {"PART", POSPart}, {"PRON", POSPron},
		{"PROPN", POSPropn}, {"PUNCT", POSPunct}, {"SCONJ", POSSConj}, {"SYM", POSSym},
		{"VERB", POSVerb}, {"X", POSX}, {"EOL", POSEol}, {"SPACE", POSSpace},

		{"acomp", DepAcomp}, {"advcl", DepAdvcl}, {"advmod", DepAdvmod},
		{"agent", DepAgent}, {"amod", DepAmod}, {"appos", DepAppos},
		{"attr", DepAttr}, {"aux", DepAux}, {"auxpass", DepAuxpass},
		{"cc", DepCC}, {"ccomp", DepCComp}, {"conj", DepConj},
		{"csubj", DepCSubj}, {"csubjpass", DepCSubjPass}, {"dep", DepDep},
		{"det", DepDet}, {"dobj", DepDobj}, {"expl", DepExpl},
		{"intj", DepIntj}, {"iobj", DepIobj}, {"mark", DepMark},
		{"meta", DepMeta}, {"neg", DepNeg}, {"nn", DepNn},
		{"npadvmod", DepNpadvmod}, {"nsubj", DepNsubj}, {"nsubjpass", DepNsubjPass},
		{"oprd", DepOprd}, {"obj", DepObj}, {"parataxis", DepParataxis},
		{"pcomp", DepPComp}, {"pobj", DepPobj}, {"poss", DepPoss},
		{"preconj", DepPreconj}, {"prep", DepPrep}, {"prt", DepPrt},
		{"punct", DepPunct}, {"quantmod", DepQuantmod}, {"relcl", DepRelcl},
		{"root", DepRoot}, {"xcomp", DepXComp}, {"acl", DepAcl},
	}
	for _, c := range cases {
		require.Equalf(t, c.want, ss.Hash(c.name),
			"%s constant (%d) must equal StringStore.Hash(%q)", c.name, c.want, c.name)
	}
}
