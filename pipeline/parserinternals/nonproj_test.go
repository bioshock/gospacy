package parserinternals

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/bioshock/gospacy/v3/doc"
	"github.com/bioshock/gospacy/v3/vocab"
)

// TestDeprojectivize_NoDelimiter is the en_core_web_sm-style happy path: no
// decorated label, no relabelling. Doc is unchanged.
func TestDeprojectivize_NoDelimiter(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	nsubj := ss.Add("nsubj")
	root := ss.Add("ROOT")
	d := &doc.Doc{
		Vocab: v,
		Tokens: []doc.Token{
			{Text: "She", Head: 1, Dep: nsubj},
			{Text: "ran", Head: 1, Dep: root},
			{Text: ".", Head: 1, Dep: ss.Add("punct")},
		},
	}
	Deprojectivize(d)
	require.Equal(t, nsubj, d.Tokens[0].Dep)
	require.Equal(t, 1, d.Tokens[0].Head)
}

// TestDeprojectivize_DelimiterReattaches: synthetic case proving the split
// + BFS re-attach behaviour. Tokens: ROOT(2), advmod(1), decorated(0).
// Tok 0 has label "mark||advmod" attached to tok 2 (grandfather).
// Deprojectivize should re-attach tok 0 to tok 1 (the first BFS descendant of
// tok 2 with dep == "advmod") and rewrite Dep to "mark".
func TestDeprojectivize_DelimiterReattaches(t *testing.T) {
	v := vocab.NewVocab()
	ss := v.StringStore()
	root := ss.Add("ROOT")
	advmod := ss.Add("advmod")
	deco := ss.Add("mark||advmod")
	mark := ss.Add("mark")
	d := &doc.Doc{
		Vocab: v,
		Tokens: []doc.Token{
			{Text: "before", Head: 2, Dep: deco},
			{Text: "leaving", Head: 2, Dep: advmod},
			{Text: "ran", Head: 2, Dep: root},
		},
	}
	Deprojectivize(d)
	require.Equal(t, 1, d.Tokens[0].Head, "should re-attach to first 'advmod' child of tok 2")
	require.Equal(t, mark, d.Tokens[0].Dep, "label should be split on '||' and lhs kept")
}
