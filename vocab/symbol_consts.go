package vocab

// Exported symbol IDs for hot-path comparisons. Mirrors Python's
// `spacy.symbols` module — these are the exact integer IDs spaCy itself
// uses (see SYMBOLS_BY_STR in spacy/symbols.pyx). Callers that read
// Token.POS / Token.Dep on every token avoid a per-access StringStore
// lookup by comparing the hash directly against the constant.
//
//	// Before:
//	if ss.LookupOrEmpty(tok.POS) == "NOUN" { ... }
//
//	// After:
//	if tok.POS == vocab.POSNoun { ... }
//
// The full table lives in symbolsByStr (vocab/symbols.go). Constants
// listed here are the subset commonly read by parser-driven consumers
// (segmenters, chunk extractors, dependency-rule engines). To get the
// ID for any other symbol, call StringStore.Hash(name) — for known
// symbols it short-circuits to the same integer with no map allocation.

// Universal POS tags (Token.POS). See spacy.symbols and the
// `pos_ids` table in spacy/parts_of_speech.pyx.
const (
	POSAdj   uint64 = 84
	POSAdp   uint64 = 85
	POSAdv   uint64 = 86
	POSAux   uint64 = 87
	POSConj  uint64 = 88 // deprecated alias for CCONJ in some UD versions
	POSCConj uint64 = 89
	POSDet   uint64 = 90
	POSIntj  uint64 = 91
	POSNoun  uint64 = 92
	POSNum   uint64 = 93
	POSPart  uint64 = 94
	POSPron  uint64 = 95
	POSPropn uint64 = 96
	POSPunct uint64 = 97
	POSSConj uint64 = 98
	POSSym   uint64 = 99
	POSVerb  uint64 = 100
	POSX     uint64 = 101
	POSEol   uint64 = 102
	POSSpace uint64 = 103
)

// Dependency labels (Token.Dep). See spacy/symbols.pyx. Names use the
// ClearNLP / OntoNotes convention spaCy's English parser emits (e.g.
// "nsubj", "dobj", "pobj", "conj", "cc", "amod", "ROOT" → "root").
const (
	DepAcomp     uint64 = 398
	DepAdvcl     uint64 = 399
	DepAdvmod    uint64 = 400
	DepAgent     uint64 = 401
	DepAmod      uint64 = 402
	DepAppos     uint64 = 403
	DepAttr      uint64 = 404
	DepAux       uint64 = 405
	DepAuxpass   uint64 = 406
	DepCC        uint64 = 407
	DepCComp     uint64 = 408
	DepConj      uint64 = 410
	DepCSubj     uint64 = 412
	DepCSubjPass uint64 = 413
	DepDep       uint64 = 414
	DepDet       uint64 = 415
	DepDobj      uint64 = 416
	DepExpl      uint64 = 417
	DepIntj      uint64 = 421
	DepIobj      uint64 = 422
	DepMark      uint64 = 423
	DepMeta      uint64 = 424
	DepNeg       uint64 = 425
	DepNn        uint64 = 427
	DepNpadvmod  uint64 = 428
	DepNsubj     uint64 = 429
	DepNsubjPass uint64 = 430
	DepOprd      uint64 = 433
	DepObj       uint64 = 434
	DepParataxis uint64 = 436
	DepPComp     uint64 = 438
	DepPobj      uint64 = 439
	DepPoss      uint64 = 440
	DepPreconj   uint64 = 442
	DepPrep      uint64 = 443
	DepPrt       uint64 = 444
	DepPunct     uint64 = 445
	DepQuantmod  uint64 = 446
	DepRelcl     uint64 = 447
	DepRoot      uint64 = 449
	DepXComp     uint64 = 450
	DepAcl       uint64 = 451
)
