package en

import (
	"github.com/bioshock/gospacy/v3/tokenizer"
)

// URLMatch matches a URL as a single token.
// Ported from spaCy's url_match (spacy/lang/punctuation.py → _URL_PATTERN).
// The Python \Uxxxxxxxx Unicode escapes are inlined as literal UTF-8 chars.
const URLMatch = `^(?:(?:[\w\+\-\.]{2,})://)?(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[A-Za-z0-9¡-￿][A-Za-z0-9¡-￿_-]{0,62})?[A-Za-z0-9¡-￿]\.)+(?:[a-zａ-ｚß-öø-ÿāăąćĉċčďđēĕėęěĝğġģĥħĩīĭįıĳĵķĸĺļľŀłńņňŉŋōŏőœŕŗřśŝşšţťŧũūŭůűųŵŷźżžſƀƃƅƈƌƍƒƕƙ-ƛƞơƣƥƨƪƫƭưƴƶƹƺƽ-ƿǆǉǌǎǐǒǔǖǘǚǜǝǟǡǣǥǧǩǫǭǯǰǳǵǹǻǽǿȁȃȅȇȉȋȍȏȑȓȕȗșțȝȟȡȣȥȧȩȫȭȯȱȳ-ȹȼȿɀɂɇɉɋɍɏⱡⱥⱦⱨⱪⱬⱱⱳⱴⱶ-ⱻꜣꜥꜧꜩꜫꜭꜯ-ꜱꜳꜵꜷꜹꜻꜽꜿꝁꝃꝅꝇꝉꝋꝍꝏꝑꝓꝕꝗꝙꝛꝝꝟꝡꝣꝥꝧꝩꝫꝭꝯꝱ-ꝸꝺꝼꝿꞁꞃꞅꞇꞌꞎꞑꞓ-ꞕꞗꞙꞛꞝꞟꞡꞣꞥꞧꞩꞯꞵꞷꞹꟺꬰ-ꭚꭠ-ꭤɐ-ʯᴀ-ᴥᵫ-ᵷᵹ-ᶚḁḃḅḇḉḋḍḏḑḓḕḗḙḛḝḟḡḣḥḧḩḫḭḯḱḳḵḷḹḻḽḿṁṃṅṇṉṋṍṏṑṓṕṗṙṛṝṟṡṣṥṧṩṫṭṯṱṳṵṷṹṻṽṿẁẃẅẇẉẋẍẏẑẓẕ-ẝẟạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹỻỽỿёа-яәөүҗңһα-ωάέίόώήύа-щюяіїєґѓѕјљњќѐѝሀ-፿ঀ-৿֑-״יִ-ﭏؠ-يٮ-ەۥ-ۿݐ-ݿࢠ-ࢽﭐ-ﮱﯓ-ﴽﵐ-ﷇﷰ-ﷻﹰ-ﻼ𞸀-𞺻඀-෿ऀ-ॿಀ-೿஀-௿ఀ-౿가-힯ᄀ-ᇿ぀-ゟ゠-ヿー一-拿挀-矿砀-賿贀-鿿㐀-䶿𠀀-𡗿𡘀-𣃿𣄀-𤗿𤘀-𦃿𦄀-𧗿𧘀-𩃿𩄀-𪛟𪜀-𫜿𫝀-𫠟𫠠-𬺯𬺰-𮯯⺀-⻿⼀-⿟⿰-⿿　-〿㇀-㇯㈀-㋿㌀-㏿豈-﫿︰-﹏🈀-🋿丽-𯨟]{2,63}))(?::\d{2,5})?(?:[/?#]\S*)?$`

// MakeRules returns the compiled English tokenizer rules: prefixes + suffixes
// + infixes from punctuation.go, specials from the generated exception table.
func MakeRules() (*tokenizer.Rules, error) {
	return tokenizer.NewRules(tokenizer.RulesInput{
		Prefixes: Prefixes,
		Suffixes: Suffixes,
		Infixes:  Infixes,
		URLMatch: URLMatch,
		Specials: GeneratedExceptions,
	})
}
