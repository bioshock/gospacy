// Package config implements a small parser for spaCy's config.cfg files
// (a subset of confection/thinc.config). It supports:
//   - [section] and nested [a.b.c] headers
//   - key = value pairs (int, float, bool, JSON-quoted string, JSON list)
//   - @registry-prefixed keys, kept verbatim (treated as strings)
//   - ${section.key} and ${section:key} interpolation (resolved post-parse,
//     up to 16 passes; fails loud on unresolved or cyclic refs)
package config

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// Config is a nested map keyed by dot-separated section paths.
// Internally, every leaf value is one of: int64, float64, bool, string, []any.
type Config struct {
	flat map[string]any
}

// Parse parses a config.cfg byte slice and resolves all ${...} interpolation
// references. Fails loud on unresolved or cyclic references.
func Parse(data []byte) (*Config, error) {
	cfg := &Config{flat: make(map[string]any)}
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 1024*1024), 16*1024*1024)
	currentSection := ""
	lineno := 0
	for sc.Scan() {
		lineno++
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") {
			if !strings.HasSuffix(line, "]") {
				return nil, fmt.Errorf("Parse: line %d: malformed section header: %q", lineno, line)
			}
			currentSection = strings.TrimSpace(line[1 : len(line)-1])
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq < 0 {
			return nil, fmt.Errorf("Parse: line %d: expected `key = value`, got %q", lineno, line)
		}
		key := strings.TrimSpace(line[:eq])
		rawVal := strings.TrimSpace(line[eq+1:])
		if key == "" {
			return nil, fmt.Errorf("Parse: line %d: empty key", lineno)
		}
		val, err := parseValue(rawVal)
		if err != nil {
			return nil, fmt.Errorf("Parse: line %d: %w", lineno, err)
		}
		fullKey := key
		if currentSection != "" {
			fullKey = currentSection + "." + key
		}
		cfg.flat[fullKey] = val
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("Parse: scanner: %w", err)
	}
	if err := cfg.resolveInterpolations(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// resolveInterpolations expands `${section.key}` and `${section:key}` references
// in every string value. Iterates up to 16 passes to support transitive refs
// (a → b → c). Fails loud on unresolved or cyclic references.
func (c *Config) resolveInterpolations() error {
	for pass := 0; pass < 16; pass++ {
		changed := false
		for k, v := range c.flat {
			s, ok := v.(string)
			if !ok {
				continue
			}
			if !strings.Contains(s, "${") {
				continue
			}
			expanded, replaced, err := c.expandRefs(s)
			if err != nil {
				return err
			}
			if replaced {
				// If the expanded value is exactly one ${...} (no surrounding
				// chars) preserve the referenced type. Otherwise treat as
				// string (concatenation defaults to string).
				if v2, isFull := c.fullReplacement(s, expanded); isFull {
					c.flat[k] = v2
				} else {
					c.flat[k] = expanded
				}
				changed = true
			}
		}
		if !changed {
			return nil
		}
	}
	return fmt.Errorf("Parse: ${...} interpolation did not converge after 16 passes (cycle?)")
}

// expandRefs replaces every ${...} in s with the looked-up flat value
// converted to string. Returns the expanded string and whether any
// replacement happened. Fails when a referenced key is absent.
func (c *Config) expandRefs(s string) (string, bool, error) {
	out := s
	replaced := false
	for {
		i := strings.Index(out, "${")
		if i < 0 {
			return out, replaced, nil
		}
		j := strings.Index(out[i:], "}")
		if j < 0 {
			return "", false, fmt.Errorf("Parse: unterminated ${ in %q", s)
		}
		j += i
		ref := out[i+2 : j]
		// spaCy uses both ${a.b} and ${a:b}. Normalise the LAST colon to a dot.
		key := strings.Replace(ref, ":", ".", -1)
		val, ok := c.flat[key]
		if !ok {
			return "", false, fmt.Errorf("Parse: unresolved reference ${%s}", ref)
		}
		out = out[:i] + valToString(val) + out[j+1:]
		replaced = true
	}
}

// fullReplacement returns the typed value when src is exactly "${ref}" and the
// referenced flat entry is non-string. Lets `width = ${a:x}` carry the int64
// through, not the string "96".
func (c *Config) fullReplacement(src, expanded string) (any, bool) {
	if !strings.HasPrefix(src, "${") || !strings.HasSuffix(src, "}") {
		return nil, false
	}
	if strings.Count(src, "${") != 1 {
		return nil, false
	}
	ref := src[2 : len(src)-1]
	key := strings.Replace(ref, ":", ".", -1)
	v, ok := c.flat[key]
	if !ok {
		return nil, false
	}
	// Don't bother for string targets — expanded already is the string.
	if _, isStr := v.(string); isStr {
		return nil, false
	}
	_ = expanded
	return v, true
}

func valToString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case int64:
		return strconv.FormatInt(x, 10)
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	case bool:
		if x {
			return "true"
		}
		return "false"
	case nil:
		return ""
	}
	return fmt.Sprintf("%v", v)
}

func parseValue(raw string) (any, error) {
	if raw == "" {
		return "", nil
	}
	if strings.HasPrefix(raw, `"`) && strings.HasSuffix(raw, `"`) {
		var s string
		if err := json.Unmarshal([]byte(raw), &s); err != nil {
			return nil, fmt.Errorf("parseValue: bad quoted string %q: %w", raw, err)
		}
		return s, nil
	}
	if strings.HasPrefix(raw, "[") && strings.HasSuffix(raw, "]") {
		var arr []any
		if err := json.Unmarshal([]byte(raw), &arr); err != nil {
			return nil, fmt.Errorf("parseValue: bad list %q: %w", raw, err)
		}
		return arr, nil
	}
	switch strings.ToLower(raw) {
	case "true":
		return true, nil
	case "false":
		return false, nil
	case "null", "none":
		return nil, nil
	}
	if i, err := strconv.ParseInt(raw, 10, 64); err == nil {
		return i, nil
	}
	if f, err := strconv.ParseFloat(raw, 64); err == nil {
		return f, nil
	}
	return raw, nil
}

// GetString returns the string at path or "" if missing/wrong-type.
func (c *Config) GetString(path string) string {
	v, ok := c.flat[path]
	if !ok {
		return ""
	}
	s, _ := v.(string)
	return s
}

// GetInt returns the int64 at path or 0 if missing/wrong-type.
func (c *Config) GetInt(path string) int64 {
	v, ok := c.flat[path]
	if !ok {
		return 0
	}
	switch x := v.(type) {
	case int64:
		return x
	case float64:
		return int64(x)
	}
	return 0
}

// GetFloat returns the float64 at path or 0 if missing/wrong-type.
func (c *Config) GetFloat(path string) float64 {
	v, ok := c.flat[path]
	if !ok {
		return 0
	}
	switch x := v.(type) {
	case float64:
		return x
	case int64:
		return float64(x)
	}
	return 0
}

// GetBool returns the bool at path. Missing → false.
func (c *Config) GetBool(path string) bool {
	v, _ := c.flat[path].(bool)
	return v
}

// GetList returns the list at path or nil if missing/wrong-type.
func (c *Config) GetList(path string) []any {
	v, _ := c.flat[path].([]any)
	return v
}

// Has reports whether path is present.
func (c *Config) Has(path string) bool {
	_, ok := c.flat[path]
	return ok
}

// Subkeys returns every key (dot path) under prefix (immediate children + descendants).
// Useful for iterating components: `cfg.Subkeys("components")`.
func (c *Config) Subkeys(prefix string) []string {
	pre := prefix + "."
	var out []string
	for k := range c.flat {
		if strings.HasPrefix(k, pre) {
			out = append(out, k)
		}
	}
	return out
}
