package nn

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/vmihailenco/msgpack/v5"
	"github.com/vmihailenco/msgpack/v5/msgpcode"
)

// thincPayload mirrors the top-level dict produced by thinc Model.to_bytes().
type thincPayload struct {
	Nodes  []thincNode                     `msgpack:"nodes"`
	Attrs  []map[string]msgpack.RawMessage `msgpack:"attrs"`
	Params []map[string]ndarray            `msgpack:"params"`
	Shims  []msgpack.RawMessage            `msgpack:"shims"`
}

type thincNode struct {
	Index int                           `msgpack:"index"`
	Name  string                        `msgpack:"name"`
	Dims  map[string]int                `msgpack:"dims"`
	Refs  map[string]msgpack.RawMessage `msgpack:"refs"`
}

// ndarray is the Go decoding of srsly's msgpack-numpy ext payload.
// The msgpack-numpy convention encodes numpy ndarrays as ext type with body
// being a sub-msgpack dict containing nd/type/kind/shape/data.
type ndarray struct {
	Shape []int
	Dtype string // e.g., "<f4" for little-endian float32, or "float32"
	Data  []byte
}

func (n *ndarray) DecodeMsgpack(dec *msgpack.Decoder) error {
	code, err := dec.PeekCode()
	if err != nil {
		return err
	}
	if msgpcode.IsExt(code) {
		_, extLen, err := dec.DecodeExtHeader()
		if err != nil {
			return fmt.Errorf("ndarray.DecodeMsgpack: decode numpy ext header: %w", err)
		}
		body := make([]byte, extLen)
		if err := dec.ReadFull(body); err != nil {
			return fmt.Errorf("ndarray.DecodeMsgpack: read numpy ext body: %w", err)
		}
		return n.decodeFromMap(body)
	}
	var m map[string]any
	if err := dec.Decode(&m); err != nil {
		return err
	}
	return n.fromMap(m)
}

func (n *ndarray) decodeFromMap(body []byte) error {
	var m map[string]any
	if err := msgpack.Unmarshal(body, &m); err != nil {
		return fmt.Errorf("unmarshal numpy body: %w", err)
	}
	return n.fromMap(m)
}

func (n *ndarray) fromMap(m map[string]any) error {
	rawShape, ok := m["shape"]
	if !ok {
		return fmt.Errorf("numpy payload missing 'shape'")
	}
	shape, err := toIntSlice(rawShape)
	if err != nil {
		return fmt.Errorf("decode shape: %w", err)
	}
	n.Shape = shape

	dtype, _ := m["type"].(string)
	if dtype == "" {
		dtype, _ = m["dtype"].(string)
	}
	if dtype == "" {
		dtype, _ = m["kind"].(string)
	}
	n.Dtype = dtype

	rawData, ok := m["data"]
	if !ok {
		return fmt.Errorf("numpy payload missing 'data'")
	}
	switch v := rawData.(type) {
	case []byte:
		n.Data = v
	case string:
		n.Data = []byte(v)
	default:
		return fmt.Errorf("numpy 'data' has unexpected type %T", v)
	}
	return nil
}

func toIntSlice(v any) ([]int, error) {
	switch x := v.(type) {
	case []any:
		out := make([]int, len(x))
		for i, e := range x {
			n, err := toInt(e)
			if err != nil {
				return nil, err
			}
			out[i] = n
		}
		return out, nil
	case []int64:
		out := make([]int, len(x))
		for i, e := range x {
			out[i] = int(e)
		}
		return out, nil
	case []int:
		return x, nil
	default:
		return nil, fmt.Errorf("cannot convert %T to []int", v)
	}
}

func toInt(v any) (int, error) {
	switch x := v.(type) {
	case int8:
		return int(x), nil
	case int16:
		return int(x), nil
	case int32:
		return int(x), nil
	case int64:
		return int(x), nil
	case uint8:
		return int(x), nil
	case uint16:
		return int(x), nil
	case uint32:
		return int(x), nil
	case uint64:
		return int(x), nil
	case int:
		return x, nil
	default:
		return 0, fmt.Errorf("not an int: %T", v)
	}
}

// asFloat32 decodes an ndarray into a []float32 slice (little-endian).
// Accepts dtype strings like "float32", "<f4", "f4".
func (n *ndarray) asFloat32() ([]float32, error) {
	if !isFloat32Dtype(n.Dtype) {
		return nil, fmt.Errorf("expected float32 ndarray, got dtype %q", n.Dtype)
	}
	if len(n.Data)%4 != 0 {
		return nil, fmt.Errorf("float32 data length %d not divisible by 4", len(n.Data))
	}
	out := make([]float32, len(n.Data)/4)
	for i := range out {
		bits := binary.LittleEndian.Uint32(n.Data[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

func isFloat32Dtype(d string) bool {
	switch d {
	case "float32", "<f4", "f4", "=f4":
		return true
	}
	return false
}

// FromBytes deserialises a thinc model byte payload into m and its children.
// The caller must have pre-built a Model tree whose Walk() order matches what
// was serialised. Names and dims are overwritten from the payload.
func (m *Model) FromBytes(data []byte) error {
	var p thincPayload
	if err := msgpack.Unmarshal(data, &p); err != nil {
		return fmt.Errorf("FromBytes: decode thinc payload: %w", err)
	}
	walked := m.Walk()
	if len(walked) != len(p.Nodes) {
		return fmt.Errorf("FromBytes: walk-order mismatch: tree has %d nodes, payload has %d",
			len(walked), len(p.Nodes))
	}
	for i, node := range walked {
		info := p.Nodes[i]
		if info.Name != "" {
			node.Name = info.Name
		}
		if node.Dims == nil && len(info.Dims) > 0 {
			node.Dims = map[string]int{}
		}
		for k, v := range info.Dims {
			node.Dims[k] = v
		}
		if i < len(p.Params) {
			pmap := p.Params[i]
			if len(pmap) > 0 && node.Params == nil {
				node.Params = map[string][]float32{}
			}
			for name, arr := range pmap {
				vals, err := arr.asFloat32()
				if err != nil {
					return fmt.Errorf("FromBytes: param %q on node %q (%d): %w", name, node.Name, i, err)
				}
				node.Params[name] = vals
			}
		}
	}
	return nil
}
