//! Model export to PyTorch (.pt), ONNX (.onnx), and GGUF (.gguf).
//!
//! All three formats are produced from a `neuralcabin_engine::nn::Model`'s
//! linear weights + activations. We avoid pulling in PyTorch/ONNX runtime as
//! dependencies — each writer constructs the file format bytes directly.
//!
//! Feed-forward and next-token networks share the same internal representation
//! (a stack of Linear + Activation layers), so the exporters work on either.
//! GGUF is conventionally an LLM format, so we only enable it for next-token
//! networks in the UI; this module accepts both regardless.

use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Cursor, Write};
use zip::write::SimpleFileOptions;

use neuralcabin_engine::activations::Activation;
use neuralcabin_engine::nn::{Layer, Model};

use crate::models::Network;

pub const FORMATS: &[&str] = &["pytorch", "onnx", "gguf"];

/// Returns the recommended file extension for an export format.
pub fn extension_for(format: &str) -> &'static str {
    match format {
        "pytorch" => "pt",
        "onnx"    => "onnx",
        "gguf"    => "gguf",
        _ => "bin",
    }
}

/// Dispatch to the right writer. Each writer returns the file bytes.
pub fn export_model(format: &str, net: &Network, model: &Model) -> Result<Vec<u8>, String> {
    match format {
        "pytorch" => export_pytorch(net, model),
        "onnx"    => export_onnx(net, model),
        "gguf"    => export_gguf(net, model),
        other     => Err(format!("unknown export format '{other}'")),
    }
}

// ─── PyTorch (.pt) ─────────────────────────────────────────────────────────
//
// A `.pt` file written by `torch.save(state_dict, path)` is a ZIP archive
// containing:
//   archive/data.pkl       - pickled state_dict referencing storages
//   archive/data/N         - raw little-endian float32 bytes for tensor N
//   archive/version        - text "3"
//
// We construct a minimal pickle (protocol 2) that builds an `OrderedDict`
// with keys "layers.{i}.weight" / "layers.{i}.bias" mapping to rebuilt
// torch tensors via `torch._utils._rebuild_tensor_v2`.
//
// Loading: `state_dict = torch.load("model.pt", weights_only=False)` — and
// then a simple `torch.nn.Sequential` of `nn.Linear` + activations can load
// the entries (or you can build a model yourself from the keys).

fn export_pytorch(_net: &Network, model: &Model) -> Result<Vec<u8>, String> {
    // Gather linear (weight, bias) tensors in layer order.
    let mut linears: Vec<(&[f32], usize, usize, &[f32])> = Vec::new();
    for layer in &model.layers {
        if let Layer::Linear(l) = layer {
            // PyTorch nn.Linear weights are (out_dim, in_dim) — transpose ours.
            linears.push((&l.w.data, l.in_dim, l.out_dim, &l.b.data));
        }
    }

    let pickle = build_pytorch_pickle(&linears)?;

    // Build zip archive.
    let buf = Cursor::new(Vec::<u8>::new());
    let mut zw = zip::ZipWriter::new(buf);
    let opts = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(false);

    zw.start_file("archive/version", opts)
        .map_err(|e| format!("zip version: {e}"))?;
    zw.write_all(b"3\n").map_err(|e| format!("write version: {e}"))?;

    zw.start_file("archive/data.pkl", opts)
        .map_err(|e| format!("zip pkl: {e}"))?;
    zw.write_all(&pickle).map_err(|e| format!("write pkl: {e}"))?;

    for (i, (w, in_dim, out_dim, b)) in linears.iter().enumerate() {
        let w_idx = i * 2;
        let b_idx = i * 2 + 1;

        // Weight storage: transposed to (out_dim, in_dim).
        let mut w_t = Vec::with_capacity(in_dim * out_dim);
        for r in 0..*out_dim {
            for c in 0..*in_dim {
                w_t.push(w[c * out_dim + r]);
            }
        }
        zw.start_file(&format!("archive/data/{w_idx}"), opts)
            .map_err(|e| format!("zip w{i}: {e}"))?;
        write_f32_le(&mut zw, &w_t)?;

        zw.start_file(&format!("archive/data/{b_idx}"), opts)
            .map_err(|e| format!("zip b{i}: {e}"))?;
        write_f32_le(&mut zw, b)?;
    }

    let cursor = zw.finish().map_err(|e| format!("zip finish: {e}"))?;
    Ok(cursor.into_inner())
}

fn write_f32_le<W: Write>(w: &mut W, data: &[f32]) -> Result<(), String> {
    for &v in data {
        w.write_f32::<LittleEndian>(v).map_err(|e| format!("write f32: {e}"))?;
    }
    Ok(())
}

/// Build a pickle (protocol 2) that, when unpickled by torch, yields a
/// `collections.OrderedDict` of tensor entries. Each tensor is reconstructed
/// via `torch._utils._rebuild_tensor_v2(storage, storage_offset, size,
/// stride, requires_grad, backward_hooks)`.
///
/// Storages are referenced by `persistent_id` strings — torch's persistent
/// loader resolves them to file entries inside the zip (the `archive/data/N`
/// files we wrote alongside this pickle).
fn build_pytorch_pickle(
    linears: &[(&[f32], usize, usize, &[f32])],
) -> Result<Vec<u8>, String> {
    let mut p = PickleWriter::new();
    p.proto2();
    p.empty_dict();
    p.mark();
    for (i, (_w, in_dim, out_dim, _b)) in linears.iter().enumerate() {
        let w_idx = i * 2;
        let b_idx = i * 2 + 1;
        let key_w = format!("layers.{i}.weight");
        let key_b = format!("layers.{i}.bias");
        let w_len = in_dim * out_dim;
        let b_len = *out_dim;

        // weight: shape (out_dim, in_dim), stride (in_dim, 1)
        p.short_binunicode(&key_w);
        p.rebuild_tensor(w_idx, vec![*out_dim, *in_dim], vec![*in_dim, 1], w_len);

        p.short_binunicode(&key_b);
        p.rebuild_tensor(b_idx, vec![b_len], vec![1], b_len);
    }
    p.setitems();
    p.stop();
    Ok(p.into_bytes())
}

/// Minimal pickle (protocol 2) writer for the exact shapes we need.
struct PickleWriter { buf: Vec<u8>, memo_counter: u32 }

impl PickleWriter {
    fn new() -> Self { Self { buf: Vec::new(), memo_counter: 0 } }
    fn into_bytes(self) -> Vec<u8> { self.buf }

    fn proto2(&mut self) { self.buf.push(0x80); self.buf.push(0x02); }
    fn stop(&mut self) { self.buf.push(b'.'); }
    fn mark(&mut self) { self.buf.push(b'('); }
    fn empty_dict(&mut self) { self.buf.push(b'}'); }
    fn empty_tuple(&mut self) { self.buf.push(b')'); }
    fn tuple1(&mut self) { self.buf.push(0x85); }
    fn tuple2(&mut self) { self.buf.push(0x86); }
    fn tuple3(&mut self) { self.buf.push(0x87); }
    fn setitems(&mut self) { self.buf.push(b'u'); }
    fn reduce(&mut self) { self.buf.push(b'R'); }
    fn binint1(&mut self, v: u8) { self.buf.push(b'K'); self.buf.push(v); }
    fn binint(&mut self, v: i32) { self.buf.push(b'J'); self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn newfalse(&mut self) { self.buf.push(0x89); }

    fn memoize(&mut self) {
        // BINPUT (q) takes a 1-byte memo index; LONG_BINPUT (r) takes 4 bytes.
        // Always emit LONG_BINPUT for safety.
        self.buf.push(b'r');
        self.buf.extend_from_slice(&self.memo_counter.to_le_bytes());
        self.memo_counter += 1;
    }

    fn short_binunicode(&mut self, s: &str) {
        // SHORT_BINUNICODE: 0x8C + 1-byte length; for strings ≤ 255. Otherwise BINUNICODE.
        let bytes = s.as_bytes();
        if bytes.len() <= 255 {
            self.buf.push(0x8c);
            self.buf.push(bytes.len() as u8);
            self.buf.extend_from_slice(bytes);
        } else {
            self.buf.push(0x8d);
            self.buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(bytes);
        }
    }

    fn global(&mut self, module: &str, name: &str) {
        // GLOBAL opcode: 'c' module '\n' name '\n'
        self.buf.push(b'c');
        self.buf.extend_from_slice(module.as_bytes());
        self.buf.push(b'\n');
        self.buf.extend_from_slice(name.as_bytes());
        self.buf.push(b'\n');
    }

    /// Emit a value that is `torch._utils._rebuild_tensor_v2(storage,
    /// storage_offset, size, stride, requires_grad, backward_hooks)`.
    /// Leaves a single object on the stack.
    fn rebuild_tensor(
        &mut self,
        storage_idx: usize,
        size: Vec<usize>,
        stride: Vec<usize>,
        numel: usize,
    ) {
        // function ref
        self.global("torch._utils", "_rebuild_tensor_v2");
        // Build args tuple via MARK + items + 't'.
        self.mark();

        // arg 1: persistent storage reference. PyTorch wants a tuple
        // ('storage', FloatStorage, key, 'cpu', numel) reduced via BINPERSID.
        self.mark();
        self.short_binunicode("storage");
        self.global("torch", "FloatStorage");
        self.short_binunicode(&storage_idx.to_string());
        self.short_binunicode("cpu");
        self.binint(numel as i32);
        self.buf.push(b't'); // TUPLE → builds tuple from MARK
        self.buf.push(b'Q'); // BINPERSID → consumes tuple, leaves persistent id

        // arg 2: storage_offset (int 0)
        self.binint1(0);

        // arg 3: size tuple
        self.push_int_tuple(&size);
        // arg 4: stride tuple
        self.push_int_tuple(&stride);
        // arg 5: requires_grad (False)
        self.newfalse();
        // arg 6: backward_hooks (OrderedDict)
        self.global("collections", "OrderedDict");
        self.empty_tuple();
        self.reduce();

        // close args tuple
        self.buf.push(b't');
        self.reduce();
        // memoize result for repeatability
        self.memoize();
    }

    fn push_int_tuple(&mut self, vals: &[usize]) {
        match vals.len() {
            0 => self.empty_tuple(),
            1 => { self.push_int(vals[0]); self.tuple1(); }
            2 => { self.push_int(vals[0]); self.push_int(vals[1]); self.tuple2(); }
            3 => { self.push_int(vals[0]); self.push_int(vals[1]); self.push_int(vals[2]); self.tuple3(); }
            _ => {
                self.mark();
                for &v in vals { self.push_int(v); }
                self.buf.push(b't');
            }
        }
    }

    fn push_int(&mut self, v: usize) {
        if v <= 255 { self.binint1(v as u8); }
        else { self.binint(v as i32); }
    }
}

// ─── ONNX (.onnx) ──────────────────────────────────────────────────────────
//
// ONNX is a protobuf format. Rather than pull in `prost` or a generated
// `onnx.proto` set, we encode the protobuf wire format directly. The schema
// we target is the ONNX `ModelProto` (opset 13) with a `GraphProto` of Gemm +
// activation nodes.

fn export_onnx(net: &Network, model: &Model) -> Result<Vec<u8>, String> {
    let mut graph = ProtoBuf::new();

    // Map our layers to a sequence of node specs. Each Linear becomes a Gemm
    // node: Y = X*W^T + B (with transB=1 since we store W as (in,out)).
    let mut initializers: Vec<(String, Vec<i64>, Vec<f32>)> = Vec::new();
    let mut nodes: Vec<OnnxNode> = Vec::new();
    let mut cur_value = "input".to_string();
    let mut linear_idx = 0usize;

    for layer in &model.layers {
        match layer {
            Layer::Linear(l) => {
                let w_name = format!("layers.{linear_idx}.weight");
                let b_name = format!("layers.{linear_idx}.bias");
                // PyTorch convention: weight (out, in); we store (in, out) so
                // transpose for storage to match the typical ONNX consumer.
                let mut w_t = Vec::with_capacity(l.in_dim * l.out_dim);
                for r in 0..l.out_dim {
                    for c in 0..l.in_dim {
                        w_t.push(l.w.data[c * l.out_dim + r]);
                    }
                }
                initializers.push((w_name.clone(), vec![l.out_dim as i64, l.in_dim as i64], w_t));
                initializers.push((b_name.clone(), vec![l.out_dim as i64], l.b.data.clone()));

                let out_name = format!("h{linear_idx}");
                nodes.push(OnnxNode {
                    op_type: "Gemm".into(),
                    inputs: vec![cur_value.clone(), w_name, b_name],
                    output: out_name.clone(),
                    attrs: vec![
                        ("transB".into(), OnnxAttr::Int(1)),
                        ("alpha".into(),  OnnxAttr::Float(1.0)),
                        ("beta".into(),   OnnxAttr::Float(1.0)),
                    ],
                });
                cur_value = out_name;
                linear_idx += 1;
            }
            Layer::Activation(a) => {
                let (op, alpha): (&str, Option<f32>) = match a {
                    Activation::Identity => continue,
                    Activation::ReLU     => ("Relu", None),
                    Activation::Sigmoid  => ("Sigmoid", None),
                    Activation::Tanh     => ("Tanh", None),
                    Activation::Softmax  => ("Softmax", None),
                };
                let out_name = format!("a{}", nodes.len());
                let mut attrs = Vec::new();
                if let Some(a) = alpha { attrs.push(("alpha".into(), OnnxAttr::Float(a))); }
                if op == "Softmax" { attrs.push(("axis".into(), OnnxAttr::Int(-1))); }
                nodes.push(OnnxNode {
                    op_type: op.into(),
                    inputs: vec![cur_value.clone()],
                    output: out_name.clone(),
                    attrs,
                });
                cur_value = out_name;
            }
        }
    }

    let output_name = cur_value;

    // GraphProto fields (per onnx.proto):
    //   1: NodeProto node (repeated)
    //   2: string name
    //   5: TensorProto initializer (repeated)
    //   11: ValueInfoProto input (repeated)
    //   12: ValueInfoProto output (repeated)
    for n in &nodes {
        let mut node = ProtoBuf::new();
        for inp in &n.inputs { node.write_string(1, inp); }
        node.write_string(2, &n.output);
        node.write_string(4, &n.op_type);
        for (k, v) in &n.attrs {
            node.write_message(5, &onnx_attr(k, v));
        }
        graph.write_message(1, &node.bytes);
    }
    graph.write_string(2, &net.name);

    for (name, shape, data) in &initializers {
        graph.write_message(5, &onnx_tensor(name, shape, data));
    }

    graph.write_message(11, &onnx_value_info("input", &[-1, net.input_dim as i64]));
    graph.write_message(12, &onnx_value_info(&output_name, &[-1, net.output_dim as i64]));

    // ModelProto:
    //   1: int64 ir_version
    //   2: string producer_name
    //   3: string producer_version
    //   7: GraphProto graph
    //   8: OperatorSetIdProto opset_import (repeated)
    let mut model_proto = ProtoBuf::new();
    model_proto.write_int64(1, 8); // IR version 8
    model_proto.write_string(2, "neuralcabin");
    model_proto.write_string(3, "0.1");
    model_proto.write_message(7, &graph.bytes);

    let mut opset = ProtoBuf::new();
    opset.write_string(1, ""); // domain ai.onnx
    opset.write_int64(2, 13); // version
    model_proto.write_message(8, &opset.bytes);

    Ok(model_proto.bytes)
}

struct OnnxNode {
    op_type: String,
    inputs: Vec<String>,
    output: String,
    attrs: Vec<(String, OnnxAttr)>,
}
enum OnnxAttr { Int(i64), Float(f32) }

fn onnx_attr(name: &str, v: &OnnxAttr) -> Vec<u8> {
    // AttributeProto:
    //   1: name string
    //   2: f float
    //   3: i int64
    //   20: type AttributeType
    let mut p = ProtoBuf::new();
    p.write_string(1, name);
    match v {
        OnnxAttr::Float(f) => { p.write_float(2, *f); p.write_int64(20, 1); /* FLOAT */ }
        OnnxAttr::Int(i)   => { p.write_int64(3, *i); p.write_int64(20, 2); /* INT */ }
    }
    p.bytes
}

fn onnx_tensor(name: &str, shape: &[i64], data: &[f32]) -> Vec<u8> {
    // TensorProto:
    //   1: dims (repeated int64, packed)
    //   2: data_type int32 (1 = FLOAT)
    //   8: name string
    //   9: raw_data bytes
    let mut p = ProtoBuf::new();
    for &d in shape { p.write_int64(1, d); }
    p.write_int32(2, 1);
    p.write_string(8, name);
    let mut raw = Vec::with_capacity(data.len() * 4);
    for v in data { raw.extend_from_slice(&v.to_le_bytes()); }
    p.write_bytes(9, &raw);
    p.bytes
}

fn onnx_value_info(name: &str, shape: &[i64]) -> Vec<u8> {
    // ValueInfoProto:
    //   1: name string
    //   2: TypeProto type
    let mut p = ProtoBuf::new();
    p.write_string(1, name);

    // TypeProto:
    //   1: TypeProto.Tensor tensor_type
    let mut tp = ProtoBuf::new();
    // TypeProto.Tensor:
    //   1: int32 elem_type
    //   2: TensorShapeProto shape
    let mut tt = ProtoBuf::new();
    tt.write_int32(1, 1); // FLOAT
    let mut shp = ProtoBuf::new();
    for &d in shape {
        // Dimension { dim_value: int64 (1) OR dim_param: string }
        let mut dim = ProtoBuf::new();
        if d >= 0 { dim.write_int64(1, d); }
        else { dim.write_string(2, "batch"); }
        shp.write_message(1, &dim.bytes);
    }
    tt.write_message(2, &shp.bytes);
    tp.write_message(1, &tt.bytes);
    p.write_message(2, &tp.bytes);
    p.bytes
}

/// Minimal protobuf wire-format writer. Tag = (field_number << 3) | wire_type.
struct ProtoBuf { bytes: Vec<u8> }
impl ProtoBuf {
    fn new() -> Self { Self { bytes: Vec::new() } }
    fn write_varint(&mut self, mut v: u64) {
        while v >= 0x80 { self.bytes.push((v as u8) | 0x80); v >>= 7; }
        self.bytes.push(v as u8);
    }
    fn write_tag(&mut self, field: u32, wire: u32) {
        self.write_varint(((field as u64) << 3) | (wire as u64));
    }
    fn write_int64(&mut self, field: u32, v: i64) {
        self.write_tag(field, 0); self.write_varint(v as u64);
    }
    fn write_int32(&mut self, field: u32, v: i32) {
        self.write_tag(field, 0); self.write_varint(v as u64);
    }
    fn write_float(&mut self, field: u32, v: f32) {
        self.write_tag(field, 5);
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn write_string(&mut self, field: u32, s: &str) {
        self.write_tag(field, 2);
        self.write_varint(s.len() as u64);
        self.bytes.extend_from_slice(s.as_bytes());
    }
    fn write_bytes(&mut self, field: u32, b: &[u8]) {
        self.write_tag(field, 2);
        self.write_varint(b.len() as u64);
        self.bytes.extend_from_slice(b);
    }
    fn write_message(&mut self, field: u32, b: &[u8]) {
        self.write_tag(field, 2);
        self.write_varint(b.len() as u64);
        self.bytes.extend_from_slice(b);
    }
}

// ─── GGUF (.gguf) ──────────────────────────────────────────────────────────
//
// GGUF spec: little-endian, magic "GGUF", version 3.
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//
// Layout:
//   magic u32 = 0x46554747
//   version u32 = 3
//   tensor_count u64
//   metadata_kv_count u64
//   metadata_kv[]              (key + type + value)
//   tensor_info[]              (name + n_dim + dims + type + offset)
//   <padding to alignment>
//   tensor_data                (raw bytes, aligned)

fn export_gguf(net: &Network, model: &Model) -> Result<Vec<u8>, String> {
    const ALIGN: u64 = 32;
    const GGUF_TYPE_F32: u32 = 0;

    // Collect tensors.
    struct GgufTensor {
        name: String,
        dims: Vec<u64>,
        data: Vec<f32>,
    }
    let mut tensors: Vec<GgufTensor> = Vec::new();
    let mut linear_idx = 0usize;
    for layer in &model.layers {
        if let Layer::Linear(l) = layer {
            // GGUF/ggml dim convention is (n_in, n_out) in row-major; our
            // tensor is laid out (in, out) already. Emit dims as [in, out]
            // so importers see the same shape we documented.
            tensors.push(GgufTensor {
                name: format!("layers.{linear_idx}.weight"),
                dims: vec![l.in_dim as u64, l.out_dim as u64],
                data: l.w.data.clone(),
            });
            tensors.push(GgufTensor {
                name: format!("layers.{linear_idx}.bias"),
                dims: vec![l.out_dim as u64],
                data: l.b.data.clone(),
            });
            linear_idx += 1;
        }
    }

    // Metadata. Architecture is opaque to GGML for our custom net, but we
    // include enough information that a consumer can rebuild the model.
    let mut activations: Vec<String> = Vec::new();
    for layer in &model.layers {
        if let Layer::Activation(a) = layer {
            activations.push(a.name().to_string());
        }
    }

    let mut buf = Vec::<u8>::new();
    buf.write_u32::<LittleEndian>(0x46554747).unwrap(); // "GGUF"
    buf.write_u32::<LittleEndian>(3).unwrap();          // version

    buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();

    // Build metadata first so we can count entries.
    let mut meta = Vec::<u8>::new();
    let mut kv_count: u64 = 0;
    write_gguf_kv_string(&mut meta, "general.architecture", "neuralcabin"); kv_count += 1;
    write_gguf_kv_string(&mut meta, "general.name", &net.name); kv_count += 1;
    write_gguf_kv_string(&mut meta, "neuralcabin.kind", &net.kind); kv_count += 1;
    write_gguf_kv_u32(&mut meta, "neuralcabin.input_dim", net.input_dim as u32); kv_count += 1;
    write_gguf_kv_u32(&mut meta, "neuralcabin.output_dim", net.output_dim as u32); kv_count += 1;
    if let Some(ctx) = net.context_size {
        write_gguf_kv_u32(&mut meta, "neuralcabin.context_size", ctx as u32);
        kv_count += 1;
    }
    write_gguf_kv_string_array(&mut meta, "neuralcabin.activations", &activations);
    kv_count += 1;

    buf.write_u64::<LittleEndian>(kv_count).unwrap();
    buf.extend_from_slice(&meta);

    // Tensor info, with a placeholder for offsets — we patch them after we
    // know the start-of-data position.
    let mut info_section = Vec::<u8>::new();
    let mut offset_patch_positions: Vec<usize> = Vec::new();
    for t in &tensors {
        gguf_write_string(&mut info_section, &t.name);
        info_section.write_u32::<LittleEndian>(t.dims.len() as u32).unwrap();
        for &d in &t.dims { info_section.write_u64::<LittleEndian>(d).unwrap(); }
        info_section.write_u32::<LittleEndian>(GGUF_TYPE_F32).unwrap();
        // remember where the offset goes; fill with 0 for now.
        offset_patch_positions.push(info_section.len());
        info_section.write_u64::<LittleEndian>(0).unwrap();
    }
    let info_start_in_buf = buf.len();
    buf.extend_from_slice(&info_section);

    // Pad to ALIGN.
    while (buf.len() as u64) % ALIGN != 0 { buf.push(0); }
    let data_start = buf.len() as u64;

    // Lay out tensor data with per-tensor alignment, compute offsets.
    let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    for t in &tensors {
        while (buf.len() as u64 - data_start) % ALIGN != 0 { buf.push(0); }
        offsets.push(buf.len() as u64 - data_start);
        for v in &t.data { buf.write_f32::<LittleEndian>(*v).unwrap(); }
    }

    // Patch offsets back into the tensor-info section in `buf`.
    for (i, pos) in offset_patch_positions.iter().enumerate() {
        let abs_pos = info_start_in_buf + *pos;
        buf[abs_pos..abs_pos + 8].copy_from_slice(&offsets[i].to_le_bytes());
    }

    Ok(buf)
}

fn gguf_write_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.extend_from_slice(s.as_bytes());
}

const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;

fn write_gguf_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    gguf_write_string(buf, key);
    buf.write_u32::<LittleEndian>(GGUF_TYPE_STRING).unwrap();
    gguf_write_string(buf, value);
}

fn write_gguf_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    gguf_write_string(buf, key);
    buf.write_u32::<LittleEndian>(GGUF_TYPE_U32).unwrap();
    buf.write_u32::<LittleEndian>(value).unwrap();
}

fn write_gguf_kv_string_array(buf: &mut Vec<u8>, key: &str, values: &[String]) {
    gguf_write_string(buf, key);
    buf.write_u32::<LittleEndian>(GGUF_TYPE_ARRAY).unwrap();
    buf.write_u32::<LittleEndian>(GGUF_TYPE_STRING).unwrap();
    buf.write_u64::<LittleEndian>(values.len() as u64).unwrap();
    for v in values { gguf_write_string(buf, v); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neuralcabin_engine::activations::Activation;
    use neuralcabin_engine::nn::{LayerSpec, Model};
    use crate::models::{kinds, Network};
    use chrono::Utc;

    fn sample_model() -> (Network, Model) {
        let specs = vec![
            LayerSpec::Linear { in_dim: 4, out_dim: 8 },
            LayerSpec::Activation(Activation::ReLU),
            LayerSpec::Linear { in_dim: 8, out_dim: 2 },
        ];
        let model = Model::from_specs(4, &specs, 7);
        let net = Network {
            id: "x".into(), name: "tiny".into(),
            kind: kinds::FEEDFORWARD.into(),
            seed: 7, created_at: Utc::now(), trained: true,
            input_dim: 4, output_dim: 2,
            layers: vec![], parameter_count: model.parameter_count(),
            hidden_layers: None, context_size: None,
        };
        (net, model)
    }

    #[test]
    fn pytorch_export_produces_zip() {
        let (net, model) = sample_model();
        let bytes = export_pytorch(&net, &model).expect("export");
        assert!(bytes.starts_with(b"PK"), "pt file must be a zip archive");
    }

    #[test]
    fn onnx_export_has_protobuf_structure() {
        let (net, model) = sample_model();
        let bytes = export_onnx(&net, &model).expect("export");
        assert!(!bytes.is_empty());
        // First byte is the tag for field 1 (ir_version, varint) — 0x08.
        assert_eq!(bytes[0], 0x08, "first protobuf tag should be ir_version");
    }

    #[test]
    fn gguf_export_has_magic() {
        let (net, model) = sample_model();
        let bytes = export_gguf(&net, &model).expect("export");
        assert_eq!(&bytes[..4], b"GGUF", "must start with GGUF magic");
        // version 3, little-endian
        assert_eq!(&bytes[4..8], &3u32.to_le_bytes());
    }
}
